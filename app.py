import os
import uuid
import logging
import time # 작업 시간 측정용 (선택 사항)
from pathlib import Path
from typing import Dict, Any, Optional # 타입 힌팅 추가, Optional 추가
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from dotenv import load_dotenv
import uvicorn
import shutil # 파일 이동을 위해 추가
import aiofiles # 비동기 파일 저장을 위해 추가
import re # 페이지 파싱용 정규식
from fastapi.middleware.cors import CORSMiddleware # CORS 미들웨어 임포트
import asyncio
import unicodedata # <<< 추가: 파일명 정규화용
import string # <<< 추가: 유효한 문자 확인용
from logging.handlers import RotatingFileHandler # <<< 추가: 파일 로깅 핸들러

# .env 파일 로드 (translator.py에서도 로드하지만, 앱 시작 시에도 명시적으로 로드하는 것이 안전)
load_dotenv()

# --- 로깅 설정 (파일 핸들러 추가) ---
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file_path = LOG_DIR / "app.log"

# 기본 로거 설정
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger() # 루트 로거 가져오기
logger.setLevel(logging.INFO) # 전체 로거 레벨 설정 (핸들러에서 개별 설정 가능)

# 콘솔 핸들러 (기존 유지 또는 필요시 수정)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
# 루트 로거에 핸들러가 이미 추가되어 있을 수 있으므로 중복 추가 방지
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    logger.addHandler(console_handler)

# 파일 핸들러 추가 (Rotating)
# 예: 파일당 10MB, 최대 5개 백업 파일 유지
file_handler = RotatingFileHandler(log_file_path, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO) # 파일에는 INFO 레벨 이상만 기록 (DEBUG 등 필요시 조정)
# 중복 추가 방지
if not any(isinstance(h, RotatingFileHandler) and h.baseFilename == str(log_file_path.resolve()) for h in logger.handlers):
     logger.addHandler(file_handler)

# pdf2zh 라이브러리 로거도 설정 (선택 사항, 라이브러리 로그 레벨 조정)
pdf2zh_logger = logging.getLogger('pdf2zh')
# pdf2zh_logger.setLevel(logging.DEBUG) # 필요시 DEBUG 레벨 설정
# pdf2zh 로거에도 핸들러 추가 (루트 로거 전파를 사용하면 불필요할 수 있음)
# pdf2zh_logger.addHandler(console_handler)
# pdf2zh_logger.addHandler(file_handler)
# pdf2zh_logger.propagate = False # 루트로 전파 막기

logger.info("Application starting up. Logging configured.") # 시작 로그
# --- 로깅 설정 끝 ---

# --- 페이지당 평균 번역 시간 상수 정의 ---
AVERAGE_SECONDS_PER_PAGE = 15.0 # <<< 15초로 변경
# --- 상수 정의 끝 ---

# pdf2zh 모듈 임포트
try:
    from pdf2zh.high_level import translate
    from pdf2zh.doclayout import OnnxModel, DocLayoutModel # DocLayoutModel 추가
except ImportError as e:
    logger.error(f"Failed to import from pdf2zh: {e}. Make sure pdf2zh is in the Python path.")
    def translate(*args, **kwargs): raise ImportError("pdf2zh.high_level.translate not found")
    class OnnxModel: pass # 임시 클래스
    class DocLayoutModel: pass

# --- 디렉토리 설정 ---
TEMP_DIR = Path("./temp_files")
STORAGE_DIR = Path("./storage") # <<< 영구 저장 디렉토리 정의
TEMP_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_DIR.mkdir(parents=True, exist_ok=True) # <<< 시작 시 생성
# --- 디렉토리 설정 끝 ---

# --- ONNX 모델 로드 (앱 시작 시 1회) ---
onnx_model: OnnxModel | None = None
try:
    # OnnxModel.from_pretrained()를 직접 호출하여 모델 로드
    onnx_model = OnnxModel.from_pretrained()
    logger.info("ONNX layout model loaded successfully.")
except Exception as e:
    logger.exception(f"Failed to load ONNX layout model: {e}")
    onnx_model = None
# --- ONNX 모델 로드 끝 ---

# --- 작업 상태 저장을 위한 인메모리 딕셔너리 ---
# key: job_id, value: {status, total_pages, current_page, start_time, end_time, error, output_file}
# 주의: 서버 재시작 시 초기화됨. 영구 저장이 필요하면 Redis 등 사용.
job_status: Dict[str, Dict[str, Any]] = {}
# --- 상태 저장소 끝 ---

app = FastAPI()

# --- CORS 미들웨어 추가 시작 ---
# TODO: 프로덕션 환경에서는 origins를 특정 도메인으로 제한해야 합니다.
origins = [
    "http://localhost:3000", # Next.js 기본 개발 포트
    "http://localhost",      # 경우에 따라 필요할 수 있음
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # 허용할 오리진 목록
    allow_credentials=True,
    allow_methods=["*"],    # 모든 HTTP 메소드 허용 (GET, POST 등)
    allow_headers=["*"],    # 모든 헤더 허용
)
# --- CORS 미들웨어 추가 끝 ---

@app.get("/")
async def read_root():
    return {"message": "PDF Translation API is running."}

# --- 페이지 파싱 함수 추가 ---
def parse_page_string(page_str: str | None) -> list[int] | None:
    """ 페이지 범위 문자열 (예: "1-3,5, 8")을 0-based 인덱스 리스트로 변환 """
    if not page_str:
        return None

    pages = []
    try:
        # 공백 제거 및 쉼표로 분리
        parts = page_str.replace(" ", "").split(',')
        for part in parts:
            if not part:
                continue
            if '-' in part:
                # 범위 처리 (예: "1-5")
                start_str, end_str = part.split('-')
                start = int(start_str)
                end = int(end_str)
                if start <= 0 or end <= 0 or start > end:
                    raise ValueError(f"Invalid page range: {part}")
                # 0-based 인덱스로 변환하여 추가
                pages.extend(range(start - 1, end))
            else:
                # 단일 페이지 처리 (예: "8")
                page_num = int(part)
                if page_num <= 0:
                    raise ValueError(f"Invalid page number: {part}")
                # 0-based 인덱스로 변환하여 추가
                pages.append(page_num - 1)

        # 중복 제거 및 정렬
        return sorted(list(set(pages)))
    except ValueError as e:
        logger.error(f"Invalid page string format '{page_str}': {e}")
        # 유효하지 않은 형식은 None으로 처리 (전체 페이지 번역)
        # 또는 HTTPException 발생 고려
        return None
# --- 페이지 파싱 함수 끝 ---

# --- 번역 진행률 업데이트 콜백 함수 ---
def translation_progress_callback(job_id: str, progress: Any):
    """ pdf2zh.translate 함수에서 호출될 콜백 """
    if job_id not in job_status:
        return

    # progress 객체 구조 확인 필요 (tqdm 객체로 추정)
    # 예시: progress.n (현재 진행 값), progress.total (전체 값)
    try:
        current = progress.n
        total = progress.total
        if total > 0:
            job_status[job_id]["current_page"] = current
            job_status[job_id]["total_pages"] = total
            job_status[job_id]["status"] = "Translating"

            # --- 예상 총 시간 최초 계산 (total 확인 시, 변경된 상수 사용) ---
            if not job_status[job_id].get("estimated_total_time"):
                job_status[job_id]["estimated_total_time"] = total * AVERAGE_SECONDS_PER_PAGE # <<< 15초 사용
            # --- 계산 끝 ---
            # logger.debug(f"Job {job_id} progress: {current}/{total}") # 너무 자주 로깅될 수 있음
    except Exception as e:
        # 콜백 내부 오류는 로깅만 하고 무시 (번역 자체에 영향 주지 않도록)
        logger.warning(f"Error in translation callback for job {job_id}: {e}")
# --- 콜백 함수 수정 끝 ---

# --- 파일명을 안전한 폴더명으로 변환하는 함수 ---
def sanitize_filename_for_directory(filename: str) -> str:
    """ 파일명을 폴더명으로 사용하기 안전하게 변환 (확장자 제거, 공백/특수문자 대체) """
    # 1. 확장자 제거
    name_without_ext = Path(filename).stem
    # 2. 유니코드 정규화 (NFC) - 조합 문자 등을 처리
    normalized_name = unicodedata.normalize('NFC', name_without_ext)
    # 3. 허용 문자 외에는 '_'로 대체 (알파벳, 숫자, 밑줄, 하이픈)
    valid_chars = "-_" + string.ascii_letters + string.digits
    sanitized = ''.join(c if c in valid_chars else '_' for c in normalized_name)
    # 4. 너무 긴 이름 자르기 (예: 100자)
    sanitized = sanitized[:100]
    # 5. 혹시 이름이 비거나 '_'만 남는 경우 대비
    if not sanitized or all(c == '_' for c in sanitized):
        return "pdf_file" # 기본 이름
    return sanitized
# --- 변환 함수 끝 ---

# --- Job 로그 추출 함수 ---
def extract_job_log(job_id: str, source_log_path: Path, output_log_path: Path):
    """ 중앙 로그 파일에서 특정 job_id 로그를 추출하여 별도 파일로 저장 """
    try:
        if not source_log_path.exists() and not any(LOG_DIR.glob('app.log.*')):
            logger.warning(f"Source log file or backups not found in {LOG_DIR}. Cannot extract logs for job {job_id}.")
            return

        extracted_lines = []
        log_files_to_check = sorted(LOG_DIR.glob('app.log*'), reverse=True)
        if not log_files_to_check:
             logger.warning(f"No log files found in {LOG_DIR} matching app.log*. Cannot extract logs.")
             return

        logger.info(f"Extracting logs for job {job_id} from {len(log_files_to_check)} file(s)...")

        processed_lines = set() # 중복 로그 방지 (회전 시 발생 가능)
        for log_file in log_files_to_check:
             try:
                  with open(log_file, 'r', encoding='utf-8') as f:
                      for line in f:
                          # 로그 메시지에 job_id가 포함되어 있고, 이전에 처리되지 않은 라인인지 확인
                          log_identifier = f" [Job: {job_id}] "
                          if log_identifier in line and line not in processed_lines:
                              extracted_lines.append(line)
                              processed_lines.add(line)
             except Exception as e:
                  logger.error(f"Error reading log file {log_file}: {e}")

        if extracted_lines:
            # 추출된 로그를 시간 순서대로 정렬 (로그 형식 첫 부분에 타임스탬프 가정)
            extracted_lines.sort()

            with open(output_log_path, 'w', encoding='utf-8') as f_out:
                f_out.writelines(extracted_lines)
            logger.info(f"Extracted {len(extracted_lines)} log lines for job {job_id} to {output_log_path}")
        else:
            logger.info(f"No specific log lines found for job {job_id} in checked files.")

    except Exception as e:
        logger.exception(f"Failed to extract logs for job {job_id}: {e}")
# --- 로그 추출 함수 끝 ---

# --- perform_translation_sync 함수 수정 (로깅 및 로그 저장 추가) ---
def perform_translation_sync(
    input_path: Path,
    job_id: str,
    original_filename: str,
    pages_str: str | None,
    custom_instructions: str
):
    """ 실제 번역 작업을 *동기적으로* 수행하는 함수 (로깅 강화 및 로그 저장) """
    start_time = time.time()
    estimated_total_time: float | None = None
    job_storage_path: Path | None = None # finally 블록에서 사용 위해 정의
    final_folder_name: str | None = None # 로그 파일명 위해 정의

    # <<< 로깅: 작업 시작 시 job_id 포함 >>>
    logger.info(f"[Job: {job_id}] Starting translation process for file '{original_filename}'.")

    try:
        # --- 안전한 폴더명 생성 ---
        sanitized_base_name = sanitize_filename_for_directory(original_filename)
        job_id_prefix = job_id.split('-')[0]
        final_folder_name = f"{sanitized_base_name}_{job_id_prefix}"
        job_storage_path = STORAGE_DIR / final_folder_name
        # --- 생성 끝 ---

        job_status[job_id] = {
            "status": "Starting",
            "total_pages": 0,
            "current_page": 0,
            "start_time": start_time,
            "estimated_total_time": estimated_total_time,
            "end_time": None,
            "error": None,
            "output_file": None,
            "options": {
                "pages": pages_str,
                "custom_instructions": custom_instructions,
                "original_filename": original_filename,
            }
        }

        # 모델 로드 확인
        if onnx_model is None:
            error_msg = f"[Job: {job_id}] ONNX model not loaded." # <<< job_id 추가
            logger.error(error_msg + f" Cannot perform translation.")
            job_status[job_id]["status"] = "Error"
            job_status[job_id]["error"] = error_msg
            job_status[job_id]["end_time"] = time.time()
            # 모델 로드 실패 시에도 finally 실행되도록 return 위치 조정 필요 없음
            # 단, input_path 정리 로직은 여기서 처리하면 finally 에서 중복될 수 있으니 finally로 이동
            # if input_path.exists():
            #    try: os.remove(input_path)
            #    except Exception as e: logger.error(f"[Job: {job_id}] Error removing input file on model load fail: {e}")
            raise RuntimeError(error_msg) # 에러를 발생시켜 finally 블록 실행 유도

        output_path = None
        dual_output_path = None

        # 출력 파일 경로 설정
        expected_output_filename = f"{input_path.stem}-mono.pdf"
        output_path = TEMP_DIR / expected_output_filename
        dual_output_path = TEMP_DIR / f"{input_path.stem}-dual.pdf"
        if output_path.exists(): os.remove(output_path)
        if dual_output_path.exists(): os.remove(dual_output_path)

        logger.info(f"[Job: {job_id}] Starting background translation details. Options: {job_status[job_id]['options']}")
        job_status[job_id]["status"] = "Parsing"

        # 페이지 파싱
        parsed_pages = parse_page_string(pages_str)
        logger.info(f"[Job: {job_id}] Parsed pages: {parsed_pages if parsed_pages else 'All'}")

        # 프롬프트 옵션 설정
        prompt_options = {"custom_instructions": custom_instructions} if custom_instructions else {}

        # pdf2zh.high_level.translate 호출
        logger.info(f"[Job: {job_id}] Calling pdf2zh.translate...")
        translate(
            files=[str(input_path)],
            output=str(TEMP_DIR),
            pages=parsed_pages,
            lang_in="en",
            lang_out="ko",
            service="azure-openai",
            ignore_cache=True,
            model=onnx_model,
            thread=4,
            callback=lambda p: translation_progress_callback(job_id, p),
            prompt_options=prompt_options
        )
        logger.info(f"[Job: {job_id}] pdf2zh.translate finished.")

        # 번역 성공 확인 및 파일 이동
        if output_path.exists():
            logger.info(f"[Job: {job_id}] Translation complete. Moving files to storage: {job_storage_path}")
            job_storage_path.mkdir(parents=True, exist_ok=True)

            final_input_path = job_storage_path / input_path.name
            final_mono_path = job_storage_path / output_path.name
            final_dual_path = job_storage_path / dual_output_path.name

            shutil.move(str(input_path), str(final_input_path))
            shutil.move(str(output_path), str(final_mono_path))
            if dual_output_path.exists():
                shutil.move(str(dual_output_path), str(final_dual_path))

            # 상태 업데이트
            job_status[job_id]["status"] = "Done"
            job_status[job_id]["output_file"] = str(final_mono_path)
            job_status[job_id]["final_input_path"] = str(final_input_path)
            if dual_output_path.exists():
                 job_status[job_id]["final_dual_path"] = str(final_dual_path)
            logger.info(f"[Job: {job_id}] Files moved successfully.")

        else:
            files_in_temp = list(TEMP_DIR.glob(f"{input_path.stem}*.pdf"))
            error_msg = f"Output file not found at {output_path}. Found in temp: {files_in_temp}"
            logger.error(f"[Job: {job_id}] Translation finished, but {error_msg}")
            job_status[job_id]["status"] = "Error"
            job_status[job_id]["error"] = error_msg

    except Exception as e:
        # 이미 로그된 에러 외 추가 로깅 (예: translate 함수 내부 오류)
        if job_status.get(job_id) and job_status[job_id].get("status") != "Error": # 아직 에러 상태가 아니면
             logger.exception(f"[Job: {job_id}] Unexpected error during translation: {e}")
             job_status[job_id]["status"] = "Error"
             job_status[job_id]["error"] = str(e)
        # 모델 로드 실패 시 여기서 잡힘
        elif not job_status.get(job_id): # 상태가 아예 생성 안된 경우 (모델 로드 실패)
             logger.exception(f"[Job: {job_id}] Critical error before status initialization (likely model load fail): {e}")
             # 상태를 임시로 만들어 에러 기록 (선택적)
             job_status[job_id] = {"status": "Error", "error": str(e), "start_time": start_time} 
    finally:
        # finally 블록은 try에서 에러 발생 여부와 관계없이 항상 실행됨
        current_status = job_status.get(job_id, {}).get("status", "Unknown")
        if current_status != "Unknown": # 상태가 설정되었다면 (즉, try 블록 진입 후)
             end_time = time.time()
             job_status[job_id]["end_time"] = end_time
             logger.info(f"[Job: {job_id}] Process finished with status '{current_status}' in {end_time - start_time:.2f} seconds.")

        # --- 임시 파일 정리 --- 
        # 성공 시에는 파일이 이동되었으므로 input_path만 정리하면 안됨
        # 오류 발생 시 또는 정상 종료 시에도 임시 파일이 남아있을 수 있음 (예: dual 파일)
        # 따라서 finally에서 항상 남아있는 임시 파일 삭제 시도 (이미 이동된 파일은 에러 무시)
        logger.info(f"[Job: {job_id}] Cleaning up temporary files...")
        # input 파일 (오류 시 또는 성공 시 이동 후 여기서는 삭제 시도 X)
        if input_path.exists() and job_status.get(job_id, {}).get("status") == "Error":
            try: os.remove(input_path)
            except Exception as e: logger.error(f"[Job: {job_id}] Error removing temp input file {input_path}: {e}")
        # mono 파일 (성공 시 이동, 오류 시 남아있을 수 있음)
        if output_path and output_path.exists():
            try: os.remove(output_path)
            except Exception as e: logger.error(f"[Job: {job_id}] Error removing temp mono file {output_path}: {e}")
        # dual 파일 (성공 시 이동, 오류 시 또는 생성 안될 수 있음)
        if dual_output_path and dual_output_path.exists():
            try: os.remove(dual_output_path)
            except Exception as e: logger.error(f"[Job: {job_id}] Error removing temp dual file {dual_output_path}: {e}")

        # --- 로그 추출 및 저장 --- 
        if job_storage_path and final_folder_name: # 최종 폴더 경로가 생성되었는지 확인
             output_log_filename = f"{final_folder_name}_log.txt"
             output_log_path = job_storage_path / output_log_filename
             extract_job_log(job_id, log_file_path, output_log_path)
        else:
             logger.warning(f"[Job: {job_id}] Final storage path not defined (job might have failed early), skipping log extraction.")
        # --- 로그 저장 끝 --- 
# --- perform_translation_sync 함수 끝 ---

# --- translate_pdf 엔드포인트 ---
@app.post("/api/translate", status_code=202)
async def translate_pdf(
    background_tasks: BackgroundTasks,
    pdf: UploadFile = File(...),
    pages: Optional[str] = Form(None),
    custom_instructions: str = Form('')
):
    if pdf.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF is allowed.")

    job_id = str(uuid.uuid4())
    input_filename = f"{job_id}_input.pdf"
    input_path = TEMP_DIR / input_filename
    original_filename = pdf.filename

    try:
        async with aiofiles.open(input_path, 'wb') as out_file:
            content = await pdf.read()
            await out_file.write(content)
        logger.info(f"[Job: {job_id}] Uploaded file '{original_filename}' saved to temporary path: {input_path}")
    except Exception as e:
        logger.exception(f"[Job: {job_id}] Failed to save uploaded file '{original_filename}': {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.")

    # 백그라운드 작업 시작
    background_tasks.add_task(
        perform_translation_sync,
        input_path,
        job_id,
        original_filename,
        pages,
        custom_instructions
    )

    return {"job_id": job_id, "message": "Translation job started."}
# --- translate_pdf 엔드포인트 끝 ---

# --- 상태 조회 엔드포인트 추가 ---
@app.get("/api/translate/status/{job_id}")
async def get_translation_status(job_id: str):
    status_info = job_status.get(job_id)
    if not status_info:
        # 작업 시작 전이거나 실패하여 상태가 생성되지 않았을 수 있음
        # 여기서는 404 반환 유지 (또는 다른 상태 코드 고려)
        raise HTTPException(status_code=404, detail="Job ID not found or job failed very early.")

    estimated_remaining_time_seconds: float | None = None
    current_status = status_info.get("status")
    current_page = status_info.get("current_page", 0)
    total_pages = status_info.get("total_pages", 0)
    start_time = status_info.get("start_time")
    initial_estimated_total_time = status_info.get("estimated_total_time")

    calculation_timestamp = time.time()

    if current_status == "Translating" and total_pages > 0 and start_time and initial_estimated_total_time:
        try:
            current_time = calculation_timestamp
            elapsed_time = current_time - start_time
            linear_remaining_time = max(0, initial_estimated_total_time - elapsed_time)

            if current_page > 0 and elapsed_time > 0:
                progress_pct = min(1.0, current_page / total_pages)
                current_speed_per_page = elapsed_time / current_page
                adjusted_total_time = current_speed_per_page * total_pages
                weight = progress_pct ** 2
                smoothed_total_time = initial_estimated_total_time * (1 - weight) + adjusted_total_time * weight
                corrected_remaining_time = max(0, smoothed_total_time - elapsed_time)
                estimated_remaining_time_seconds = corrected_remaining_time
            else:
                estimated_remaining_time_seconds = linear_remaining_time

        except ZeroDivisionError:
             estimated_remaining_time_seconds = max(0, initial_estimated_total_time - elapsed_time) if initial_estimated_total_time and elapsed_time else None
        except Exception as e:
            logger.warning(f"[Job: {job_id}] Error calculating estimated remaining time: {e}")
            estimated_remaining_time_seconds = None
    elif current_status == "Done":
        estimated_remaining_time_seconds = 0

    response = {
        "job_id": job_id,
        "status": current_status, # Use already fetched status
        "current_page": current_page,
        "total_pages": total_pages,
        "error": status_info.get("error"), # Use .get() for safety
        "estimated_remaining_time_seconds": estimated_remaining_time_seconds,
        "calculation_timestamp": calculation_timestamp
    }
    return response
# --- 상태 조회 엔드포인트 끝 ---

# --- 다운로드 엔드포인트 추가 ---
@app.get("/api/translate/download/{job_id}")
async def download_translated_pdf(job_id: str):
    status_info = job_status.get(job_id)
    if not status_info or status_info.get("status") != "Done" or not status_info.get("output_file"):
        raise HTTPException(status_code=404, detail="Translated file not found or job not completed.")

    final_output_file_path = Path(status_info["output_file"]) # Path 객체로 변환

    if not final_output_file_path.exists():
         logger.error(f"[Job: {job_id}] Output file path found in status ({final_output_file_path}), but file does not exist on disk.")
         # 상태는 유지하고 404 반환
         raise HTTPException(status_code=404, detail="Translated file missing from storage.")

    # 다운로드 시 파일명은 원본 파일명 기반으로 (상태에 저장된 정보 사용)
    original_filename = status_info.get("options", {}).get("original_filename", "translated_output.pdf")
    download_filename = f"{Path(original_filename).stem}_translated_ko.pdf"

    return FileResponse(
        path=str(final_output_file_path), # 문자열로 변환
        filename=download_filename,
        media_type='application/pdf'
    )
# --- 다운로드 엔드포인트 끝 ---

if __name__ == "__main__":
    # 개발 환경에서는 uvicorn 직접 실행
    # 프로덕션 환경에서는 gunicorn 등 사용 권장
    uvicorn.run(app, host="0.0.0.0", port=8000) 