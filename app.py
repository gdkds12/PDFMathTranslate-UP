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

# .env 파일 로드 (translator.py에서도 로드하지만, 앱 시작 시에도 명시적으로 로드하는 것이 안전)
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 페이지당 평균 번역 시간 상수 정의 ---
AVERAGE_SECONDS_PER_PAGE = 15.0 # <<< 15초로 변경
# --- 상수 정의 끝 ---

# pdf2zh 모듈 임포트
try:
    from pdf2zh.high_level import translate
    from pdf2zh.doclayout import OnnxModel # ModelInstance 대신 OnnxModel 임포트
except ImportError as e:
    logger.error(f"Failed to import from pdf2zh: {e}. Make sure pdf2zh is in the Python path.")
    def translate(*args, **kwargs): raise ImportError("pdf2zh.high_level.translate not found")
    class OnnxModel: pass # 임시 클래스

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

# --- perform_translation_sync 함수 수정 (백그라운드 실행 및 상태 업데이트) ---
def perform_translation_sync(
    input_path: Path,
    job_id: str,
    pages_str: str | None,
    custom_instructions: str
):
    """ 실제 번역 작업을 *동기적으로* 수행하는 함수 (파일 이동 로직 추가) """
    start_time = time.time()
    estimated_total_time: float | None = None # <<< 예상 총 시간 변수 추가

    job_status[job_id] = {
        "status": "Starting",
        "total_pages": 0,
        "current_page": 0,
        "start_time": start_time,
        "estimated_total_time": estimated_total_time, # <<< 상태에 추가
        "end_time": None,
        "error": None,
        "output_file": None,
        # 옵션 정보 저장 (디버깅 또는 추후 활용 목적)
        "options": {
            "pages": pages_str,
            "custom_instructions": custom_instructions,
        }
    }

    # 모델 로드 확인
    if onnx_model is None:
        error_msg = "ONNX model not loaded."
        logger.error(f"{error_msg} Cannot perform translation for job {job_id}.")
        job_status[job_id]["status"] = "Error"
        job_status[job_id]["error"] = error_msg
        job_status[job_id]["end_time"] = time.time()
        if input_path.exists(): os.remove(input_path)
        return

    output_path = None
    dual_output_path = None
    job_storage_path = STORAGE_DIR / job_id # 최종 저장될 job 디렉토리

    try:
        # 출력 파일 경로 설정
        expected_output_filename = f"{input_path.stem}-mono.pdf"
        output_path = TEMP_DIR / expected_output_filename
        dual_output_path = TEMP_DIR / f"{input_path.stem}-dual.pdf"
        if output_path.exists(): os.remove(output_path)
        if dual_output_path.exists(): os.remove(dual_output_path)

        logger.info(f"Starting background translation for {input_path} (Job ID: {job_id}) Options: {job_status[job_id]['options']}")
        job_status[job_id]["status"] = "Parsing"

        # 페이지 파싱
        parsed_pages = parse_page_string(pages_str)

        # --- 프롬프트 옵션 딕셔너리 (custom_instructions만 포함) ---
        prompt_options = {
            "custom_instructions": custom_instructions,
        }
        # 만약 custom_instructions도 없다면 빈 dict 전달
        if not custom_instructions:
            prompt_options = {} # 또는 None 전달 (라이브러리 처리 방식에 따라)

        # pdf2zh.high_level.translate 호출 시 새 인자(prompt_options) 추가
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
            # 가정: translate 함수에 prompt_options 인자 추가
            prompt_options=prompt_options
        )

        # 번역 성공 확인
        if output_path.exists():
            logger.info(f"Translation complete for {job_id}. Moving files to storage.")
            job_storage_path.mkdir(parents=True, exist_ok=True) # Job별 저장 폴더 생성

            # 최종 저장 경로 정의
            final_input_path = job_storage_path / input_path.name
            final_mono_path = job_storage_path / output_path.name
            final_dual_path = job_storage_path / dual_output_path.name

            # 파일 이동
            shutil.move(str(input_path), str(final_input_path))
            shutil.move(str(output_path), str(final_mono_path))
            if dual_output_path.exists():
                shutil.move(str(dual_output_path), str(final_dual_path))

            # 상태 업데이트 (최종 경로 저장)
            job_status[job_id]["status"] = "Done"
            job_status[job_id]["output_file"] = str(final_mono_path) # 최종 mono 파일 경로
            # 필요시 다른 최종 경로도 저장
            job_status[job_id]["final_input_path"] = str(final_input_path)
            if dual_output_path.exists():
                 job_status[job_id]["final_dual_path"] = str(final_dual_path)
            logger.info(f"Files for job {job_id} moved to {job_storage_path}")

        else:
            files_in_temp = list(TEMP_DIR.glob(f"{input_path.stem}*.pdf"))
            error_msg = f"Output file not found at {output_path}. Found in temp: {files_in_temp}"
            logger.error(f"Translation finished for {job_id}, but {error_msg}")
            job_status[job_id]["status"] = "Error"
            job_status[job_id]["error"] = error_msg

    except Exception as e:
        logger.exception(f"Translation failed for {job_id}: {e}")
        job_status[job_id]["status"] = "Error"
        job_status[job_id]["error"] = str(e)
    finally:
        job_status[job_id]["end_time"] = time.time()
        # --- 임시 파일 정리 (이동되지 않은 파일만) ---
        # 입력 파일은 성공 시 이동되므로 여기서 삭제하지 않음
        # if input_path.exists():
        #     try: os.remove(input_path)
        #     except Exception as e: logger.error(f"Error removing temp input file {input_path}: {e}")

        # 듀얼 파일도 성공 시 이동되므로, 이동되지 않았을 때만 (오류 발생 등) 삭제 시도
        if dual_output_path and dual_output_path.exists():
             try: os.remove(dual_output_path)
             except Exception as e: logger.error(f"Error removing temp dual file {dual_output_path}: {e}")
        # 모노 파일도 마찬가지 (오류 발생 시)
        if output_path and output_path.exists():
             try: os.remove(output_path)
             except Exception as e: logger.error(f"Error removing temp mono file {output_path}: {e}")
        # --- 임시 파일 정리 끝 ---
# --- perform_translation_sync 함수 끝 ---

# --- translate_pdf 엔드포인트 수정 (백그라운드 작업 시작 및 job_id 반환) ---
@app.post("/api/translate", status_code=202)
async def translate_pdf(
    background_tasks: BackgroundTasks,
    pdf: UploadFile = File(...),
    pages: Optional[str] = Form(None), # Optional 명시적 사용
    custom_instructions: str = Form('') # 이것만 남김
):
    if pdf.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF is allowed.")

    job_id = str(uuid.uuid4())
    input_filename = f"{job_id}_input.pdf"
    input_path = TEMP_DIR / input_filename

    try:
        async with aiofiles.open(input_path, 'wb') as out_file:
            content = await pdf.read()
            await out_file.write(content)
        logger.info(f"Uploaded file saved to: {input_path} for job {job_id}")
    except Exception as e:
        logger.exception(f"Failed to save uploaded file for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.")

    # 백그라운드 작업 시작 시 새 옵션 전달
    background_tasks.add_task(
        perform_translation_sync,
        input_path,
        job_id,
        pages,
        custom_instructions   # 이것만 전달
    )

    # 작업 ID 반환
    return {"job_id": job_id, "message": "Translation job started."}

# --- 상태 조회 엔드포인트 추가 ---
@app.get("/api/translate/status/{job_id}")
async def get_translation_status(job_id: str):
    status_info = job_status.get(job_id)
    if not status_info:
        raise HTTPException(status_code=404, detail="Job ID not found.")

    estimated_remaining_time_seconds: float | None = None
    current_status = status_info.get("status")
    current_page = status_info.get("current_page", 0)
    total_pages = status_info.get("total_pages", 0)
    start_time = status_info.get("start_time")
    initial_estimated_total_time = status_info.get("estimated_total_time")

    calculation_timestamp = time.time() # <<< 현재 타임스탬프 기록

    if current_status == "Translating" and total_pages > 0 and start_time and initial_estimated_total_time:
        try:
            current_time = calculation_timestamp # <<< 계산 시점 통일
            elapsed_time = current_time - start_time

            # 1. 초기 선형 감소 예상 시간
            linear_remaining_time = max(0, initial_estimated_total_time - elapsed_time)

            # 2. 보정 로직 (첫 페이지 완료 후부터 적용)
            if current_page > 0 and elapsed_time > 0:
                progress_pct = min(1.0, current_page / total_pages) # 진행률 (0 ~ 1)
                current_speed_per_page = elapsed_time / current_page # 현재까지의 실제 페이지당 평균 시간
                adjusted_total_time = current_speed_per_page * total_pages # 현재 속도 기반 총 예상 시간

                # 3. 초기 예상과 현재 속도 기반 예상을 블렌딩 (진행률에 따라 가중치 조절)
                # weight = progress_pct # 단순 선형 가중치
                # 좀 더 부드러운 보정을 위해 가중치 조절 (예: 제곱 사용)
                weight = progress_pct ** 2
                smoothed_total_time = initial_estimated_total_time * (1 - weight) + adjusted_total_time * weight

                # 4. 보정된 총 시간으로 최종 잔여 시간 계산
                corrected_remaining_time = max(0, smoothed_total_time - elapsed_time)
                estimated_remaining_time_seconds = corrected_remaining_time
            else:
                # 작업 시작 직후 (첫 페이지 처리 전)에는 선형 감소 시간 사용
                estimated_remaining_time_seconds = linear_remaining_time

        except ZeroDivisionError:
             # current_page가 0인 경우 등 나눗셈 오류 방지
             estimated_remaining_time_seconds = max(0, initial_estimated_total_time - elapsed_time) if initial_estimated_total_time and elapsed_time else None
        except Exception as e:
            logger.warning(f"Error calculating estimated remaining time for job {job_id}: {e}")
            estimated_remaining_time_seconds = None # 오류 시 None 반환
    elif current_status == "Done":
        estimated_remaining_time_seconds = 0

    response = {
        "job_id": job_id,
        "status": status_info["status"],
        "current_page": current_page, # <<< .get() 제거 가능 (위에서 이미 처리)
        "total_pages": total_pages,   # <<< .get() 제거 가능
        "error": status_info["error"],
        "estimated_remaining_time_seconds": estimated_remaining_time_seconds,
        "calculation_timestamp": calculation_timestamp # <<< 응답에 추가
    }

    # 작업 완료 시 다운로드 URL 제공 (이 엔드포인트에서 직접 파일을 보내지 않음)
    if status_info["status"] == "Done" and status_info["output_file"]:
        # FileResponse를 위한 별도 엔드포인트 URL을 생성하거나,
        # 여기서는 output_file 경로를 직접 전달 (보안상 좋지 않음, 임시)
        # response["download_url"] = f"/api/translate/download/{job_id}" # 예시
        pass # 다운로드 로직은 별도 구현 필요

    # 완료 또는 에러 상태 확인 후 메모리에서 상태 정보 제거 (선택 사항)
    # if status_info["status"] in ["Done", "Error"]:
    #     # background_tasks.add_task(job_status.pop, job_id, None)
    #     pass

    return response
# --- 상태 조회 엔드포인트 끝 ---

# --- (선택 사항) 다운로드 엔드포인트 추가 --- 
@app.get("/api/translate/download/{job_id}")
async def download_translated_pdf(job_id: str): # background_tasks 제거
    status_info = job_status.get(job_id)
    # --- 상태 및 파일 경로 유효성 검사 (job_status["output_file"] 사용) ---
    if not status_info or status_info["status"] != "Done" or not status_info.get("output_file"): # .get() 사용
        raise HTTPException(status_code=404, detail="Translated file not found or job not completed.")

    # 이제 output_file 키에는 STORAGE_DIR 아래의 최종 경로가 저장되어 있음
    final_output_file_path = Path(status_info["output_file"])

    if not final_output_file_path.exists():
         # job_status에는 있지만 실제 파일이 없는 경우
         job_status.pop(job_id, None)
         raise HTTPException(status_code=404, detail="Translated file not found on server storage.")
    # --- 검사 끝 ---

    # 다운로드 파일명 설정 (job_id 기반 유지 또는 개선 가능)
    download_filename = f"{final_output_file_path.stem}.pdf" # 실제 파일명 사용

    # --- 파일 삭제 로직은 이전 단계에서 제거됨 ---

    return FileResponse(
        path=final_output_file_path, # 최종 경로 사용
        filename=download_filename,
        media_type='application/pdf'
    )
# --- 다운로드 엔드포인트 끝 ---

if __name__ == "__main__":
    # 개발 환경에서는 uvicorn 직접 실행
    # 프로덕션 환경에서는 gunicorn 등 사용 권장
    uvicorn.run(app, host="0.0.0.0", port=8000) 