"use client"; // Add this directive for client-side interactivity

import React, { useState, useEffect, useRef } from 'react';
import { ArrowUpTrayIcon, DocumentArrowDownIcon, InformationCircleIcon, CheckCircleIcon, XCircleIcon, ClockIcon, QuestionMarkCircleIcon } from '@heroicons/react/24/outline'; // Import icons
import dynamic from 'next/dynamic'; // Import dynamic
import type { PdfPreviewProps } from '@/components/PdfPreview'; // Import the props type
import { pdfjs } from 'react-pdf'; // Import pdfjs here

// Dynamically import the PdfPreview component, disabling SSR and specifying props type
const PdfPreview = dynamic<PdfPreviewProps>(() => import('@/components/PdfPreview'), {
  ssr: false,
  loading: () => <p className="text-center text-gray-500 mt-4">미리보기 로딩 중...</p>, // Optional loading indicator
});

// --- 시간 포맷팅 함수 추가 ---
function formatTime(totalSeconds: number | null | undefined): string {
  if (totalSeconds === null || totalSeconds === undefined || totalSeconds < 0) {
    return '';
  }
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = Math.floor(totalSeconds % 60);
  if (minutes > 0) {
    return `약 ${minutes}분 ${seconds}초 남음`;
  } else if (seconds > 0) {
    return `약 ${seconds}초 남음`;
  }
  return '잠시만 기다려주세요...'; // 0초일 때
}
// --- 포맷팅 함수 끝 ---

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [pageRange, setPageRange] = useState<string>('');
  const [status, setStatus] = useState<string>('Ready'); // Ready, File selected, Uploading, Starting, Parsing, Translating, Done, Error
  const [errorDetail, setErrorDetail] = useState<string | null>(null);
  const [translatedFileUrl, setTranslatedFileUrl] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);

  // --- 상태 폴링 관련 상태 추가 ---
  const [jobId, setJobId] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState<number>(0);
  const [totalPages, setTotalPages] = useState<number>(0);
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null); // 폴링 인터벌 ID 저장
  const timerIntervalRef = useRef<NodeJS.Timeout | null>(null); // 타이머 인터벌 ID
  // --- 상태 폴링 끝 ---

  // --- 번역 옵션 상태 변수 제거 ---
  // const [keepTechnicalTerms, setKeepTechnicalTerms] = useState<boolean>(false);
  // const [keepEnglishNames, setKeepEnglishNames] = useState<boolean>(false);
  const [customInstructions, setCustomInstructions] = useState<string>('');
  // --- 제거 끝 ---

  // --- 예상 시간 상태 추가 (백엔드 값 + 계산 타임스탬프 + 실제 표시 값) ---
  const [backendEstimatedRemainingTime, setBackendEstimatedRemainingTime] = useState<number | null>(null);
  const [calculationTimestamp, setCalculationTimestamp] = useState<number | null>(null);
  const [displayRemainingTime, setDisplayRemainingTime] = useState<number | null>(null);
  // --- 상태 추가 끝 ---

  // --- 전역 워커 소스 설정 제거 ---
  // Set the worker source globally when the page mounts on the client
  // Use useEffect to ensure it runs only once on the client side
  // useEffect(() => {
  //   pdfjs.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.js`;
  // }, []);
  // --- 제거 끝 ---

  // --- 1초 타이머 로직을 useEffect로 변경 ---
  useEffect(() => {
    let intervalId: NodeJS.Timeout | null = null;

    // 번역 중이고, 필요한 시간 정보가 있을 때만 타이머 설정
    if (status === 'Translating' && backendEstimatedRemainingTime !== null && calculationTimestamp !== null && backendEstimatedRemainingTime >= 0) {
      const updateDisplayTime = () => {
        // calculationTimestamp와 backendEstimatedRemainingTime은 useEffect dependency로 인해 최신 값 보장
        // status가 변경되었는지 다시 확인 (effect 실행 후 상태 변경 가능성)
        if (status !== 'Translating') {
            if (intervalId) clearInterval(intervalId);
            intervalId = null;
            return;
        }
        const now = Date.now() / 1000;
        const elapsedSinceCalc = now - calculationTimestamp;
        const currentRemaining = Math.max(0, backendEstimatedRemainingTime - elapsedSinceCalc);
        setDisplayRemainingTime(currentRemaining);

        // 예상 시간이 0 이하가 되면 타이머 중지 (폴링이 Done 상태로 변경할 것임)
        if (currentRemaining <= 0) {
          if (intervalId) clearInterval(intervalId);
          intervalId = null;
        }
      };

      // 즉시 한번 실행하여 초기 값 설정
      updateDisplayTime();
      // 1초 간격 타이머 설정
      intervalId = setInterval(updateDisplayTime, 1000);
    } else {
      // 번역 중이 아니거나 시간 정보가 없으면 displayRemainingTime 초기화
      if (status === 'Done') {
         setDisplayRemainingTime(0); // Done 상태에서는 0으로 명시적 설정
      } else if (status !== 'Translating') {
         setDisplayRemainingTime(null); // Translating, Done 외 상태는 null
      }
      // Translating 상태인데 시간 정보가 없는 경우(백엔드 오류 등)는 null로 유지됨
    }

    // Cleanup 함수: 컴포넌트 언마운트 또는 dependency 변경 시 타이머 해제
    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [status, backendEstimatedRemainingTime, calculationTimestamp]); // status, 시간 정보가 변경될 때마다 effect 재실행
  // --- useEffect 로직 끝 ---

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setSelectedFile(event.target.files[0]);
      setStatus('File selected');
      setErrorDetail(null);
      setTranslatedFileUrl(null);
      setJobId(null); // 이전 작업 정보 초기화
      setCurrentPage(0);
      setTotalPages(0);
      setCustomInstructions('');

      // --- 예상 시간 관련 상태 초기화 ---
      setBackendEstimatedRemainingTime(null);
      setCalculationTimestamp(null);
      // setDisplayRemainingTime(null); // useEffect가 처리
      // stopTimer(); // 제거됨
      // --- 초기화 끝 ---

      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
    }
  };

  const handlePageRangeChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setPageRange(event.target.value);
  };

  // --- 폴링 함수 수정 ---
  const pollStatus = async (currentJobId: string) => {
    const apiUrlBase = process.env.NEXT_PUBLIC_API_URL || ''; // 환경 변수 읽기
    try {
      // 환경 변수 사용으로 수정
      const response = await fetch(`${apiUrlBase}/api/translate/status/${currentJobId}`);
      if (!response.ok) {
        if (response.status === 404) {
          console.error(`Job ${currentJobId} not found.`);
          setStatus('Error');
          setErrorDetail(`Translation job not found (ID: ${currentJobId}).`);
        } else {
          console.error(`Status polling failed: ${response.status}`);
          setStatus('Error');
          setErrorDetail(`Failed to get translation status (HTTP ${response.status}).`);
        }
        stopPolling();
        // 상태 업데이트
        setBackendEstimatedRemainingTime(null);
        setCalculationTimestamp(null);
        setIsProcessing(false);
        return;
      }

      const data = await response.json();
      const newStatus = data.status || 'Unknown';
      // --- 예상 시간 상태 업데이트 (이 부분은 동일) ---
      const newBackendTime = data.estimated_remaining_time_seconds ?? null;
      const newCalcTimestamp = data.calculation_timestamp ?? null;

      // setStatus, setCurrentPage 등은 여기에 유지
      setStatus(newStatus);
      setCurrentPage(data.current_page || 0);
      setTotalPages(data.total_pages || 0);
      setErrorDetail(data.error || null);

      // 상태 업데이트를 일괄적으로 처리 (React 18+에서는 자동 배치될 수 있음)
      setBackendEstimatedRemainingTime(newBackendTime);
      setCalculationTimestamp(newCalcTimestamp);
      // --- 업데이트 끝 ---

      if (newStatus === 'Done') {
        console.log("Translation done, stopping polling.");
        // 다운로드 URL에도 환경 변수 사용
        setTranslatedFileUrl(`${apiUrlBase}/api/translate/download/${currentJobId}`);
        stopPolling();
        // setDisplayRemainingTime(0); // useEffect가 처리
        setIsProcessing(false);
      } else if (newStatus === 'Error') {
        console.error("Translation error reported by backend:", data.error);
        stopPolling();
        // setDisplayRemainingTime(null); // useEffect가 처리
        setIsProcessing(false);
      }
      // 'Translating' 상태나 다른 상태일 때 타이머 시작/중지 로직은 useEffect가 담당

    } catch (error) {
      console.error('Error polling translation status:', error);
      setStatus('Error');
      setErrorDetail('상태 폴링 중 연결 오류 발생.');
      stopPolling();
      // 상태 업데이트
      setBackendEstimatedRemainingTime(null);
      setCalculationTimestamp(null);
      // setDisplayRemainingTime(null); // useEffect가 처리
      setIsProcessing(false);
    }
  };
  // --- 폴링 함수 끝 ---

  // --- 폴링 시작/중지 함수 ---
  const startPolling = (newJobId: string) => {
    stopPolling(); // 기존 폴링 중지
    setJobId(newJobId);
    // 즉시 한번 호출 + 주기적 호출 설정
    pollStatus(newJobId);
    pollingIntervalRef.current = setInterval(() => pollStatus(newJobId), 2000); // 2초 간격
  };

  const stopPolling = () => {
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
      console.log("Polling stopped.");
    }
  };
  // --- 폴링 시작/중지 끝 ---

  // 컴포넌트 언마운트 시 폴링 중지 (useEffect 타이머 클린업은 위에서 처리)
  useEffect(() => {
    return () => {
      stopPolling();
    };
  }, []); // 빈 dependency 배열로 마운트/언마운트 시에만 실행

  const handleTranslate = async () => {
    if (!selectedFile) {
      setStatus('Error');
      setErrorDetail('PDF 파일을 선택해주세요.');
      return;
    }

    setStatus('Uploading');
    setErrorDetail(null);
    setTranslatedFileUrl(null);
    setJobId(null);
    setCurrentPage(0);
    setTotalPages(0);
    setIsProcessing(true);

    // --- 예상 시간 관련 상태 초기화 ---
    setBackendEstimatedRemainingTime(null);
    setCalculationTimestamp(null);
    // --- 초기화 끝 ---

    const apiUrlBase = process.env.NEXT_PUBLIC_API_URL || ''; // 환경 변수 읽기
    const formData = new FormData();
    formData.append('pdf', selectedFile);
    if (pageRange) {
      formData.append('pages', pageRange);
    }
    // --- 커스텀 지침 추가 ---
    formData.append('custom_instructions', customInstructions);
    // --- 추가 끝 ---

    try {
      // API 호출 시 환경 변수 사용!
      const response = await fetch(`${apiUrlBase}/api/translate`, { // <<< 환경 변수 사용
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: `HTTP error ${response.status}` }));
        setStatus('Error');
        setErrorDetail(`Error starting translation: ${errorData.detail || 'Unknown error'}`);
        setIsProcessing(false);
        return;
      } 

      const data = await response.json();
      if (data.job_id) {
        setStatus('Starting'); // 상태를 Starting으로 변경
        startPolling(data.job_id); // 폴링 시작
      } else {
        setStatus('Error');
        setErrorDetail('Failed to get job ID from server.');
        setIsProcessing(false);
      }

    } catch (error) {
      console.error('Error starting translation job:', error);
      let errorMessage = '번역 작업을 시작하는 중 오류 발생';
      if (error instanceof TypeError && error.message.includes('fetch')) {
          errorMessage += ': 네트워크 연결을 확인하거나 CORS 설정을 확인하세요.';
      } else if (error instanceof Error) { // 에러 타입을 좀 더 명확히
         errorMessage += `: ${error.message}`;
      }
      setStatus('Error'); // 상태 업데이트 시 오류 메시지 포함
      setErrorDetail(errorMessage);
      setIsProcessing(false);
      setTranslatedFileUrl(null);
    } finally {
      // 업로드 실패 시에도 isProcessing을 false로 설정할 수 있도록 finally 사용 고려
      // 현재 로직에서는 try-catch 블록 내에서 처리 중
    }
  };

  // Helper to render status icon and text
  const renderStatus = () => {
    let icon = <QuestionMarkCircleIcon className="h-6 w-6 text-gray-400" />;
    let message = '상태 확인 중...';
    let progress = 0;
    if (totalPages > 0) {
      progress = Math.min(100, (currentPage / totalPages) * 100);
    }
    // 예상 시간 표시 로직 수정 (displayRemainingTime 사용)
    const timeString = formatTime(displayRemainingTime);

    switch (status) {
      case 'Ready':
        icon = <InformationCircleIcon className="h-6 w-6 text-blue-500" />;
        message = '번역할 PDF 파일을 선택하세요.';
        break;
      case 'File selected':
        icon = <CheckCircleIcon className="h-6 w-6 text-green-500" />;
        message = '파일이 선택되었습니다. 번역 버튼을 누르세요.';
        break;
      case 'Uploading':
        icon = <ArrowUpTrayIcon className="h-6 w-6 text-blue-500 animate-pulse" />;
        message = 'PDF 파일 업로드 중...';
        break;
      case 'Starting':
      case 'Parsing':
        icon = <ClockIcon className="h-6 w-6 text-yellow-500 animate-spin" />;
        message = '번역 작업 준비 중...';
        break;
      case 'Translating':
        icon = <ClockIcon className="h-6 w-6 text-yellow-500 animate-spin" />;
        message = `번역 중 (${currentPage}/${totalPages} 페이지)... ${timeString}`;
        break;
      case 'Done':
        icon = <CheckCircleIcon className="h-6 w-6 text-green-500" />;
        message = '번역 완료!';
        break;
      case 'Error':
        icon = <XCircleIcon className="h-6 w-6 text-red-500" />;
        message = `오류 발생: ${errorDetail || '알 수 없는 오류'}`;
        break;
      default:
        message = `알 수 없는 상태: ${status}`;
    }

    return (
      <div className="mt-4 p-4 bg-gray-50 rounded-lg shadow-inner">
        <div className="flex items-center space-x-3">
          {icon}
          <p className="text-sm font-medium text-gray-700">{message}</p>
        </div>
        {(status === 'Translating' || status === 'Parsing' || status === 'Starting') && totalPages > 0 && (
          <div className="mt-2 w-full bg-gray-200 rounded-full h-2.5 dark:bg-gray-700">
            <div
              className="bg-blue-600 h-2.5 rounded-full transition-width duration-500 ease-in-out"
              style={{ width: `${progress}%` }}
            ></div>
          </div>
        )}
      </div>
    );
  };

  // Callback function for PdfPreview component to report errors
  const handlePreviewError = (message: string) => {
    console.warn("PDF Preview Warning/Error:", message); // Log as warning instead of error
    // Don't set status to Error here, just log it.
    // setErrorDetail(`PDF 미리보기 오류: ${message}`);
  }

  return (
    <main className="container mx-auto p-4 md:p-8 max-w-4xl">
      {/* --- 페이지 제목 카드 디자인 적용 --- */}
      <div className="bg-gradient-to-r from-blue-600 to-indigo-700 text-white p-6 rounded-lg shadow-md mb-8">
        <h1 className="text-3xl font-bold text-center">PDF 한국어 번역기</h1>
        <p className="text-center text-blue-100 mt-1 text-sm">영문 PDF를 한국어로 번역합니다.</p>
      </div>
      {/* --- 카드 디자인 끝 --- */}

      <div className="bg-white p-6 rounded-lg shadow-md mb-6">
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-1">1. PDF 파일 선택</label>
          <label htmlFor="file-upload" className={`group flex flex-col items-center justify-center w-full h-32 px-4 transition bg-white border-2 ${isProcessing ? 'border-gray-200' : 'border-gray-300 hover:border-blue-400'} border-dashed rounded-md appearance-none ${isProcessing ? 'cursor-not-allowed' : 'cursor-pointer'} focus:outline-none`}>
             <span className="flex flex-col items-center justify-center space-y-2">
                <ArrowUpTrayIcon className={`w-10 h-10 ${selectedFile ? 'text-blue-600' : 'text-gray-400'} ${!isProcessing && 'group-hover:text-blue-500'}`} />
                <span className={`font-medium text-sm ${selectedFile ? 'text-blue-700' : 'text-gray-600'} ${!isProcessing && 'group-hover:text-blue-600'} truncate max-w-xs text-center`}>
                  {selectedFile ? selectedFile.name : '여기를 클릭하거나 파일을 드래그하세요'}
                </span>
                {!selectedFile && <span className="text-xs text-gray-500">PDF 파일만 가능</span>}
             </span>
             <input id="file-upload" type="file" accept=".pdf" onChange={handleFileChange} className="sr-only" disabled={isProcessing} />
          </label>
          <p className="text-xs text-gray-500 mt-1">※ 디지털 형식의 PDF만 지원됩니다. 스캔/이미지 PDF는 OCR 기능이 없어 번역할 수 없습니다.</p>
        </div>

        <div className="mb-4">
          <label htmlFor="page-range" className="block text-sm font-medium text-gray-700 mb-1">2. 번역할 페이지 범위 (선택 사항)</label>
          <input
            id="page-range"
            type="text"
            value={pageRange}
            onChange={handlePageRangeChange}
            placeholder="예: 1-5, 8, 10-12 (비워두면 전체 페이지)"
            className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
            disabled={isProcessing}
          />
        </div>

        <div className="mb-6">
          <label htmlFor="custom-instructions" className="block text-sm font-medium text-gray-700 mb-1">3. 추가 번역 지침 (선택 사항)</label>
          <textarea
            id="custom-instructions"
            rows={3}
            value={customInstructions}
            onChange={(e) => setCustomInstructions(e.target.value)}
            placeholder="예: 특정 용어는 특정 단어로 번역해주세요."
            className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
            disabled={isProcessing}
          />
        </div>

        <button
          onClick={handleTranslate}
          disabled={!selectedFile || isProcessing}
          className={`w-full py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white transition duration-150 ease-in-out ${!selectedFile || isProcessing ? 'bg-gray-400 cursor-not-allowed' : 'bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500'}`}
        >
          {isProcessing ? '처리 중...' : '번역 시작'}
        </button>
      </div>

      {status !== 'Ready' && renderStatus()}

      {translatedFileUrl && status === 'Done' && (
        <div className="mt-6 text-center">
          <a
            href={translatedFileUrl}
            download
            className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition duration-150 ease-in-out space-x-2"
          >
            <DocumentArrowDownIcon className="h-5 w-5" />
            <span>번역된 PDF 다운로드</span>
          </a>
        </div>
      )}

      {selectedFile && (
        <div className="mt-8">
          <h2 className="text-xl font-semibold mb-4 text-gray-700">
            {status === 'Done' && translatedFileUrl ? 'PDF 미리보기 (번역본)' : 'PDF 미리보기 (원본)'}
          </h2>
          <div className="border rounded-lg overflow-hidden shadow-lg" style={{ height: '600px', overflowY: 'auto' }}>
            <PdfPreview
                file={status === 'Done' && translatedFileUrl ? translatedFileUrl : selectedFile}
                onError={handlePreviewError}
             />
          </div>
        </div>
      )}
    </main>
  );
} 