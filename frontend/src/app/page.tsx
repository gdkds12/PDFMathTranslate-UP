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
  // --- 상태 폴링 끝 ---

  // --- 새로운 상태 변수 추가 ---
  const [keepTechnicalTerms, setKeepTechnicalTerms] = useState<boolean>(false);
  const [keepEnglishNames, setKeepEnglishNames] = useState<boolean>(false);
  const [customInstructions, setCustomInstructions] = useState<string>('');
  // --- 새로운 상태 변수 끝 ---

  // Set the worker source globally when the page mounts on the client
  // Use useEffect to ensure it runs only once on the client side
  useEffect(() => {
    pdfjs.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.js`;
  }, []);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setSelectedFile(event.target.files[0]);
      setStatus('File selected');
      setErrorDetail(null);
      setTranslatedFileUrl(null);
      setJobId(null); // 이전 작업 정보 초기화
      setCurrentPage(0);
      setTotalPages(0);

      // --- 파일 변경 시 새 옵션 상태 초기화 ---
      setKeepTechnicalTerms(false);
      setKeepEnglishNames(false);
      setCustomInstructions('');
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

  // --- 폴링 함수 ---
  const pollStatus = async (currentJobId: string) => {
    try {
      const response = await fetch(`http://localhost:8000/api/translate/status/${currentJobId}`);
      if (!response.ok) {
        // 404 (Job Not Found) 등 처리
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
        setIsProcessing(false);
        return;
      }

      const data = await response.json();
      setStatus(data.status || 'Unknown');
      setCurrentPage(data.current_page || 0);
      setTotalPages(data.total_pages || 0);
      setErrorDetail(data.error || null);

      if (data.status === 'Done') {
        console.log("Translation done, stopping polling.");
        setTranslatedFileUrl(`http://localhost:8000/api/translate/download/${currentJobId}`);
        stopPolling();
        setIsProcessing(false);
      } else if (data.status === 'Error') {
        console.error("Translation error reported by backend:", data.error);
        stopPolling();
        setIsProcessing(false);
      }

    } catch (error) {
      console.error('Error polling translation status:', error);
      setStatus('Error');
      setErrorDetail('상태 폴링 중 연결 오류 발생.');
      stopPolling();
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

  // 컴포넌트 언마운트 시 폴링 중지
  useEffect(() => {
    return () => {
      stopPolling();
    };
  }, []);

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

    const formData = new FormData();
    formData.append('pdf', selectedFile);
    if (pageRange.trim()) {
      formData.append('pages', pageRange.trim());
    }
    // --- 새로운 옵션 FormData에 추가 ---
    // 백엔드에서 받을 키 이름은 추후 협의 필요 (예: keep_terms, keep_names, instructions)
    formData.append('keep_technical_terms', String(keepTechnicalTerms)); // boolean을 문자열로
    formData.append('keep_english_names', String(keepEnglishNames));
    formData.append('custom_instructions', customInstructions);
    // --- FormData 추가 끝 ---

    try {
      // 백엔드에 작업 시작 요청
      const response = await fetch('http://localhost:8000/api/translate', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        // 시작 요청 자체 실패 처리
        let detail = `Failed to start translation (HTTP ${response.status})`;
        try {
          const errorData = await response.json();
          detail = errorData.detail || detail;
        } catch (jsonError) {
          console.error("Failed to parse start error response JSON", jsonError);
        }
        throw new Error(detail);
      }

      // 작업 시작 성공, job_id 받고 폴링 시작
      const data = await response.json();
      if (data.job_id) {
        setStatus('Starting'); // 서버에서 작업 시작됨
        startPolling(data.job_id);
      } else {
        throw new Error("백엔드에서 작업 ID를 반환하지 않았습니다.");
      }

    } catch (error) {
      console.error('Error starting translation job:', error);
      setStatus('Error');
      setErrorDetail((error as Error).message || '알 수 없는 오류가 발생했습니다.');
      setTranslatedFileUrl(null);
      setIsProcessing(false);
      stopPolling();
    }
    // handleTranslate는 이제 즉시 반환하고, 실제 상태 업데이트는 폴링에서 처리
  };

  // Helper to render status icon and text
  const renderStatus = () => {
    switch (status) {
      case 'Uploading':
      case 'Starting':
      case 'Parsing':
        return <span className="flex items-center text-sm text-blue-600"><InformationCircleIcon className="w-4 h-4 mr-1 animate-spin" /> {status}...</span>;
      case 'Translating':
        // 진행률 표시 추가
        const progress = totalPages > 0 ? Math.round((currentPage / totalPages) * 100) : 0;
        return (
          <div className="w-full">
            <span className="flex items-center text-sm text-blue-600 mb-1">
              <ClockIcon className="w-4 h-4 mr-1 animate-spin" />
              Translating page {currentPage} of {totalPages}...
            </span>
            {/* Progress Bar */}
            <div className="w-full bg-gray-200 rounded-full h-1.5">
              <div className="bg-blue-600 h-1.5 rounded-full transition-all duration-300 ease-in-out" style={{ width: `${progress}%` }}></div>
            </div>
          </div>
        );
      case 'File selected':
      case 'Ready':
        return <span className="flex items-center text-sm text-gray-500"><InformationCircleIcon className="w-4 h-4 mr-1" /> {status}</span>;
      case 'Done':
        return <span className="flex items-center text-sm text-green-600"><CheckCircleIcon className="w-4 h-4 mr-1" /> Translation successful!</span>;
      case 'Error':
        return <span className="flex items-center text-sm text-red-600"><XCircleIcon className="w-4 h-4 mr-1" /> Error</span>;
      default:
        return <span className="text-sm text-gray-500">{status}</span>;
    }
  };

  // Callback function for PdfPreview component to report errors
  const handlePreviewError = (message: string) => {
      setErrorDetail(message);
      // Optionally set status to 'Error' as well if preview failing is critical
      // setStatus('Error');
  }

  return (
    <main className="flex min-h-screen items-center justify-center bg-gradient-to-br from-gray-100 to-gray-200 p-4">
      <div className="w-full max-w-lg bg-white rounded-xl shadow-lg overflow-hidden">
        {/* Header */}
        <div className="bg-blue-600 p-6 text-white">
          <h1 className="text-2xl font-bold text-center">PDF 번역 서비스</h1>
          <p className="text-sm text-blue-100 text-center mt-1">영문 PDF를 한국어로 번역합니다.</p>
        </div>

        {/* Content Area */}
        <div className="p-6 md:p-8 space-y-6">
          {/* File Upload */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">1. PDF 파일 업로드</label>
            <label htmlFor="pdf-upload" className={`group flex justify-center w-full h-32 px-4 transition bg-white border-2 ${isProcessing ? 'border-gray-200' : 'border-gray-300 hover:border-blue-400'} border-dashed rounded-md appearance-none ${isProcessing ? 'cursor-not-allowed' : 'cursor-pointer'} focus:outline-none`}>
              <span className="flex flex-col items-center justify-center space-x-2 pt-4">
                <ArrowUpTrayIcon className={`w-10 h-10 ${selectedFile ? 'text-blue-600' : 'text-gray-400'} ${!isProcessing && 'group-hover:text-blue-500'}`} />
                <span className={`font-medium text-sm ${selectedFile ? 'text-blue-700' : 'text-gray-600'} ${!isProcessing && 'group-hover:text-blue-600'} truncate max-w-xs`}>
                  {selectedFile ? selectedFile.name : '여기를 클릭하거나 파일을 드래그하세요'}
                </span>
                {!selectedFile && <span className="text-xs text-gray-500">PDF 파일만 가능</span>}
              </span>
              <input id="pdf-upload" type="file" accept=".pdf" onChange={handleFileChange} className="sr-only" disabled={isProcessing} />
            </label>
          </div>

          {/* 2. Page Range Input */}
          <div>
            <label htmlFor="page-range" className="block text-sm font-semibold text-gray-700 mb-2">2. 페이지 범위 (선택)</label>
            <input
              id="page-range"
              type="text"
              placeholder="예: 1-5, 8, 11-13 (전체: 비워두기)"
              value={pageRange}
              onChange={handlePageRangeChange}
              className={`w-full px-4 py-2 text-gray-700 ${isProcessing ? 'bg-gray-100 cursor-not-allowed' : 'bg-gray-50'} border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition`}
              disabled={isProcessing}
            />
            <p className="text-xs text-gray-500 mt-1">쉼표(,)로 구분, 하이픈(-)으로 범위 지정</p>
          </div>

          {/* --- 3. 번역 옵션 추가 (위치 수정) --- */}
          <div className="space-y-3">
             <label className="block text-sm font-semibold text-gray-700">3. 번역 옵션 (선택)</label>
             <div className="relative flex items-start">
               <div className="flex h-6 items-center">
                 <input
                   id="keepTechnicalTerms"
                   aria-describedby="keepTechnicalTerms-description"
                   name="keepTechnicalTerms"
                   type="checkbox"
                   checked={keepTechnicalTerms}
                   onChange={(e) => setKeepTechnicalTerms(e.target.checked)}
                   disabled={isProcessing}
                   className={`h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-600 ${isProcessing ? 'cursor-not-allowed opacity-50' : ''}`}
                 />
               </div>
               <div className="ml-3 text-sm leading-6">
                 <label htmlFor="keepTechnicalTerms" className={`font-medium text-gray-900 ${isProcessing ? 'cursor-not-allowed' : ''}`}>
                   전문 용어 번역하지 않기
                 </label>
                 {/* <p id="keepTechnicalTerms-description" className="text-gray-500">예: 모델 이름, 특정 기술 용어 등</p> */}
               </div>
             </div>
             <div className="relative flex items-start">
               <div className="flex h-6 items-center">
                 <input
                   id="keepEnglishNames"
                   aria-describedby="keepEnglishNames-description"
                   name="keepEnglishNames"
                   type="checkbox"
                   checked={keepEnglishNames}
                   onChange={(e) => setKeepEnglishNames(e.target.checked)}
                   disabled={isProcessing}
                   className={`h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-600 ${isProcessing ? 'cursor-not-allowed opacity-50' : ''}`}
                 />
               </div>
               <div className="ml-3 text-sm leading-6">
                 <label htmlFor="keepEnglishNames" className={`font-medium text-gray-900 ${isProcessing ? 'cursor-not-allowed' : ''}`}>
                   영문 이름 번역하지 않기
                 </label>
                 {/* <p id="keepEnglishNames-description" className="text-gray-500">예: 사람 이름, 고유 명사 등</p> */}
               </div>
             </div>
           </div>
           {/* --- 번역 옵션 끝 --- */}

           {/* --- 4. 세부 지침 추가 (위치 수정) --- */}
           <div>
             <label htmlFor="customInstructions" className="block text-sm font-semibold text-gray-700 mb-2">4. 세부 지침 (선택)</label>
             <textarea
               id="customInstructions"
               name="customInstructions"
               rows={3}
               placeholder="번역 스타일, 특정 단어 처리 방식 등 추가적인 지시사항을 입력하세요. (예: '모든 인용구는 원문 그대로 유지해주세요', 'Chapter는 장으로 번역해주세요')"
               value={customInstructions}
               onChange={(e) => setCustomInstructions(e.target.value)}
               disabled={isProcessing}
               className={`block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-blue-600 sm:text-sm sm:leading-6 transition ${isProcessing ? 'bg-gray-100 cursor-not-allowed opacity-50' : 'bg-white'}`}
             />
           </div>
           {/* --- 세부 지침 끝 --- */}

          {/* Translate Button */}
          <div className="pt-2">
            <button
              onClick={handleTranslate}
              disabled={!selectedFile || isProcessing}
              className="w-full flex justify-center items-center px-6 py-3 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition duration-150 ease-in-out"
            >
              {isProcessing ? (
                <>
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  처리 중...
                </>
              ) : (
                '번역 시작'
              )}
            </button>
          </div>

          {/* Status and Result */}
          <div className="pt-4 border-t border-gray-200 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm font-semibold text-gray-700">상태:</span>
              {renderStatus()}
            </div>

            {/* 에러 메시지 표시 개선 */}
            {status === 'Error' && errorDetail && (
              <p className="text-sm text-red-600 bg-red-50 p-3 rounded-md border border-red-200">{errorDetail}</p>
            )}

            {/* 다운로드 버튼 표시 */}
            {status === 'Done' && translatedFileUrl && (
              <div className="text-center">
                <a
                  href={translatedFileUrl}
                  download
                  className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 transition"
                >
                  <DocumentArrowDownIcon className="-ml-1 mr-2 h-5 w-5" />
                  번역된 PDF 다운로드
                </a>
              </div>
            )}

            {/* Dynamically loaded PDF Preview */}
            {status === 'Done' && translatedFileUrl && (
                <PdfPreview fileUrl={translatedFileUrl} onError={handlePreviewError} />
            )}

            {/* 에러 메시지 표시 (Unified) */}
            {status !== 'Error' && errorDetail && status === 'Done' && (
                <p className="text-sm text-orange-600 bg-orange-50 p-3 rounded-md border border-orange-200">미리보기 오류: {errorDetail}</p>
            )}
          </div>
        </div>
      </div>
    </main>
  );
} 