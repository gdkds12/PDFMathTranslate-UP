"use client"; // This component uses client-side features

import React, { useState, useEffect } from 'react';
// Remove pdfjs import from here
import { Document, Page } from 'react-pdf';
// import { Document, Page, pdfjs } from 'react-pdf'; // Re-import pdfjs here
import 'react-pdf/dist/esm/Page/AnnotationLayer.css';
import 'react-pdf/dist/esm/Page/TextLayer.css';
import { ChevronLeftIcon, ChevronRightIcon } from '@heroicons/react/24/solid'; // Icons for navigation

// Remove the global workerSrc setting from this file
// if (typeof window !== 'undefined') {
//     pdfjs.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.js`;
// }

// Export the props interface
export interface PdfPreviewProps {
  fileUrl: string;
  onError: (message: string) => void; // Function to report errors back to parent
}

export default function PdfPreview({ fileUrl, onError }: PdfPreviewProps) {
  const [numPages, setNumPages] = useState<number | null>(null);
  const [pageNumber, setPageNumber] = useState<number>(1);
  const [containerWidth, setContainerWidth] = useState<number | undefined>(undefined);

  // Handle dynamic width calculation on the client-side
  useEffect(() => {
    const handleResize = () => {
      // Adjust the calculation based on your layout needs
      // Example: Use 90% of window width, max 600px
      const width = Math.min(window.innerWidth * 0.9, 600);
      setContainerWidth(width);
    };

    // Set initial width
    handleResize();

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);


  function onDocumentLoadSuccess({ numPages }: { numPages: number }): void {
    setNumPages(numPages);
    setPageNumber(1); // Reset to first page on new document load
  }

  function handleLoadError(error: Error): void {
    const isWorkerDestroyedError = error.message?.includes('Worker was destroyed');

    // 개발자 콘솔에는 항상 오류/경고를 기록합니다.
    if (isWorkerDestroyedError) {
      // 'Worker was destroyed'는 경고 수준으로 기록 (기능은 정상 작동하므로)
      console.warn('PDF Preview Warning (Worker destroyed):', error.message);
    } else {
      // 다른 미리보기 로드 오류는 에러 수준으로 기록
      console.error('Error loading PDF for preview:', error);
    }

    // 'Worker was destroyed' 오류가 아닐 경우에만 onError 콜백을 호출하여 GUI에 표시
    if (!isWorkerDestroyedError) {
      const errorMessage = `미리보기를 로드하는 중 오류 발생: ${error.message}`;
      onError(errorMessage);
    }
    // 'Worker was destroyed' 오류인 경우, onError를 호출하지 않으므로 GUI에는 표시되지 않음
  }

  const goToPrevPage = () => setPageNumber(prev => Math.max(prev - 1, 1));
  const goToNextPage = () => setPageNumber(prev => Math.min(prev + 1, numPages || 1));

  return (
    <div className="mt-6 border-t pt-6">
      <h3 className="text-lg font-semibold text-gray-800 mb-4 text-center">번역 결과 미리보기</h3>
      <div className="pdf-container max-w-full overflow-x-auto bg-gray-100 p-2 sm:p-4 rounded-md shadow-inner flex flex-col items-center border border-gray-200 min-h-[600px]">
        <Document
          file={fileUrl}
          onLoadSuccess={onDocumentLoadSuccess}
          onLoadError={handleLoadError}
          loading={<div className="flex justify-center items-center min-h-[600px]"><p className="text-sm text-gray-500">미리보기 로딩 중...</p></div>}
          error={<div className="flex justify-center items-center min-h-[600px]"><p className="text-sm text-red-500">미리보기를 로드할 수 없습니다.</p></div>}
        >
          <Page
            pageNumber={pageNumber}
            renderTextLayer={true}
            width={containerWidth}
            key={pageNumber}
          />
        </Document>
      </div>

      {/* Page Navigation */}
      {numPages && numPages > 1 && (
        <div className="flex justify-center items-center mt-4 space-x-4">
          <button
            onClick={goToPrevPage}
            disabled={pageNumber <= 1}
            className="p-1 border rounded-full text-gray-600 bg-white hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed transition"
            aria-label="Previous Page"
          >
            <ChevronLeftIcon className="w-5 h-5" />
          </button>
          <span className="text-sm text-gray-600 font-medium">
            페이지 {pageNumber} / {numPages}
          </span>
          <button
            onClick={goToNextPage}
            disabled={pageNumber >= numPages}
            className="p-1 border rounded-full text-gray-600 bg-white hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed transition"
            aria-label="Next Page"
          >
             <ChevronRightIcon className="w-5 h-5" />
          </button>
        </div>
      )}
    </div>
  );
} 