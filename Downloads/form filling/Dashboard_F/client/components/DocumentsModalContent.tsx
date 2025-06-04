import React, { useState, useEffect } from 'react';
import { DocumentItem } from '../types';
import { apiService } from '../services/api';

import { DocumentTextIcon, ArrowDownTrayIcon, EyeIcon } from './icons';

interface DocumentsModalContentProps {
  documents: DocumentItem[];
  T: any; // Translation object
}

const getFileIcon = (): React.ReactNode => {
  return <DocumentTextIcon className="w-8 h-8 text-accent" />; 
};

const DocumentsModalContent: React.FC<DocumentsModalContentProps> = ({ documents, T }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [pdfBlob, setPdfBlob] = useState<Blob | null>(null);
  const [lastPdfFetchTime, setLastPdfFetchTime] = useState<number>(0);
  const [pdfSize, setPdfSize] = useState<string>('Loading...');
  
  const labels = T.documentsContent || {};
  const documentTypeNames = labels.documentTypes || {
    'PDF': "PDF Document",
    'DOCX': "Word Document",
    'TXT': "Text File",
    'Email': "Email Message",
    'Image': "Image File",
  };
  
  // Function to fetch PDF from backend
  const fetchPdfFromBackend = async () => {
    console.log('🔄 Fetching PDF from backend...');
    setIsLoading(true);
    try {
      const blob = await apiService.getPdfBlob();
      const now = Date.now();
      setLastPdfFetchTime(now);
      
      console.log('✅ PDF loaded, size:', blob.size, 'bytes');
      setPdfBlob(blob);
      const formattedSize = `${(blob.size / 1024).toFixed(1)} KB`;
      setPdfSize(formattedSize);
      
      // Update the size in the PDF document
      const pdfDoc = documents.find(doc => doc.type === 'PDF');
      if (pdfDoc) {
        pdfDoc.size = formattedSize;
      }
      
      return blob;
    } catch (error) {
      console.error('❌ Error fetching PDF:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error('❌ Error details:', errorMessage);
      setPdfSize('Error loading');
      return null;
    } finally {
      setIsLoading(false);
    }
  };

  const handlePreview = async (doc: DocumentItem) => {
    // Only attempt to fetch from API if it's a PDF
    if (doc.type === 'PDF') {
      try {
        console.log('📄 Previewing PDF document:', doc.name);
        setIsLoading(true);
        
        // Use cached PDF blob if available and recently fetched (within the last 30 seconds)
        const now = Date.now();
        const useCache = pdfBlob && (now - lastPdfFetchTime) < 30000;
        
        let blobToUse: Blob;
        if (useCache) {
          console.log('🔄 Using cached PDF blob, age:', ((now - lastPdfFetchTime)/1000).toFixed(1), 'seconds');
          blobToUse = pdfBlob;
        } else {
          console.log('🔄 Fetching fresh PDF blob...');
          blobToUse = await apiService.getPdfBlob();
          setPdfBlob(blobToUse);
          setLastPdfFetchTime(now);
          setPdfSize(`${(blobToUse.size / 1024).toFixed(1)} KB`);
        }
        
        console.log('✅ PDF blob received, size:', blobToUse.size, 'bytes');
        
        if (blobToUse.size === 0) {
          console.error('❌ Empty PDF blob received');
          alert(labels.previewError || "Error: Empty PDF document received.");
          setIsLoading(false);
          return;
        }
        
        const pdfUrl = URL.createObjectURL(blobToUse);
        console.log('🔗 Created object URL for PDF:', pdfUrl);
        window.open(pdfUrl, '_blank');
        // Don't revoke the URL immediately as the browser needs it open
        setTimeout(() => URL.revokeObjectURL(pdfUrl), 60000); // Cleanup after 1 minute
      } catch (error: unknown) {
        console.error('❌ Error fetching PDF:', error);
        if (error instanceof Error) {
          console.error('❌ Error details:', error.message);
          alert(labels.previewError || "Error loading PDF document.");
        } else {
          console.error('❌ Unknown error:', error);
          alert(labels.previewError || "An unknown error occurred.");
        }
      } finally {
        setIsLoading(false);
      }
    } else if (doc.url && doc.url !== '#') {
      window.open(doc.url, '_blank');
    } else {
      alert(labels.previewNotAvailable || "Preview not available for this document.");
    }
  };

  const handleDownload = async (doc: DocumentItem) => {
    if (doc.type === 'PDF') {
      try {
        console.log('💾 Downloading PDF document:', doc.name);
        setIsLoading(true);
        
        // Use cached PDF blob if available and recently fetched (within the last 30 seconds)
        const now = Date.now();
        const useCache = pdfBlob && (now - lastPdfFetchTime) < 30000;
        
        let blobToUse: Blob;
        if (useCache) {
          console.log('🔄 Using cached PDF blob for download, age:', ((now - lastPdfFetchTime)/1000).toFixed(1), 'seconds');
          blobToUse = pdfBlob;
        } else {
          console.log('🔄 Fetching fresh PDF blob for download...');
          blobToUse = await apiService.getPdfBlob();
          setPdfBlob(blobToUse);
          setLastPdfFetchTime(now);
          setPdfSize(`${(blobToUse.size / 1024).toFixed(1)} KB`);
        }
        
        console.log('✅ PDF blob received for download, size:', blobToUse.size, 'bytes');
        
        if (blobToUse.size === 0) {
          console.error('❌ Empty PDF blob received for download');
          alert(labels.downloadError || "Error: Empty PDF document received.");
          setIsLoading(false);
          return;
        }
        
        // Create a download link
        const downloadUrl = URL.createObjectURL(blobToUse);
        const a = document.createElement('a');
        a.href = downloadUrl;
        a.download = doc.name;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        
        // Cleanup the created object URL
        URL.revokeObjectURL(downloadUrl);
      } catch (error: unknown) {
        console.error('❌ Error downloading PDF:', error);
        if (error instanceof Error) {
          console.error('❌ Error details:', error.message);
          alert(labels.downloadError || "Error downloading PDF document.");
        } else {
          console.error('❌ Unknown error:', error);
          alert(labels.downloadError || "An unknown error occurred.");
        }
      } finally {
        setIsLoading(false);
      }
    } else if (doc.url && doc.url !== '#') {
      const link = document.createElement('a');
      link.href = doc.url;
      link.setAttribute('download', doc.name);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } else {
      alert(labels.downloadNotAvailable || "Download not available for this document.");
    }
  };

  

  // Effect to fetch PDF only when component mounts (when user opens Documents panel)
  useEffect(() => {
    console.log('📄 DocumentsModalContent component mounted');
    console.log('📋 Documents list:', documents);
    
    // Initial fetch when component mounts - this happens when user clicks on Documents and Forms
    fetchPdfFromBackend();
    
    return () => {
      console.log('📄 DocumentsModalContent component unmounted');
    };
  }, []); // Empty dependency array means it only runs once on mount

  if (documents.length === 0) {
    return <p className="text-text-secondary">{labels.noDocumentsFound || "No documents found."}</p>;
  }

  return (
    <div className="flex-grow overflow-y-auto bg-[#f4f0ff] p-4 sm:p-6 rounded-b-lg">
      <div className="space-y-4">
        {documents.map((doc) => (
          <div key={doc.id} className="bg-[#f8f5fc] p-4 rounded-lg shadow-sm border border-border-color flex items-center gap-4 flex-wrap justify-between w-full min-w-0">
            <div className="flex items-center gap-3 min-w-0 w-full">
              {getFileIcon()}
              <div className="flex flex-col gap-1 min-w-0 w-full">
                <p className="text-sm font-semibold text-text-primary break-words max-w-[220px]" title={doc.name}>{doc.name}</p>
                <p className="text-xs text-text-secondary leading-tight break-words max-w-full">
                  Type: {documentTypeNames[doc.type] || doc.type} • Added: {doc.dateAdded} • Size: {doc.type === 'PDF' ? pdfSize : doc.size}
                </p>
                {doc.type === 'PDF' && lastPdfFetchTime > 0 && (
                  <p className="text-xs text-blue-500">
                    PDF last updated: {new Date(lastPdfFetchTime).toLocaleTimeString()}
                  </p>
                )}
              </div>
            </div>
            <div className="flex items-center gap-2">
              {isLoading && <span className="text-xs text-accent animate-pulse">Loading...</span>}
              <button
                className="rounded-full bg-accent/10 p-2 text-accent hover:bg-accent/20 focus:outline-none focus:ring-2 focus:ring-accent"
                onClick={() => handlePreview(doc)}
                type="button"
                title="Preview"
                aria-label={`Preview ${doc.name}`}
              >
                <EyeIcon className="w-5 h-5" />
              </button>
              <button
                className="rounded-md bg-green-100 p-2 text-green-700 hover:bg-green-200 focus:outline-none focus:ring-2 focus:ring-green-400 transition flex items-center gap-1"
                onClick={() => handleDownload(doc)}
                type="button"
                title="Download"
                aria-label={`Download ${doc.name}`}
                disabled={isLoading}
                style={{ paddingLeft: '12px', paddingRight: '12px' }}
              >
                <ArrowDownTrayIcon className="w-5 h-5" />
                <span className="font-semibold text-xs whitespace-nowrap">Download</span>
              </button>

            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default DocumentsModalContent;