import React, { useState, forwardRef } from 'react';

const API_ENDPOINT = "https://test.mayacode.io/api/api";

export const PDFViewer = forwardRef(({ buttonClassName, onFetchPdf = null }, ref) => {
  const [pdfData, setPdfData] = useState(null);
  const [pdfFilename, setPdfFilename] = useState('filled_form.pdf');
  const [loading, setLoading] = useState(false);
  const [showPdfModal, setShowPdfModal] = useState(false);
  const [message, setMessage] = useState(null);
  const [emailStatus, setEmailStatus] = useState(null);
  const [pdfModified, setPdfModified] = useState(false);

  // Fetch PDF with action=show
  const fetchPdf = async () => {
    setLoading(true);
    setMessage(null);
    setEmailStatus(null);
    
    try {
      const response = await fetch(`${API_ENDPOINT}/get-pdf?action=show`);
      
      if (response.ok) {
        const result = await response.json();
        
        if (result.status === 'success' && result.pdf_data) {
          setPdfData(result.pdf_data);
          setPdfFilename(result.pdf_filename || 'filled_form.pdf');
          setShowPdfModal(true);
          
          // Notify parent component if callback provided
          if (onFetchPdf) {
            onFetchPdf();
          }
        } else {
          setMessage(`Error: ${result.message || 'Please try later'}`);
        }
      } else {
        setMessage('Error: Please try later');
      }
    } catch (error) {
      setMessage(`Error: Please try later`);
    } finally {
      setLoading(false);
    }
  };

  // Send PDF via email with action=send
  const sendPdfEmail = async () => {
    setEmailStatus('sending');
    setMessage(null);
    
    try {
      const response = await fetch(`${API_ENDPOINT}/get-pdf?action=send`);
      
      if (response.ok) {
        const result = await response.json();
        
        if (result.status === 'success' && result.email_sent) {
          setEmailStatus('success');
          // Wait for a moment to show the success animation before closing
          setTimeout(() => {
            setShowPdfModal(false);
          }, 2000);
        } else {
          setEmailStatus('error');
          setMessage(`Error: ${result.message || 'Service is down'}`);
        }
      } else {
        setEmailStatus('error');
        setMessage('Error: Service is down');
      }
    } catch (error) {
      setEmailStatus('error');
      setMessage(`Error: ${error.message}`);
    }
  };

  // Download the PDF
  const downloadPdf = () => {
    if (!pdfData) return;
    
    try {
      // Create a Blob from the base64 data
      const byteCharacters = atob(pdfData);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      const pdfBlob = new Blob([byteArray], { type: 'application/pdf' });
      
      // Create an object URL from the Blob
      const blobUrl = URL.createObjectURL(pdfBlob);
      
      // Create a download link and trigger it
      const downloadLink = document.createElement('a');
      downloadLink.href = blobUrl;
      downloadLink.download = pdfFilename;
      document.body.appendChild(downloadLink);
      downloadLink.click();
      
      // Clean up
      setTimeout(() => {
        document.body.removeChild(downloadLink);
        URL.revokeObjectURL(blobUrl);
      }, 100);
      
      setMessage('PDF downloaded successfully');
    } catch (error) {
      setMessage(`Error downloading PDF: ${error.message}`);
    }
  };

  // Close the PDF modal
  const closeModal = () => {
    setShowPdfModal(false);
  };

  return (
    <div>
      {/* Get Form Button */}
      <button
        ref={ref}
        onClick={fetchPdf}
        disabled={loading}
        className={buttonClassName || "bg-purple-500 hover:bg-purple-600 text-white p-4 rounded-md flex items-center justify-center gap-2"}
      >
        {loading ? 'Loading...' : 'Get Form'}
      </button>
      
      {/* Message Display */}
      {message && (
        <div className={`mt-2 p-2 rounded text-sm ${message.includes('Error') ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'}`}>
          {message}
        </div>
      )}
      
      {/* PDF Modal */}
      {showPdfModal && pdfData && (
        <div 
          className="fixed inset-0 bg-gradient-to-br from-purple-100 to-pink-100 bg-opacity-95 flex items-center justify-center z-50 p-4"
          onClick={closeModal} // Close when clicking the backdrop
        >
          <div 
            className="bg-white rounded-lg shadow-xl w-full max-w-4xl flex flex-col max-h-screen border border-purple-100"
            onClick={(e) => e.stopPropagation()} // Prevent closing when clicking the modal itself
          >
            {/* Modal Header */}
            <div className="p-4 border-b border-purple-100 bg-purple-50 rounded-t-lg flex justify-between items-center">
              <h3 className="text-lg font-semibold">Form Preview</h3>
              <button 
                onClick={closeModal}
                className="text-gray-500 hover:text-gray-700 text-3xl font-light leading-none"
                aria-label="Close"
              >
                &times;
              </button>
            </div>
            
            {/* PDF Viewer with Email Sending Animation Overlay */}
            <div className="flex-grow overflow-auto p-4 relative">
              {emailStatus === 'sending' && (
                <div className="absolute inset-0 bg-white bg-opacity-90 flex flex-col items-center justify-center z-10">
                  <div className="mb-4">
                    <div className="w-16 h-16 border-t-4 border-blue-500 border-solid rounded-full animate-spin"></div>
                  </div>
                  <div className="text-lg font-medium text-gray-700 mb-2">Sending PDF...</div>
                  <div className="text-sm text-gray-500">Your document is being emailed</div>
                </div>
              )}
              
              {emailStatus === 'success' && (
                <div className="absolute inset-0 bg-white bg-opacity-90 flex flex-col items-center justify-center z-10">
                  <div className="mb-4 text-green-500">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  </div>
                  <div className="text-lg font-medium text-gray-700 mb-2">Email Sent!</div>
                  <div className="text-sm text-gray-500">Your PDF has been sent successfully</div>
                </div>
              )}
              
              {emailStatus === 'error' && (
                <div className="absolute inset-0 bg-white bg-opacity-90 flex flex-col items-center justify-center z-10">
                  <div className="mb-4 text-red-500">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </div>
                  <div className="text-lg font-medium text-gray-700 mb-2">Email Failed</div>
                  <div className="text-sm text-gray-500">{message || 'Something went wrong. Please try again.'}</div>
                </div>
              )}
              
              <iframe 
                src={`data:application/pdf;base64,${pdfData}`}
                className="w-full h-full min-h-[500px] border"
                title="PDF Viewer"
              />
            </div>
            
            {/* Modal Footer with Actions */}
            <div className="p-4 border-t border-purple-100 bg-purple-50 rounded-b-lg flex justify-end space-x-4">
              <button
                onClick={sendPdfEmail}
                disabled={emailStatus === 'sending'}
                className="bg-purple-400 hover:bg-purple-500 text-white px-4 py-2 rounded transition-colors"
              >
                {emailStatus === 'sending' ? 'Sending...' : 'Email PDF'}
              </button>
              
              <button
                onClick={downloadPdf}
                className="bg-purple-300 hover:bg-purple-400 text-white px-4 py-2 rounded transition-colors"
              >
                Download PDF
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
});
