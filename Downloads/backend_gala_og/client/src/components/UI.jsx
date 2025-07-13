import { useState, useEffect, useRef } from "react";
import { useChat } from "../hooks/useChat";
import { useAuth } from "../context/AuthContext";
import { VoiceRecorder } from "./VoiceRecorder";
import { UserProfileDrawer } from "./UserProfileDrawer";
import { LanguageSelector } from "./LanguageSelector";
import { ProgressBar } from "./ProgressBar"; // Import the progress bar component
import { RecommendationDrawer } from "./RecommendationDrawer"; // Import the recommendation drawer component
import { apiService } from "../services/api";  // Import the API service

// Define a consistent button style to use across components
const buttonStyle = "bg-purple-400 hover:bg-purple-500 text-white p-4 rounded-md flex items-center justify-center";

export const UI = ({ hidden, jsonFilePath = "./chat_data.json", ...props }) => {
  const { chat, loading, cameraZoomed, setCameraZoomed, message } = useChat();
  const { user, logout, isAuthenticated } = useAuth();
  const [isLoggingOut, setIsLoggingOut] = useState(false);
  
  // Simple state for language selection without persistence
  const [currentLanguage, setCurrentLanguage] = useState("auto");
  
  // State for tracking progress steps (default to step 1)
  const [currentStep, setCurrentStep] = useState(1);
  
  // State for thank you message
  const [showThankYouMessage, setShowThankYouMessage] = useState(false);
  
  // PDF viewer states
  const [pdfData, setPdfData] = useState(null);
  const [pdfFilename, setPdfFilename] = useState('filled_form.pdf');
  const [pdfLoading, setPdfLoading] = useState(false);
  const [showPdfModal, setShowPdfModal] = useState(false);
  const [pdfMessage, setPdfMessage] = useState(null);
  const [pdfEmailStatus, setPdfEmailStatus] = useState(null);
  const [translationStatus, setTranslationStatus] = useState(null);
  
  // Recommendation drawer states
  const [isRecommendationOpen, setIsRecommendationOpen] = useState(false);
  const [recommendation, setRecommendation] = useState("");
  const [isRecommendationLoading, setIsRecommendationLoading] = useState(false);
  const [recommendationError, setRecommendationError] = useState(null);
  const recommendationDrawerRef = useRef(null);
  
  // Utility to convert literal \n sequences into actual line breaks
  const normalizeContent = (raw) =>
    raw.replace(/\\n\\n/g, "\n\n").replace(/\\n/g, "\n");
  
  // Fetch recommendation data from the backend endpoint
  const fetchRecommendation = async () => {
    try {
      // First open the drawer with loading animation
      setIsRecommendationOpen(true);
      setIsRecommendationLoading(true);
      setRecommendationError(null);
      
      // Set up a timeout for the request
      const timeoutPromise = new Promise((_, reject) => {
        setTimeout(() => {
          reject(new Error('Request timeout after 20 seconds. Please try again.'));
        }, 20000); // 20 seconds timeout
      });
      
      // The actual fetch request
      const response = await Promise.race([apiService.recommendations.getRecommendations(), timeoutPromise]);
      
      // Backend returns the recommendation string directly in response.data
      const recommendationText = response.data;
      
      if (recommendationText && typeof recommendationText === 'string') {
        setRecommendation(recommendationText);
      } else {
        throw new Error("No recommendation data received");
      }
    } catch (err) {
      setRecommendationError(err.message);
    } finally {
      setIsRecommendationLoading(false);
    }
  };
  
  // Convert text to JSX with paragraphs and line breaks
  const formatContent = (content) => {
    if (!content) return null;
    const normalized = normalizeContent(content);
    const blocks = normalized.split(/\n\n+/);
    return blocks.map((block, i) => {
      const lines = block.split(/\n/);
      // Header check
      if (block.trim().startsWith('**') && block.trim().endsWith('**') && block.split('**').length === 3) {
        return (
          <h2 key={i} className="text-lg font-bold mt-4 mb-2">
            {processParagraph(block.trim().replace(/^\*\*|\*\*$/g, ''))}
          </h2>
        );
      }
      const processedLines = lines.map((line, idx) => (
        <span key={`${i}-${idx}`}>
          {processParagraph(line)}{idx < lines.length - 1 && <br />}
        </span>
      ));
      return <p key={i} className="mb-3">{processedLines}</p>;
    });
  };

  // Process bold and links in recommendation text
  const processParagraph = (text) => {
    if (!text) return null;
    const segments = [];
    let remaining = text;
    const linkRegex = /\[([^\]]+)\]\(([^)]+)\)/;
    const boldRegex = /\*\*([^*]+)\*\*/;
    while (remaining) {
      const linkMatch = remaining.match(linkRegex);
      const boldMatch = remaining.match(boldRegex);
      const linkIndex = linkMatch?.index ?? Infinity;
      const boldIndex = boldMatch?.index ?? Infinity;
      if (linkIndex === Infinity && boldIndex === Infinity) {
        segments.push({ type: 'text', content: remaining });
        break;
      }
      if (linkIndex < boldIndex) {
        if (linkIndex > 0) segments.push({ type: 'text', content: remaining.slice(0, linkIndex) });
        segments.push({ type: 'link', text: linkMatch[1], url: linkMatch[2] });
        remaining = remaining.slice(linkIndex + linkMatch[0].length);
      } else {
        if (boldIndex > 0) segments.push({ type: 'text', content: remaining.slice(0, boldIndex) });
        segments.push({ type: 'bold', content: boldMatch[1] });
        remaining = remaining.slice(boldIndex + boldMatch[0].length);
      }
    }
    return segments.map((seg, idx) => {
      if (seg.type === 'text') return seg.content;
      if (seg.type === 'link') return (
        <a key={idx} href={seg.url} target="_blank" rel="noopener noreferrer" className="text-purple-400 hover:text-purple-500 underline">
          {seg.text}
        </a>
      );
      if (seg.type === 'bold') return <strong key={idx} className="font-bold">{seg.content}</strong>;
      return null;
    });
  };
  
  // Function to fetch PDF form
  const fetchPdf = async () => {
    setPdfLoading(true);
    setPdfMessage(null);
    setPdfEmailStatus(null);
    setTranslationStatus(null); // Reset translation status on new fetch
    
    try {
      const response = await apiService.pdf.showPdf();
      
      if (response.data.status === 'success' && response.data.pdf_data) {
        setPdfData(response.data.pdf_data);
        setPdfFilename(response.data.pdf_filename || 'filled_form.pdf');
        setShowPdfModal(true);
      } else {
        setPdfMessage(`Error: ${response.data.message || 'Please try later'}`);
        // Clear the error message after 3 seconds
        setTimeout(() => {
          setPdfMessage(null);
        }, 3000);
      }
    } catch (error) {
      setPdfMessage(`Error: ${error.response?.data?.message || error.message || 'Please try later'}`);
      // Clear the error message after 3 seconds
      setTimeout(() => {
        setPdfMessage(null);
      }, 3000);
    } finally {
      setPdfLoading(false);
    }
  };
  
  // Function to send PDF via email
  const sendPdfEmail = async () => {
    setPdfEmailStatus('sending');
    setPdfMessage(null);
    
    try {
      const response = await apiService.pdf.sendPdf();
      
      if (response.data.status === 'success') {
        setPdfEmailStatus('success');
        setPdfMessage(response.data.message || 'PDF sent via email');
        // Wait for a moment to show the success animation before closing
        setTimeout(() => {
          setShowPdfModal(false);
        }, 2000);
        // Clear the message after 3 seconds
        setTimeout(() => {
          setPdfMessage(null);
        }, 3000);
      } else {
        setPdfEmailStatus('error');
        setPdfMessage(`Error: ${response.data.message || 'Service is down'}`);
        // Clear the error message after 3 seconds
        setTimeout(() => {
          setPdfMessage(null);
        }, 3000);
      }
    } catch (error) {
      setPdfEmailStatus('error');
      setPdfMessage(`Error: ${error.response?.data?.message || error.message || 'Service is down'}`);
      // Clear the error message after 3 seconds
      setTimeout(() => {
        setPdfMessage(null);
      }, 3000);
    }
  };
  
  // Function to download the PDF
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
    } catch (error) {
      setPdfMessage(`Error downloading PDF: ${error.message}`);
    }
  };
  
  // Function to close the PDF modal
  const closePdfModal = () => {
    setShowPdfModal(false);
  };

  // Function to handle PDF translation
  const handleTranslate = async () => {
    setTranslationStatus('translating');
    setPdfMessage(null); // Clear previous messages
    try {
      const response = await apiService.pdf.translatePdf();
      if (response.data.status === 'success' && response.data.pdf_data) {
        setPdfData(response.data.pdf_data); // Update the PDF data with the translated version
        setTranslationStatus('success');
        setPdfMessage('PDF translated successfully!');
        setTimeout(() => {
          setTranslationStatus(null);
          setPdfMessage(null);
        }, 3000);
      } else {
        setTranslationStatus('error');
        setPdfMessage(`Error: ${response.data.message || 'Translation failed. Please try later.'}`);
        setTimeout(() => {
          setTranslationStatus(null);
          setPdfMessage(null);
        }, 3000);
      }
    } catch (error) {
      setTranslationStatus('error');
      setPdfMessage(`Error: ${error.response?.data?.message || error.message || 'Translation failed. Please try later.'}`);
      setTimeout(() => {
        setTranslationStatus(null);
        setPdfMessage(null);
      }, 3000);
    }
  };
  
  // Function to handle ending the chat
  const handleEndChat = async () => {
    try {
      const response = await apiService.chat.endChat({ endedAt: new Date().toISOString(), json_file_path: jsonFilePath });
      
      if (response.data.success) {
        // Show thank you message
        setShowThankYouMessage(true);
        
        // Dispatch a custom event for UserProfileDrawer to listen for
        // This will trigger the location refresh after 10 seconds
        const chatEndEvent = new CustomEvent('chat-ended', { detail: { timestamp: new Date().toISOString() } });
        window.dispatchEvent(chatEndEvent);
      } else {
        // End chat failed silently
      }
    } catch (error) {
      // Error handled silently
    }
  };
  
  if (hidden) {
    return null;
  }
  
  return (
    <div className="fixed top-0 left-0 right-0 bottom-0 z-10 flex justify-between flex-col pointer-events-none">  
      {/* User Profile and Logout Section */}
      {isAuthenticated && user && (
        <div className="absolute top-4 right-4 z-50 flex items-center gap-3 pointer-events-auto">
          <div className="bg-white/90 backdrop-blur-sm rounded-lg shadow-lg p-2 flex items-center gap-2">
            <div className="w-8 h-8 rounded-full bg-purple-100 flex items-center justify-center">
              <span className="text-sm font-medium text-purple-600">
                {user.name?.charAt(0) || user.email?.charAt(0) || '?'}
              </span>
            </div>
            <div className="pr-2">
              <p className="text-sm font-medium text-purple-700">
                {user.name || user.email}
              </p>
            </div>
            <button 
              onClick={async () => {
                setIsLoggingOut(true);
                await logout();
                setIsLoggingOut(false);
              }} 
              disabled={isLoggingOut}
              className={`ml-2 px-2 py-1 ${isLoggingOut ? 'bg-purple-50 text-purple-300' : 'bg-purple-100 hover:bg-purple-200 text-purple-700'} text-xs rounded transition-colors`}
              title="Sign out"
            >
              {isLoggingOut ? 'Logging out...' : 'Logout'}
            </button>
          </div>
        </div>
      )}

      {/* Progress Bar at the top */}
      <div className="w-full pointer-events-auto">
        <ProgressBar 
          currentStep={currentStep} 
          onStepClick={(stepId) => {
            setCurrentStep(stepId);
            
            // If step 2 (Personal Data - Form) is clicked
            if (stepId === 2) {
              // Call the PDF fetch function directly
              fetchPdf();
            }
            // If step 3 (Documents - Recommendations) is clicked
            else if (stepId === 3) {
              // Call the recommendation fetch function directly
              fetchRecommendation();
            }
            // If step 4 (End Session) is clicked
            else if (stepId === 4) {
              // Call the end chat function directly
              handleEndChat();
            }
          }}
        />
      </div>
      
      {/* Thank You Message */}
      {showThankYouMessage && (
        <div className="fixed inset-0 flex items-center justify-center z-50 bg-black/50 pointer-events-auto">
          <div className="bg-white p-8 rounded-lg shadow-lg text-center max-w-md">
            <h2 className="text-2xl font-bold text-purple-600 mb-4">Thanks for talking!</h2>
            <p className="text-gray-700 mb-6">We hope you enjoyed the conversation. Have a great day!</p>
            <button
              onClick={() => setShowThankYouMessage(false)}
              className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 transition-colors"
            >
              Close
            </button>
          </div>
        </div>
      )}
      
      {/* PDF Modal */}
      {showPdfModal && pdfData && (
        <div 
          className="fixed inset-0 bg-gradient-to-br from-purple-100 to-pink-100 bg-opacity-95 flex items-center justify-center z-50 p-4 pointer-events-auto"
          onClick={closePdfModal} // Close when clicking the backdrop
        >
          <div 
            className="bg-white rounded-lg shadow-xl w-full max-w-4xl flex flex-col max-h-screen border border-purple-100"
            onClick={(e) => e.stopPropagation()} // Prevent closing when clicking the modal itself
          >
            {/* Modal Header */}
            <div className="p-4 border-b border-purple-100 bg-purple-50 rounded-t-lg flex justify-between items-center">
              <h3 className="text-lg font-semibold">Form Preview</h3>
              <button 
                onClick={closePdfModal}
                className="text-gray-500 hover:text-gray-700 text-3xl font-light leading-none"
                aria-label="Close"
              >
                &times;
              </button>
            </div>
            
            {/* PDF Viewer with Email Sending Animation Overlay */}
            <div className="flex-grow overflow-auto p-4 relative">
              {pdfEmailStatus === 'sending' && (
                <div className="absolute inset-0 bg-gradient-to-br from-purple-50 to-pink-50 bg-opacity-95 flex flex-col items-center justify-center z-10">
                  <div className="relative mb-6">
                    {/* Animated envelope */}
                    <div className="w-20 h-16 bg-white rounded-lg border-2 border-purple-300 shadow-md relative overflow-hidden">
                      {/* Animated letter */}
                      <div className="absolute top-1 left-1 right-1 h-8 bg-gradient-to-r from-purple-200 to-purple-300 rounded animate-pulse"></div>
                      {/* Envelope flap */}
                      <div className="absolute top-0 left-0 right-0 h-0 border-b-[16px] border-l-[10px] border-r-[10px] border-b-purple-300 border-l-transparent border-r-transparent transform origin-top"></div>
                    </div>
                    {/* Animated circles */}
                    <div className="absolute -top-1 -right-1 w-3 h-3 bg-purple-400 rounded-full animate-ping opacity-75"></div>
                    <div className="absolute -bottom-1 -left-1 w-2 h-2 bg-pink-300 rounded-full animate-ping opacity-75 animation-delay-300"></div>
                  </div>
                  <div className="text-lg font-medium text-purple-700 mb-2">Sending PDF</div>
                  <div className="text-sm text-purple-500 flex items-center">
                    <span>Your document is being emailed</span>
                    <span className="ml-2 flex">
                      <span className="animate-bounce mx-px delay-0">.</span>
                      <span className="animate-bounce mx-px delay-150">.</span>
                      <span className="animate-bounce mx-px delay-300">.</span>
                    </span>
                  </div>
                </div>
              )}
              
              {pdfEmailStatus === 'success' && (
                <div className="absolute inset-0 bg-gradient-to-br from-purple-50 to-pink-50 bg-opacity-95 flex flex-col items-center justify-center z-10">
                  <div className="relative mb-6">
                    {/* Success checkmark with animation */}
                    <div className="w-20 h-20 bg-white rounded-full border-2 border-purple-300 shadow-md flex items-center justify-center">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 text-purple-400 animate-[bounce_1s_ease-in-out]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
                      </svg>
                    </div>
                    {/* Animated celebration particles */}
                    <div className="absolute -top-2 -right-2 w-4 h-4 bg-purple-300 rounded-full animate-ping opacity-75"></div>
                    <div className="absolute top-1/2 -right-3 w-3 h-3 bg-pink-300 rounded-full animate-ping opacity-75 delay-300"></div>
                    <div className="absolute -bottom-1 -left-2 w-3 h-3 bg-purple-400 rounded-full animate-ping opacity-75 delay-500"></div>
                  </div>
                  <div className="text-lg font-medium text-purple-700 mb-2">Email Sent!</div>
                  <div className="text-sm text-purple-500">Your PDF has been sent successfully</div>
                </div>
              )}
              
              {pdfEmailStatus === 'error' && (
                <div className="absolute inset-0 bg-gradient-to-br from-red-50 to-pink-50 bg-opacity-95 flex flex-col items-center justify-center z-10">
                  <div className="relative mb-6">
                    {/* Error icon with animation */}
                    <div className="w-20 h-20 bg-white rounded-full border-2 border-red-300 shadow-md flex items-center justify-center">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </div>
                    {/* Animated error particle effect */}
                    <div className="absolute -top-1 -right-1 w-3 h-3 bg-red-300 rounded-full animate-pulse opacity-75"></div>
                    <div className="absolute -bottom-1 -left-1 w-3 h-3 bg-red-400 rounded-full animate-pulse opacity-75 delay-300"></div>
                  </div>
                  <div className="text-lg font-medium text-red-600 mb-2">Email Failed</div>
                  <div className="text-sm text-red-500">{pdfMessage || 'Something went wrong. Please try again.'}</div>
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
                disabled={pdfEmailStatus === 'sending'}
                className="bg-purple-400 hover:bg-purple-500 text-white px-4 py-2 rounded transition-colors"
              >
                {pdfEmailStatus === 'sending' ? 'Sending...' : 'Email PDF'}
              </button>
              
              <button
                onClick={downloadPdf}
                className="bg-purple-300 hover:bg-purple-400 text-white px-4 py-2 rounded transition-colors"
              >
                Download PDF
              </button>

              {/* Translate PDF Button with Loading Animation */}
              <button
                onClick={handleTranslate}
                disabled={translationStatus === 'translating'}
                className="bg-purple-400 hover:bg-purple-500 text-white px-4 py-2 rounded transition-colors disabled:bg-purple-300 disabled:cursor-not-allowed flex items-center justify-center"
              >
                {translationStatus === 'translating' ? (
                  <>
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Translating...
                  </>
                ) : (
                  'Translate'
                )}
              </button>
            </div>
          </div>
        </div>
      )}
      
      {/* Message Display for PDF */}
      {pdfMessage && (
        <div className={`fixed bottom-4 left-1/2 transform -translate-x-1/2 z-50 p-4 rounded-lg shadow-lg ${pdfMessage.includes('Error') ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'}`}>
          {pdfMessage}
        </div>
      )}
      
      {/* Use the enhanced RecommendationDrawer component */}
      <RecommendationDrawer 
        ref={recommendationDrawerRef}
        isOpen={isRecommendationOpen}
        setIsOpen={setIsRecommendationOpen}
        recommendation={recommendation}
        isLoading={isRecommendationLoading}
        error={recommendationError}
      />
      
      {/* Middle section */}
      <div className="w-full flex justify-between items-center">
        {/* Left side buttons */}
        <div className="flex flex-col space-y-4">
          
          {/* PDFViewer functionality has been integrated directly into the UI component */}
          
          {/* RecommendationDrawer functionality has been integrated directly into the UI component */}
        </div>
        
        {/* Right side buttons - stacked vertically and aligned */}
        <div className="flex flex-col space-y-4">
          {/* No buttons here anymore */}
        </div>
        
        {/* User Profile Drawer with Language Selector */}
        <div className="pointer-events-auto absolute right-[5%] top-1/2 transform -translate-y-1/3">
          <UserProfileDrawer 
            currentLanguage={currentLanguage}
            setCurrentLanguage={setCurrentLanguage}
          />
        </div>
      </div>
      
      {/* Dashboard button - bottom left */}
      <div className="pointer-events-auto absolute left-4 bottom-4">
        <a 
          href="https://dashboard.mayacode.io" 
          target="_blank" 
          rel="noopener noreferrer" 
          className={`${buttonStyle} text-sm px-3 py-2 shadow-md transition-transform hover:scale-105 flex items-center space-x-2`}
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
            <path d="M2 4a1 1 0 011-1h2a1 1 0 011 1v12a1 1 0 01-1 1H3a1 1 0 01-1-1V4zM8 4a1 1 0 011-1h2a1 1 0 011 1v12a1 1 0 01-1 1H9a1 1 0 01-1-1V4zM15 3a1 1 0 00-1 1v12a1 1 0 001 1h2a1 1 0 001-1V4a1 1 0 00-1-1h-2z" />
          </svg>
          <span>Dashboard</span>
        </a>
      </div>
      
      {/* Bottom section - Voice recorder */}
      <div className="pointer-events-auto w-full flex justify-center mt-auto mb-4">
        <VoiceRecorder
          disabled={loading || message}
          onTranscriptionComplete={(data) => {
            if (data.messages) {
              chat(data.messages);
            }
          }}
        />
      </div>
      {translationStatus === 'translating' && (
        <div className="fixed inset-0 bg-purple-50 bg-opacity-90 flex flex-col items-center justify-center z-50">
          <svg className="animate-spin h-20 w-20 text-purple-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          <p className="mt-6 text-purple-700 text-xl font-semibold animate-pulse">Translating your PDF...</p>
        </div>
      )}
    </div>
  );
};