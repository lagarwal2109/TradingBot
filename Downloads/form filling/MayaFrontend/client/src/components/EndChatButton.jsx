import React, { forwardRef } from 'react';
const API_ENDPOINT = "https://test.mayacode.io/api/api";

export const EndChatButton = forwardRef(({ jsonFilePath, onEndChat }, ref) => {
  const handleEndChat = async () => {
    try {
      const response = await fetch(`${API_ENDPOINT}/end-chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          endedAt: new Date().toISOString(),
          json_file_path: jsonFilePath
        }),
      });
      
      if (response.ok) {
        const result = await response.json();
        console.log('Chat ended:', result);
        
        // Call the onEndChat callback if provided
        onEndChat && onEndChat();
      } else {
        console.error('Failed to end chat');
      }
    } catch (error) {
      console.error('Error ending chat:', error);
    }
  };

  // This component is not actually used in the UI
  // The end chat functionality is in the UI.jsx component
  // using the handleEndChat function
  
  return (
    <button
      ref={ref}
      onClick={handleEndChat}
      className="pointer-events-auto bg-pink-500 hover:bg-pink-300 text-white p-4 rounded-md flex items-center justify-center gap-2 transition-colors"
      aria-label="End chat"
    >
      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6">
        <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
      </svg>
      <span>End Chat</span>
    </button>
  );
});
