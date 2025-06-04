import React, { useState, useEffect } from 'react';
import { StarIcon } from './icons'; // Previously used SparklesIcon was replaced with StarIcon
import ChatPanel from './ChatPanel';
import { AiChatMessage } from '../types';

interface MayaAiAssistantProps {
  T: any; // Translation object
  onChatHistoryClick?: (conversation: AiChatMessage[]) => void;
  loadConversation?: AiChatMessage[] | null;
}

const MayaAiAssistant: React.FC<MayaAiAssistantProps> = ({ T, onChatHistoryClick, loadConversation }) => {
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);
  const [chatMessages, setChatMessages] = useState<AiChatMessage[]>([
    { 
      id: crypto.randomUUID(), 
      role: 'model', 
      text: "Hi! I'm Maya, your MayaCode dashboard assistant. I can help you navigate your profile, workflows, documents, and job applications. What would you like to explore?",
    }
  ]);
  const assistantLabels = T.mayaAiAssistant || {};

  // Effect to handle loading predefined conversations
  useEffect(() => {
    if (loadConversation && loadConversation.length > 0) {
      setChatMessages(loadConversation);
      setIsChatOpen(true);
      setIsMinimized(false);
    }
  }, [loadConversation]);

  const openChat = () => {
    setIsChatOpen(true);
    setIsMinimized(false);
  };

  const closeChat = () => {
    setIsChatOpen(false);
    setIsMinimized(false);
    // Reset chat messages when closing
    setChatMessages([
      { 
        id: crypto.randomUUID(), 
        role: 'model', 
        text: "Hi! I'm Maya, your MayaCode dashboard assistant. I can help you navigate your profile, workflows, documents, and job applications. What would you like to explore?",
      }
    ]);
  };

  const minimizeChat = () => {
    setIsChatOpen(false);
    setIsMinimized(true);
    // Don't reset messages when minimizing
  };

  return (
    <>
      {!isChatOpen && !isMinimized && (
        <button
          onClick={openChat}
          className="fixed bottom-6 right-6 bg-accent text-white p-3 rounded-full shadow-lg hover:bg-accent-dark focus:outline-none focus:ring-2 focus:ring-accent focus:ring-offset-2 focus:ring-offset-background transition-transform duration-200 ease-in-out hover:scale-110 z-50"
          aria-label={assistantLabels.openChatLabel || "Open Maya AI Assistant"}
          title={assistantLabels.openChatLabel || "Open Maya AI Assistant"}
        >
          <StarIcon className="w-7 h-7" />
        </button>
      )}
      
      {!isChatOpen && isMinimized && (
        <button
          onClick={openChat}
          className="fixed bottom-6 right-6 bg-accent text-white px-4 py-2 rounded-full shadow-lg hover:bg-accent-dark focus:outline-none focus:ring-2 focus:ring-accent focus:ring-offset-2 focus:ring-offset-background transition-transform duration-200 ease-in-out hover:scale-105 z-50 flex items-center space-x-2"
          aria-label={assistantLabels.reopenChatLabel || "Reopen Maya AI Assistant"}
          title={assistantLabels.reopenChatLabel || "Reopen Maya AI Assistant"}
        >
          <StarIcon className="w-5 h-5" />
          <span className="text-sm font-medium">Maya</span>
        </button>
      )}
      
      {isChatOpen && (
        <ChatPanel 
          T={T} 
          onClose={closeChat}
          onMinimize={minimizeChat}
          messages={chatMessages}
          setMessages={setChatMessages}
        />
      )}
    </>
  );
};

export default MayaAiAssistant;
