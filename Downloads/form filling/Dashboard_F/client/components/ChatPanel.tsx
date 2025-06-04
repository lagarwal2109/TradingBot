import React, { useState, useEffect, useRef, FormEvent } from 'react';
import { AiChatMessage } from '../types';
import ChatMessageDisplay from './ChatMessageDisplay';
import { PaperAirplaneIcon, XMarkIcon, SparklesIcon, ChevronDownIcon } from './icons';
import ActionButton from './ActionButton';
import { apiService } from '../services/api';

interface ChatPanelProps {
  T: any;
  onClose: () => void;
  onMinimize: () => void;
  messages: AiChatMessage[];
  setMessages: React.Dispatch<React.SetStateAction<AiChatMessage[]>>;
}

const ChatPanel: React.FC<ChatPanelProps> = ({ T, onClose, onMinimize, messages, setMessages }) => {
  const assistantLabels = T.mayaAiAssistant || {};
  const [userInput, setUserInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const chatContainerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSendMessage = async (e?: FormEvent<HTMLFormElement>) => {
    if (e) e.preventDefault();
    const trimmedInput = userInput.trim();
    if (!trimmedInput || isLoading) return;

    const userMessage: AiChatMessage = { id: crypto.randomUUID(), role: 'user', text: trimmedInput };
    setMessages(prevMessages => [...prevMessages, userMessage]);
    setUserInput('');
    setIsLoading(true);
    setError(null);

    const aiResponseId = crypto.randomUUID();
    setMessages(prevMessages => [...prevMessages, { id: aiResponseId, role: 'model', text: '', isLoading: true }]);

    try {
      // Build message history from previous messages (excluding the greeting)
      const conversationHistory: Array<{role: string, content: string}> = messages
        .filter(msg => 
          msg.role !== 'error' && 
          msg.text !== "Hi! I'm Maya, your MayaCode dashboard assistant. I can help you navigate your profile, workflows, documents, and job applications. What would you like to explore?" &&
          msg.text.trim() !== ''
        )
        .map(msg => ({
          role: msg.role === 'model' ? 'assistant' : 'user',
          content: msg.text
        }));

      // Add system message and current user message
      const messagesToSend = [
        {
          role: "system",
          content: "You are Maya, an intelligent AI assistant for the MayaCode User Dashboard. You have deep knowledge of this platform's features:\n\n- User Profile Management: Personal info, onboarding progress, profile export (PDF/JSON/TXT)\n- Workflow Navigation: Form Filling, Job Matching with multi-step processes\n- Document Management: View, preview, download, and send various document types\n- Suggested Actions: Personalized recommendations with filtering\n- Language Support: Multi-language content switching\n- Job Application Workflows: Application reviews and matching\n\nBe specific and actionable. Instead of generic responses, guide users to exact features. For example: 'Check your Profile section for completion status' or 'Use the Document Manager to upload your resume.' Keep responses under 2 sentences and always suggest the next logical step."
        },
        ...conversationHistory,
        { role: 'user', content: trimmedInput }
      ];

      // Use the backend API service
      const response = await apiService.chatWithAssistant(messagesToSend);

      setMessages(prevMessages =>
        prevMessages.map(msg =>
          msg.id === aiResponseId ? { ...msg, text: response, isLoading: false } : msg
        )
      );
    } catch (err) {
      console.error("Error sending message to backend:", err);
      let errorMessage = "I'm having trouble connecting right now. Please try again.";
      
      // Check for specific error types
      if (err instanceof Error) {
        if (err.message.includes('API key')) {
          errorMessage = assistantLabels.apiKeyMissingError || "Chat unavailable: API_KEY is not configured in the environment.";
        } else if (err.message.includes('rate limit')) {
          errorMessage = "I'm receiving too many requests right now. Please wait a moment and try again.";
        }
      }
      
      setMessages(prevMessages =>
        prevMessages.map(msg =>
          msg.id === aiResponseId ? { ...msg, role: 'error', text: errorMessage, isLoading: false } : msg
        )
      );
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  let placeholderText = assistantLabels.chatInputPlaceholder || "Ask Maya anything...";
  if (isLoading) {
    placeholderText = assistantLabels.typingIndicator || "Maya is typing...";
  }

  return (
    <div 
      className="fixed bottom-6 right-6 sm:bottom-8 sm:right-8 bg-card shadow-xl rounded-lg border border-border-color w-[calc(100%-3rem)] max-w-sm h-[calc(100%-6rem)] max-h-[500px] sm:max-h-[600px] flex flex-col z-50 transition-all duration-300 ease-in-out transform opacity-100 scale-100"
      role="dialog"
      aria-modal="true"
      aria-labelledby="chat-panel-title"
    >
      <header className="flex items-center justify-between p-3 border-b border-border-color bg-card-header rounded-t-lg">
        <div className="flex items-center">
          <SparklesIcon className="w-6 h-6 text-accent mr-2" />
          <h2 id="chat-panel-title" className="text-md font-semibold text-text-primary">
            {assistantLabels.chatWindowTitle || "Maya AI Assistant"}
          </h2>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={onMinimize}
            className="text-text-secondary hover:text-accent p-1 rounded-full"
            aria-label={assistantLabels.minimizeChatLabel || "Minimize chat"}
          >
            <ChevronDownIcon className="w-5 h-5" />
          </button>
        <button
          onClick={onClose}
          className="text-text-secondary hover:text-accent p-1 rounded-full"
          aria-label={assistantLabels.closeChatLabel || "Close chat"}
        >
          <XMarkIcon className="w-5 h-5" />
        </button>
        </div>
      </header>

      <div ref={chatContainerRef} className="flex-grow p-3 space-y-3 overflow-y-scroll bg-background h-96">
        {messages.map(msg => (
          <ChatMessageDisplay key={msg.id} message={msg} T={T} />
        ))}

      </div>

      {error && !messages.some(m => m.role === 'error' && m.text === error) && (
        <p className="p-3 text-sm text-status-red bg-red-50/50 border-t border-border-color">{error}</p>
      )}

      <form onSubmit={handleSendMessage} className="p-3 border-t border-border-color bg-card rounded-b-lg">
        <div className="flex items-center space-x-2">
          <input
            ref={inputRef}
            type="text"
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            placeholder={placeholderText}
            className="flex-grow p-2 border border-border-color rounded-md focus:ring-2 focus:ring-accent focus:border-accent text-sm text-text-primary bg-background placeholder-text-secondary disabled:bg-border-color/30 disabled:cursor-not-allowed"
            disabled={isLoading}
            aria-label="Chat input"
          />
          <ActionButton
            type="submit"
            disabled={isLoading || !userInput.trim()}
            icon={<PaperAirplaneIcon className="w-5 h-5" />}
            size="md"
            className="px-3"
            aria-label={assistantLabels.sendButtonLabel || "Send message"}
          >
            <span className="sr-only">{assistantLabels.sendButtonLabel || "Send"}</span>
          </ActionButton>
        </div>
      </form>
    </div>
  );
};

export default ChatPanel;
