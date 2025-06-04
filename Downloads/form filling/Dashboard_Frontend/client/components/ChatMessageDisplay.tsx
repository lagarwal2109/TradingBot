
import React from 'react';
import { AiChatMessage } from '../types';
import { UserCircleIcon, SparklesIcon } from './icons'; // Assuming SparklesIcon for AI

interface ChatMessageDisplayProps {
  message: AiChatMessage;
  T: any; // For potential future internationalization of message elements
}

const ChatMessageDisplay: React.FC<ChatMessageDisplayProps> = ({ message, T }) => {
  const isUser = message.role === 'user';
  const isError = message.role === 'error';
  const assistantLabels = T.mayaAiAssistant || {};


  // Basic Markdown-like link detection: [text](url)
  const formatTextWithLinks = (text: string) => {
    const linkRegex = /\[([^\]]+)]\((https?:\/\/[^\s)]+)\)/g;
    const parts = [];
    let lastIndex = 0;
    let match;

    while ((match = linkRegex.exec(text)) !== null) {
      // Add text before the link
      if (match.index > lastIndex) {
        parts.push(text.substring(lastIndex, match.index));
      }
      // Add the link
      const linkText = match[1];
      const linkUrl = match[2];
      parts.push(
        <a
          key={linkUrl + match.index}
          href={linkUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="text-accent hover:underline"
        >
          {linkText}
        </a>
      );
      lastIndex = linkRegex.lastIndex;
    }

    // Add any remaining text after the last link
    if (lastIndex < text.length) {
      parts.push(text.substring(lastIndex));
    }
    
    // Join parts, handling newlines
    return parts.reduce((acc, part, index) => {
        if (typeof part === 'string') {
            return [...acc, ...part.split('\n').map((line, i) => (
                <React.Fragment key={`${index}-${i}`}>
                    {line}
                    {i < part.split('\n').length - 1 && <br />}
                </React.Fragment>
            ))];
        }
        return [...acc, part];
    }, [] as React.ReactNode[]);

  };


  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} w-full`}>
      <div className={`flex items-start space-x-2 max-w-[85%] sm:max-w-[75%]`}>
        {!isUser && !isError && (
          <div className="flex-shrink-0 w-7 h-7 rounded-full bg-accent/20 flex items-center justify-center mt-1">
            <SparklesIcon className="w-4 h-4 text-accent" />
          </div>
        )}
        {isUser && (
           <div className="flex-shrink-0 w-7 h-7 rounded-full bg-secondary-accent/20 flex items-center justify-center mt-1 order-2">
            <UserCircleIcon className="w-4 h-4 text-secondary-accent" />
          </div>
        )}

        <div 
          className={`
            px-3 py-2 rounded-lg
            ${isUser ? 'bg-accent text-white order-1' : (isError ? 'bg-status-red/20 text-status-red' : 'bg-card border border-border-color text-text-primary')}
          `}
        >
          {message.isLoading && message.text === "" ? (
            <span className="italic text-sm">{assistantLabels.typingIndicator || "Maya is typing..."}</span>
          ) : (
             <p className="text-sm whitespace-pre-wrap">{formatTextWithLinks(message.text)}</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default ChatMessageDisplay;
