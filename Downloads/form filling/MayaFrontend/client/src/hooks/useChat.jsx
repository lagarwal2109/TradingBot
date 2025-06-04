import { createContext, useContext, useEffect, useState } from "react";

const ChatContext = createContext();

export const ChatProvider = ({ children }) => {
  const [messages, setMessages] = useState([]);
  const [message, setMessage] = useState(null);
  const [loading, setLoading] = useState(false);
  // Fixed camera zoom state - true = zoomed in, false = zoomed out
  const [cameraZoomed] = useState(true); // Set to your preferred fixed state

  const addMessage = (newMessages) => {
    setMessages((prevMessages) => [...prevMessages, ...newMessages]);
  };

  const onMessagePlayed = () => {
    setMessages((prevMessages) => prevMessages.slice(1));
  };

  useEffect(() => {
    if (messages.length > 0) {
      setMessage(messages[0]);
    } else {
      setMessage(null);
    }
  }, [messages]);

  return (
    <ChatContext.Provider
      value={{
        addMessage,
        message,
        onMessagePlayed,
        loading,
        setLoading,
        cameraZoomed, // Fixed zoom state
      }}
    >
      {children}
    </ChatContext.Provider>
  );
};

export const useChat = () => {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error("useChat must be used within a ChatProvider");
  }
  return context;
};
