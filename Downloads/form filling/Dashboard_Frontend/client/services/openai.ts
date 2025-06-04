import { apiService } from './api';

// Chat completion using the backend API instead of direct OpenAI calls
export const createChatCompletion = async (messages: { role: string, content: string }[]) => {
  try {
    // Use the backend's chat endpoint instead of direct OpenAI API
    const response = await apiService.chatWithAssistant(messages);
    return response;
  } catch (error) {
    console.error('Error in chat completion via backend:', error);
    throw error;
  }
};

// Legacy function for compatibility - now redirects to backend
export const createStreamingChatCompletion = async (messages: { role: string, content: string }[]) => {
  // Note: Backend doesn't support streaming yet, so we'll use regular completion
  return createChatCompletion(messages);
}; 