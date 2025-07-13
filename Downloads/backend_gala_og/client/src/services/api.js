import axios from 'axios';

// Base URL for all API requests
// Use local development server instead of production URL
const API_URL = 'https://test.mayacode.io/api/api';

// Create axios instance with withCredentials to send cookies
const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: true, // This enables cookies to be sent with requests
});

// Response interceptor - handles unauthorized responses
api.interceptors.response.use(
  (response) => response,
  (error) => {
    // If 401 Unauthorized, redirect to login
    if (error.response && error.response.status === 401) {
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// API service with endpoint methods organized by feature
export const apiService = {
  // Auth endpoints
  auth: {
    googleLogin: (googleData) => api.post('/auth/google', googleData),
    checkSession: () => api.get('/auth/session'),
    logout: () => api.post('/auth/logout'),
  },
  
  // User data endpoints
  user: {
    getProfile: () => api.get('/user/profile'),
    updateUserPersona: (data) => api.post('/update-user-persona', { data }),
    getUserPersona: () => api.get('/get-user-persona'),
  },
  
  // Chat endpoints
  chat: {
    sendTranscription: (formData) => {
      const config = {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      };
      return api.post('/transcribe', formData, config);
    },
    endChat: (data) => api.post('/end-chat', data),
  },
  
  // PDF operations
  pdf: {
    getPdf: (action) => {
      console.log(`🔍 API: Making PDF request with action: ${action}`);
      return api.get(`/get-pdf?action=${action}`, { timeout: 60000 }); // 60 second timeout for PDF
    },
    sendPdfEmail: () => {
      console.log('🔍 API: Sending PDF email request');
      return api.get('/get-pdf?action=send', { timeout: 60000 });
    },
    showPdf: () => {
      console.log('🔍 API: Requesting PDF show');
      return api.get('/get-pdf?action=show', { timeout: 60000 }); // 60 second timeout for PDF
    },
    getForm: () => {
      console.log('🔍 API: Getting PDF form');
      return api.get('/get-pdf?action=show', { timeout: 60000 });
    },
    sendEmail: () => {
      console.log('🔍 API: Sending PDF email (alt method)');
      return api.get('/get-pdf?action=send', { timeout: 60000 });
    },
    sendPdf: () => {
      console.log('🔍 API: Sending PDF (alt method)');
      return api.get('/get-pdf?action=send', { timeout: 60000 });
    },
    translatePdf: () => {
      console.log('🔍 API: Requesting PDF translation');
      return api.get('/get-pdf?action=translate', { timeout: 120000 }); // 120-second timeout for translation
    },
  },
  
  // Language settings
  language: {
    setLanguage: (language) => api.post('/set-language', { language }),
  },
  
  // Recommendations
  recommendations: {
    getRecommendations: () => api.post('/recommendation'),
  }
};

export default api;
