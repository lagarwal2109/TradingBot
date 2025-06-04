// API service for backend communication
import { ProgressStatus } from '../types';

const API_BASE_URL = 'https://test.mayacode.io/dash';

interface ApiResponse<T> {
  success: boolean;
  data: T;
  timestamp?: string;
  error?: string;
  message?: string;
}

interface PaginationInfo {
  currentPage: number;
  totalItems: number;
  itemsPerPage: number;
  totalPages: number;
  hasMore: boolean;
  nextPage: number | null;
  isInitialLoad: boolean;
}

interface ActionsResponse {
  actions: any[];
  pagination: PaginationInfo;
}

interface DocumentsResponse {
  documents: any[];
  pagination: PaginationInfo;
}

// Store cached user data to avoid multiple calls
let cachedUserData: any = null;

// Helper function to convert backend data to frontend format
const transformBackendData = (data: any) => {
  if (data.suggestedActions) {
    data.suggestedActions = data.suggestedActions.map((action: any) => ({
      ...action,
      status: convertStatusToEnum(action.status)
    }));
  }
  return data;
};

// Helper function to convert status strings to ProgressStatus enum
const convertStatusToEnum = (status: string): ProgressStatus => {
  switch (status) {
    case 'Not Started':
      return ProgressStatus.NotStarted;
    case 'In Progress':
      return ProgressStatus.InProgress;
    case 'Completed':
      return ProgressStatus.Completed;
    default:
      return ProgressStatus.NotStarted;
  }
};

class ApiService {
  private async fetchApi<T>(endpoint: string, options?: RequestInit): Promise<T> {
    try {
      // Create an abort controller with timeout to prevent hanging requests
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout
      
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        headers: {
          'Content-Type': 'application/json',
          ...options?.headers,
        },
        signal: controller.signal,
        ...options,
      });
      
      // Clear the timeout once request completes
      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data;
    } catch (error: unknown) {
      console.error('API request failed:', error);
      // Enhance error message for timeout or network errors
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          console.error('Request timed out - server may be unreachable');
        } else if (error.message && error.message.includes('Failed to fetch')) {
          console.error('Network error - server may be unreachable');
        }
      }
      throw error;
    }
  }

  // Get user data (profile) - matches backend /user endpoint
  async getUser() {
    const response = await this.fetchApi<ApiResponse<any>>('/user');
    if (response.success) {
      cachedUserData = transformBackendData(response.data);
      return response;
    }
    throw new Error('Failed to fetch user data');
  }

  // Get user actions with progressive loading - simulated from cached user data
  async getUserActions(params: {
    page?: number;
    limit?: number;
    status?: ProgressStatus | string;
    loadMore?: boolean;
  } = {}) {
    // Ensure we have user data
    if (!cachedUserData) {
      await this.getUser();
    }

    const actions = cachedUserData?.suggestedActions || [];
    const limit = params.limit || 10;
    const page = params.page || 1;
    
    // Filter by status if provided
    const filteredActions = params.status 
      ? actions.filter((action: any) => action.status === params.status)
      : actions;

    // Simulate pagination
    const startIndex = (page - 1) * limit;
    const endIndex = startIndex + limit;
    const paginatedActions = filteredActions.slice(startIndex, endIndex);

    const pagination: PaginationInfo = {
      currentPage: page,
      totalItems: filteredActions.length,
      itemsPerPage: limit,
      totalPages: Math.ceil(filteredActions.length / limit),
      hasMore: endIndex < filteredActions.length,
      nextPage: endIndex < filteredActions.length ? page + 1 : null,
      isInitialLoad: page === 1 && !params.loadMore
    };

    return {
      success: true,
      data: {
        actions: paginatedActions,
        pagination
      }
    };
  }

  // Get user documents with progressive loading - from cached user data
  async getUserDocuments(params: {
    page?: number;
    limit?: number;
    type?: string;
    loadMore?: boolean;
  } = {}) {
    if (!cachedUserData) {
      await this.getUser();
    }

    const documents = cachedUserData?.documents || [];
    const limit = params.limit || 10;
    const page = params.page || 1;
    
    // Filter by type if provided
    const filteredDocuments = params.type 
      ? documents.filter((doc: any) => doc.type === params.type)
      : documents;

    // Simulate pagination
    const startIndex = (page - 1) * limit;
    const endIndex = startIndex + limit;
    const paginatedDocuments = filteredDocuments.slice(startIndex, endIndex);

    const pagination: PaginationInfo = {
      currentPage: page,
      totalItems: filteredDocuments.length,
      itemsPerPage: limit,
      totalPages: Math.ceil(filteredDocuments.length / limit),
      hasMore: endIndex < filteredDocuments.length,
      nextPage: endIndex < filteredDocuments.length ? page + 1 : null,
      isInitialLoad: page === 1 && !params.loadMore
    };

    return {
      success: true,
      data: {
        documents: paginatedDocuments,
        pagination
      }
    };
  }

  // Get PDF as base64 from backend and return a Blob - matches backend /pdf endpoint
  async getPdfBlob() {
    const response = await this.fetchApi<ApiResponse<string>>('/pdf');
    if (response.success && response.data) {
      const byteCharacters = atob(response.data);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      return new Blob([byteArray], { type: 'application/pdf' });
    }
    throw new Error('Failed to fetch PDF');
  }

  // Chat with backend OpenAI endpoint - matches backend /chat endpoint
  async chatWithAssistant(messages: { role: string; content: string }[]) {
    const response = await this.fetchApi<any>('/chat', {
      method: 'POST',
      body: JSON.stringify({ 
        messages,
        model: "gpt-3.5-turbo",
        max_tokens: 150,
        temperature: 0.7
      }),
    });
    
    if (response.response) {
      return response.response;
    }
    throw new Error('Failed to get assistant response');
  }
}

export const apiService = new ApiService();
export type { ApiResponse, PaginationInfo, ActionsResponse, DocumentsResponse }; 