import { createContext, useState, useEffect, useContext } from 'react';
import { apiService } from '../services/api';

// Create the auth context
export const AuthContext = createContext();

// Provider component
export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  // Check for existing session on mount
  useEffect(() => {
    const checkAuth = async () => {
      try {
        // Check if user is logged in by verifying cookie with backend
        const response = await apiService.auth.checkSession();
        
        if (response.data.authenticated) {
          // User is authenticated, set user data
          setUser(response.data.user);
        } else {
          // No active session
          setUser(null);
        }
      } catch (error) {
        console.error('Session check error:', error);
        setUser(null);
      } finally {
        setLoading(false);
      }
    };
    
    checkAuth();
  }, []);

  // Login function - handles Google token
  const login = async (googleData) => {
    try {
      console.log('Login with Google data:', googleData);
      // Backend will set the cookie automatically in the response
      const response = await apiService.auth.googleLogin({
        // Send either credential or token based on what's available
        credential: googleData.credential,
        token: googleData.access_token
      });
      
      if (response.data.success) {
        // Get user details from session check
        const sessionResponse = await apiService.auth.checkSession();
        setUser(sessionResponse.data.user);
        return true;
      }
      return false;
    } catch (error) {
      console.error('Login error:', error);
      return false;
    }
  };

  // Logout function - calls backend to clear cookie
  const logout = async () => {
    try {
      // Call logout endpoint which will clear the cookie
      await apiService.auth.logout();
      setUser(null);
    } catch (error) {
      console.error('Logout error:', error);
    }
  };

  // Context value that will be provided to consumers
  const authValue = {
    user,
    loading,
    login,
    logout,
    isAuthenticated: !!user
  };

  return (
    <AuthContext.Provider value={authValue}>
      {children}
    </AuthContext.Provider>
  );
};

// Custom hook for easy context consumption
export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
