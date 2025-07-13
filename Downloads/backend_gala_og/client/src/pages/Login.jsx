import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { GoogleLogin } from '@react-oauth/google';
import { useAuth } from '../context/AuthContext';

const Login = () => {
  const { login, isAuthenticated } = useAuth();
  const navigate = useNavigate();
  
  // Redirect if already logged in
  useEffect(() => {
    if (isAuthenticated) {
      navigate('/');
    }
  }, [isAuthenticated, navigate]);
  
  // Google login handlers are now embedded in the GoogleLogin component
  
  return (
    <div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-purple-50 to-pink-50">
      <div className="bg-white p-8 rounded-lg shadow-xl w-full max-w-md text-center">
        <h1 className="text-2xl font-bold text-purple-600 mb-6">Welcome to Maya</h1>
        <p className="text-gray-600 mb-8">
          Please sign in with your Google account to continue
        </p>
        
        {/* Google Sign-In Button */}
        <div className="flex justify-center">
          <GoogleLogin
            onSuccess={async (credentialResponse) => {
              console.log('Google Login Success:', credentialResponse);
              const success = await login(credentialResponse);
              if (success) {
                navigate('/');
              }
            }}
            onError={() => {
              console.error('Google Login Failed');
            }}
            useOneTap
            shape="pill"
            theme="filled_blue"
            text="signin_with"
            size="large"
          />
        </div>
      </div>
    </div>
  );
};

export default Login;
