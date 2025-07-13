import { Loader } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { Leva } from "leva";
import { Routes, Route, Navigate } from "react-router-dom";
import { Experience } from "./components/Experience";
import { UI } from "./components/UI";
import Login from "./pages/Login";
import { useAuth } from "./context/AuthContext";

function App() {
  const { isAuthenticated, loading } = useAuth();

  // Show loading state while checking authentication
  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-purple-500"></div>
      </div>
    );
  }

  return (
    <>
      <Loader />
      <Leva hidden />
      
      <Routes>
        <Route path="/login" element={isAuthenticated ? <Navigate to="/" /> : <Login />} />
        
        <Route 
          path="/" 
          element={
            isAuthenticated ? (
              <>
                <UI />
                <Canvas shadows camera={{ position: [0, 0, 1], fov: 30 }}>
                  <Experience />
                </Canvas>
              </>
            ) : (
              <Navigate to="/login" />
            )
          } 
        />
      </Routes>
    </>
  );
}

export default App;
