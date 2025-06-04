import { useState, useRef } from "react";
import { useChat } from "../hooks/useChat"; // Import the useChat hook

export const VoiceRecorder = ({ disabled }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [warning, setWarning] = useState(null); // State for warning messages
  const mediaRecorder = useRef(null);
  const audioChunks = useRef([]);
  const timerInterval = useRef(null);
  const warningTimeout = useRef(null); // Ref for warning timeout
  const backendUrl = import.meta.env.VITE_API_URL || "https://test.mayacode.io/api/api"; // Use environment variable or default URL

  const { addMessage, setLoading } = useChat(); // Use chat hook

  const startRecording = async () => {
    // Clear any previous warnings and timeouts when starting a new recording
    setWarning(null);
    if (warningTimeout.current) {
      clearTimeout(warningTimeout.current);
      warningTimeout.current = null;
    }
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder.current = new MediaRecorder(stream);
      audioChunks.current = [];

      mediaRecorder.current.ondataavailable = (event) => {
        audioChunks.current.push(event.data);
      };

      mediaRecorder.current.onstop = async () => {
        // Create audio blob from chunks
        const audioBlob = new Blob(audioChunks.current, { type: "audio/wav" });
        
        // Simple check - just log the audio size
        console.log("Audio blob size (bytes):", audioBlob.size);
        
        // Only check if audio is less than 1KB (1000 bytes)
        if (audioBlob.size < 1000) {
          setWarning("No audio detected. Please try again and speak clearly.");
          console.log("Warning triggered: Audio blob too small (< 1KB)");
          
          // Clear any existing timeout
          if (warningTimeout.current) {
            clearTimeout(warningTimeout.current);
          }
          
          // Set timeout to clear warning after 3.5 seconds
          warningTimeout.current = setTimeout(() => {
            setWarning(null);
            warningTimeout.current = null;
          }, 3500);
          
          return;
        }
        
        const formData = new FormData();
        formData.append("audio", audioBlob, "recording.wav");

        setLoading(true); // Show loading state in chat

        try {
          const response = await fetch(`${backendUrl}/transcribe`, {
            method: "POST",
            body: formData,
          });
          const data = await response.json();

          if (data?.messages) {
            addMessage(data.messages); // Add transcribed messages to chat
          }
        } catch (error) {
          console.error("Error processing audio:", error);
        } finally {
          setLoading(false); // Stop loading state
        }
      };

      // Start recording and ensure time tracking works properly
      mediaRecorder.current.start();
      setIsRecording(true);
      setRecordingTime(0); // Reset timer to ensure it starts from 0
      
      // Use a more reliable timer approach
      const startTime = Date.now();
      timerInterval.current = setInterval(() => {
        const elapsedSeconds = Math.floor((Date.now() - startTime) / 1000);
        setRecordingTime(elapsedSeconds);
        console.log("Recording time updated:", elapsedSeconds);
      }, 1000);
    } catch (error) {
      console.error("Error accessing microphone:", error);
    }
  };

  const stopRecording = () => {
    if (mediaRecorder.current && isRecording) {
      mediaRecorder.current.stop();
      mediaRecorder.current.stream.getTracks().forEach((track) => track.stop());
      setIsRecording(false);
      clearInterval(timerInterval.current);
      setRecordingTime(0);
    }
  };

  return (
    <div className="relative">
      {/* Warning message */}
      {warning && (
        <div className="absolute -top-14 left-1/2 transform -translate-x-1/2 bg-purple-200 text-white px-3 py-1 rounded-md text-sm whitespace-nowrap">
          {warning}
        </div>
      )}
      
      <button
        disabled={disabled}
        onClick={isRecording ? stopRecording : startRecording}
        className={`w-16 h-16 rounded-full flex items-center justify-center transition-all ${
          isRecording
            ? "bg-purple-200 hover:bg-purple-200 scale-110"
            : "bg-purple-400 hover:bg-purple-500"
        } ${disabled ? "opacity-50 cursor-not-allowed" : ""}`}
      >
        {isRecording ? (
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-8 w-8 text-white"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <rect x="6" y="6" width="12" height="12" fill="currentColor" />
          </svg>
        ) : (
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-8 w-8 text-white"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
            />
          </svg>
        )}
      </button>
      {/* Timer display removed but functionality kept intact */}
    </div>
  );
};
