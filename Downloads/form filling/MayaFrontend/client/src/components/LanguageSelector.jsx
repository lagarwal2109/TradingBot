import { useState, useEffect, useRef } from "react";

// Language options with their display names - Auto first, then the rest
const LANGUAGES = [
  { name: "Auto", value: "auto" },
  { name: "English", value: "english" },
  { name: "Hindi", value: "hindi" },
  { name: "German", value: "german" },
  { name: "Arabic", value: "arabic" },
  { name: "Spanish", value: "spanish" },
  { name: "French", value: "french" },
  { name: "Chinese", value: "chinese" },
  { name: "Japanese", value: "japanese" },
  { name: "Korean", value: "korean" },
  { name: "Russian", value: "russian" },
  { name: "Portuguese", value: "portuguese" },
  { name: "Italian", value: "italian" }
];
const API_ENDPOINT = "https://test.mayacode.io/api/api";
export const LanguageSelector = ({ currentLanguage, setCurrentLanguage, buttonClassName }) => {
  const [showLanguages, setShowLanguages] = useState(false);
  const dropdownRef = useRef(null);

  // Handle clicks outside of the dropdown
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setShowLanguages(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  const changeLanguage = async (language) => {
    // Set language immediately in UI for responsive feedback
    setCurrentLanguage(language);
    setShowLanguages(false);
    // Mark that user has made a selection
    setHasSelected(true);
    
    try {
      const response = await fetch(`${API_ENDPOINT}/set-language`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ language }),
      });
      
      if (!response.ok) {
        console.error('Failed to set language on server');
      }
    } catch (error) {
      console.error('Error setting language:', error);
    }
  };

  // Initialize with a flag to track if user has made a selection
  const [hasSelected, setHasSelected] = useState(false);

  // If user hasn't made a selection yet, show 'Select Language'
  // Otherwise show the selected language name (even if it's 'Auto')
  const buttonText = !hasSelected ? 
    "Select Language" : 
    LANGUAGES.find(lang => lang.value === currentLanguage)?.name;

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        onClick={() => setShowLanguages(!showLanguages)}
        className={buttonClassName || "bg-purple-200 hover:bg-purple-300 text-gray-800 p-2 rounded-md flex items-center justify-between text-xs font-medium"}
      >
        <span>{buttonText}</span>
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4 ml-1">
          <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" />
        </svg>
      </button>
      
      {/* Dropdown Menu with Scrollbar */}
      {showLanguages && (
        <div className="absolute top-full mt-1 w-full bg-white rounded-md shadow-lg z-20 border border-purple-100">
          <div className="max-h-40 overflow-y-auto custom-scrollbar">
            {LANGUAGES.map((language) => (
              <button
                key={language.value}
                onClick={() => changeLanguage(language.value)}
                className={`w-full text-left px-3 py-1.5 hover:bg-purple-50 transition-colors text-xs ${
                  currentLanguage === language.value ? 'bg-purple-50 font-medium text-purple-700' : 'text-gray-700'
                }`}
              >
                {language.name}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};