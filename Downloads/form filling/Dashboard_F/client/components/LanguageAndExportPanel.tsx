import React, { useState } from 'react';
import { Language, UserProfile } from '../types';
import { ChevronDownIcon, ArrowDownTrayIcon } from './icons';

interface LanguageAndExportPanelProps {
  currentLanguage: Language;
  onLanguageChange: (lang: Language) => void;
  userProfile: UserProfile | null;
  T: any;
}

const LanguageAndExportPanel: React.FC<LanguageAndExportPanelProps> = ({ 
  currentLanguage, 
  onLanguageChange, 
  userProfile, 
  T 
}) => {
  const [isLanguageDropdownOpen, setIsLanguageDropdownOpen] = useState(false);
  const [isExportDropdownOpen, setIsExportDropdownOpen] = useState(false);

  const languageLabels = {
    en: T.languageNames?.en || 'English',
    es: T.languageNames?.es || 'Español', 
    de: T.languageNames?.de || 'Deutsch'
  };

  const handleLanguageSelect = (lang: Language) => {
    onLanguageChange(lang);
    setIsLanguageDropdownOpen(false);
  };

  const handleExport = (format: 'json' | 'pdf' | 'txt') => {
    if (!userProfile) return;
    
    let dataStr = '';
    let downloadAnchorNode = document.createElement('a');
    
    // Define txtContent once for reuse
    const txtContent = `
Profile Export
==============

Name: ${userProfile.name}
Alias: ${userProfile.alias}
Email: ${userProfile.email}
Phone: ${userProfile.phone}
Country of Origin: ${userProfile.countryOfOrigin}
Date of Registration: ${userProfile.dateOfRegistration}
Age: ${userProfile.age}
Gender: ${userProfile.gender}
Date of Birth: ${userProfile.dateOfBirth}

Biography:
${userProfile.bio}

Onboarding Summary:
${userProfile.onboardingSummary}

Challenges:
${userProfile.challenges?.join(', ') || 'None listed'}
    `;
    
    switch (format) {
      case 'json':
        dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(userProfile, null, 2));
        downloadAnchorNode.setAttribute("href", dataStr);
        downloadAnchorNode.setAttribute("download", "profile.json");
        break;
      case 'txt':
        dataStr = "data:text/plain;charset=utf-8," + encodeURIComponent(txtContent);
        downloadAnchorNode.setAttribute("href", dataStr);
        downloadAnchorNode.setAttribute("download", "profile.txt");
        break;
      case 'pdf':
        // For PDF, we'll simulate by downloading as TXT for now
        alert(T.pdfExportAlert || "PDF export simulation: Downloading as TXT.");
        dataStr = "data:text/plain;charset=utf-8," + encodeURIComponent(txtContent);
        downloadAnchorNode.setAttribute("href", dataStr);
        downloadAnchorNode.setAttribute("download", "profile_as_txt.txt");
        break;
    }
    
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    document.body.removeChild(downloadAnchorNode);
    setIsExportDropdownOpen(false);
  };

  return (
    <div className="p-3 border-t border-border-color">
      {/* Single row layout matching the screenshot */}
      <div className="flex items-center justify-between bg-card p-2 rounded-md shadow-sm">
        {/* Left side - Export icon with dropdown */}
        <div className="relative">
          <button
            onClick={() => setIsExportDropdownOpen(!isExportDropdownOpen)}
            className="flex items-center space-x-1 hover:bg-card-header p-1 rounded transition-colors focus:outline-none focus:ring-2 focus:ring-accent"
          >
            <ArrowDownTrayIcon className="w-4 h-4 text-accent" />
          </button>
          
          {isExportDropdownOpen && (
            <div className="absolute bottom-full left-0 mb-1 bg-card border border-border-color rounded-md shadow-lg z-50 min-w-32">
              <div className="py-1">
                <button
                  onClick={() => handleExport('json')}
                  className="w-full text-left px-3 py-2 text-sm text-text-primary hover:bg-accent/10 transition-colors block"
                >
                  Export as JSON
                </button>
                <button
                  onClick={() => handleExport('pdf')}
                  className="w-full text-left px-3 py-2 text-sm text-text-primary hover:bg-accent/10 transition-colors block"
                >
                  Export as PDF
                </button>
                <button
                  onClick={() => handleExport('txt')}
                  className="w-full text-left px-3 py-2 text-sm text-text-primary hover:bg-accent/10 transition-colors block"
                >
                  Export as Text
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Right side - Language dropdown */}
        <div className="relative">
          <button
            onClick={() => setIsLanguageDropdownOpen(!isLanguageDropdownOpen)}
            className="flex items-center space-x-2 hover:bg-card-header p-1 rounded transition-colors focus:outline-none focus:ring-2 focus:ring-accent"
          >
            <span className="text-sm font-medium text-text-primary">{languageLabels[currentLanguage]}</span>
            <ChevronDownIcon className={`w-4 h-4 text-text-secondary transition-transform ${isLanguageDropdownOpen ? 'rotate-180' : ''}`} />
          </button>
          
          {isLanguageDropdownOpen && (
            <div className="absolute bottom-full right-0 mb-1 bg-card border border-border-color rounded-md shadow-lg z-50">
              <div className="py-1">
                {(['en', 'es', 'de'] as Language[]).map((lang) => (
                  <button
                    key={lang}
                    onClick={() => handleLanguageSelect(lang)}
                    className={`w-full text-left px-3 py-2 text-sm transition-colors whitespace-nowrap ${
                      currentLanguage === lang 
                        ? 'bg-accent text-white' 
                        : 'text-text-primary hover:bg-accent/10'
                    }`}
                  >
                    {languageLabels[lang]}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default LanguageAndExportPanel; 