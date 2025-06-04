
import React, { useState, useRef, useEffect } from 'react';
import ActionButton from './ActionButton';
import { ArrowUpTrayIcon, ChevronDownIcon, ChevronUpIcon } from './icons';
import { UserProfile, Language } from '../types';
import { AVAILABLE_LANGUAGES_RAW } from '../constants';

interface SidebarCombinedMenuProps {
  userProfile: UserProfile;
  currentLanguage: Language;
  onLanguageChange: (language: Language) => void;
  T: any; // Translation object
}

const SidebarCombinedMenu: React.FC<SidebarCombinedMenuProps> = ({ 
  userProfile, 
  currentLanguage, 
  onLanguageChange, 
  T 
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const toggleDropdown = () => setIsOpen(!isOpen);

  const menuLabels = T.sidebarCombinedMenu || {
    triggerAriaLabel: "Open settings",
    exportSectionTitle: "Export Profile",
    languageSectionTitle: "Select Language"
  };
  const exportOptionLabels = T.exportProfileOptions || { pdf: "PDF", json: "JSON", text: "TXT" };
  const pdfAlert = T.pdfExportAlert || "PDF export simulation: Downloading as TXT.";

  const handleExport = (format: 'PDF' | 'JSON' | 'TXT') => {
    let content = '';
    let filename = `${userProfile.name || 'user'}_profile.${format.toLowerCase()}`;
    let mimeType = '';

    switch (format) {
      case 'JSON':
        content = JSON.stringify(userProfile, null, 2);
        mimeType = 'application/json';
        break;
      case 'TXT':
        content = Object.entries(userProfile)
          .map(([key, value]) => `${key}: ${typeof value === 'object' ? JSON.stringify(value) : value}`)
          .join('\n');
        mimeType = 'text/plain';
        break;
      case 'PDF':
        alert(pdfAlert);
        content = `User Profile: ${userProfile.name}\n\n` + Object.entries(userProfile)
          .map(([key, value]) => `${key}: ${typeof value === 'object' ? JSON.stringify(value) : value}`)
          .join('\n');
        mimeType = 'text/plain';
        filename = `${userProfile.name || 'user'}_profile_summary.txt`;
        break;
    }

    if (content) {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    setIsOpen(false);
  };

  const handleSelectLanguage = (langCode: Language) => {
    onLanguageChange(langCode);
    setIsOpen(false);
  };

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const selectedLanguageName = T.languageNames[currentLanguage] || currentLanguage;

  return (
    <div className="relative text-left w-full" ref={dropdownRef}>
      <ActionButton
        onClick={toggleDropdown}
        variant="sidebar"
        className="w-full flex justify-between items-center"
        icon={<ArrowUpTrayIcon className="w-5 h-5" />}
        aria-label={menuLabels.triggerAriaLabel}
        aria-expanded={isOpen}
        aria-controls="sidebar-combined-menu-options"
      >
        <span>{selectedLanguageName}</span>
        {isOpen ? <ChevronUpIcon className="w-4 h-4 ml-1" /> : <ChevronDownIcon className="w-4 h-4 ml-1" />}
      </ActionButton>

      {isOpen && (
        <div
          id="sidebar-combined-menu-options"
          className="origin-top-right absolute right-0 bottom-full mb-2 w-full rounded-md shadow-lg bg-card ring-1 ring-border-color ring-opacity-50 z-20 max-h-60 overflow-y-auto" // Positioned above, added max-height and overflow
          role="menu"
          aria-orientation="vertical"
        >
          <div className="py-1" role="none">
            {/* Export Options Section */}
            <div className="px-4 pt-2 pb-1 text-xs font-semibold text-text-secondary uppercase tracking-wider">
              {menuLabels.exportSectionTitle}
            </div>
            <button
              onClick={() => handleExport('PDF')}
              className="block w-full text-left px-4 py-2 text-sm text-text-secondary hover:bg-border-color/50 hover:text-text-primary"
              role="menuitem"
            >
              <span>{exportOptionLabels.pdf}</span>
            </button>
            <button
              onClick={() => handleExport('JSON')}
              className="block w-full text-left px-4 py-2 text-sm text-text-secondary hover:bg-border-color/50 hover:text-text-primary"
              role="menuitem"
            >
              <span>{exportOptionLabels.json}</span>
            </button>
            <button
              onClick={() => handleExport('TXT')}
              className="block w-full text-left px-4 py-2 text-sm text-text-secondary hover:bg-border-color/50 hover:text-text-primary"
              role="menuitem"
            >
              <span>{exportOptionLabels.text}</span>
            </button>

            <hr className="my-1 border-border-color/40 mx-2" />

            {/* Language Options Section */}
            <div className="px-4 pt-2 pb-1 text-xs font-semibold text-text-secondary uppercase tracking-wider">
              {menuLabels.languageSectionTitle}
            </div>
            {AVAILABLE_LANGUAGES_RAW.map((lang) => (
              <button
                key={lang.code}
                onClick={() => handleSelectLanguage(lang.code)}
                className={`block w-full text-left px-4 py-2 text-sm ${
                  lang.code === currentLanguage ? 'bg-accent/10 text-accent font-semibold' : 'text-text-secondary'
                } hover:bg-border-color/50 hover:text-text-primary`}
                role="menuitem"
              >
                <span>{T.languageNames[lang.code] || lang.name}</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default SidebarCombinedMenu;
