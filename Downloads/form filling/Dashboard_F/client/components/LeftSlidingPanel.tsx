
import React from 'react';
import { XMarkIcon } from './icons';

interface LeftSlidingPanelProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
}

const LeftSlidingPanel: React.FC<LeftSlidingPanelProps> = ({ isOpen, onClose, title, children }) => {
  return (
    <>
      {/* Backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 backdrop-blur-sm z-30 transition-opacity duration-300 ease-in-out"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      {/* Panel */}
      <div
        className={`
          fixed top-0 left-0 h-full w-full max-w-md /* Slide in from left, full height */
          bg-card text-text-primary shadow-2xl rounded-r-lg /* Only right corners rounded */
          flex flex-col 
          transform transition-transform duration-300 ease-in-out z-40 border border-border-color
          ${isOpen ? 'translate-x-0 opacity-100' : '-translate-x-full opacity-0 pointer-events-none'}
        `}
        role="dialog"
        aria-modal="true"
        aria-labelledby={`panel-title-${title.replace(/\s+/g, '-')}`}
      >
        {/* Panel Header */}
        <div className="flex justify-between items-center p-4 sm:p-5 border-b border-border-color bg-card-header rounded-t-lg"> {/* Added rounded-t-lg */}
          <h2 id={`panel-title-${title.replace(/\s+/g, '-')}`} className="text-lg sm:text-xl font-semibold text-text-primary">
            {title}
          </h2>
          <button
            onClick={onClose}
            className="text-text-secondary hover:text-accent transition-colors rounded-full p-1 -mr-1"
            aria-label="Close panel"
          >
            <XMarkIcon className="w-5 h-5 sm:w-6 sm:h-6" />
          </button>
        </div>

        {/* Panel Content */}
        <div className="flex-grow overflow-y-auto bg-card p-4 sm:p-6 rounded-b-lg"> {/* Added rounded-b-lg */}
          {children}
        </div>
      </div>
    </>
  );
};

export default LeftSlidingPanel;
