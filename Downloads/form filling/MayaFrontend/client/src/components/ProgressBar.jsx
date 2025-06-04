import React from 'react';

export const ProgressBar = ({ currentStep = 1, onStepClick }) => {
  // Define the steps
  const steps = [
    { id: 1, name: 'Welcome' },
    { id: 2, name: 'Fill Form' },
    { id: 3, name: 'Recommendations' },
    { id: 4, name: 'End Session' }
  ];

  return (
    <div className="w-full max-w-3xl mx-auto py-3 px-2">
      <div className="flex items-center justify-between">
        {steps.map((step, index) => (
          <React.Fragment key={step.id}>
            {/* Step Circle */}
            <div className="flex flex-col items-center">
              <button 
                onClick={() => {
                  // Only allow clicks if:
                  // 1. We're on step 4 and clicking step 1 (restart)
                  // 2. We're not on step 4 (normal navigation)
                  if ((currentStep === 4 && step.id === 1) || currentStep !== 4) {
                    onStepClick && onStepClick(step.id);
                  }
                }}
                disabled={currentStep === 4 && step.id !== 1 && step.id !== 4}
                className={`w-8 h-8 rounded-full flex items-center justify-center transition-colors
                  ${currentStep > step.id 
                    ? 'bg-purple-300 text-white' 
                    : currentStep === step.id 
                      ? 'bg-purple-400 text-white' 
                      : currentStep === 4 && (step.id === 2 || step.id === 3)
                        ? 'bg-gray-300 text-gray-400 border border-gray-300 cursor-not-allowed'
                        : 'bg-white/70 text-gray-500 border border-gray-200 hover:bg-white/90'}`}
              >
                {currentStep > step.id ? (
                  <svg 
                    xmlns="http://www.w3.org/2000/svg" 
                    className="h-4 w-4" 
                    fill="none" 
                    viewBox="0 0 24 24" 
                    stroke="currentColor"
                  >
                    <path 
                      strokeLinecap="round" 
                      strokeLinejoin="round" 
                      strokeWidth={2} 
                      d="M5 13l4 4L19 7" 
                    />
                  </svg>
                ) : (
                  <span className="text-xs font-medium">{step.id}</span>
                )}
              </button>
              {/* Step Label */}
              <div className="text-[10px] mt-1 font-medium text-center text-gray-500">
                {step.name}
              </div>
            </div>
            
            {/* Connector Line (if not the last step) */}
            {index < steps.length - 1 && (
              <div 
                className={`flex-1 h-0.5 mx-2 
                  ${currentStep > index + 1 
                    ? 'bg-purple-300' 
                    : 'bg-gray-200'}`}
              ></div>
            )}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
};
