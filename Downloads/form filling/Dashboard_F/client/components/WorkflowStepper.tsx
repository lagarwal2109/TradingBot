import React from 'react';
import { WorkflowStepConfig } from '../types';
import { CheckCircleIcon } from './icons'; // For completed steps

interface WorkflowStepperProps {
  steps: WorkflowStepConfig[];
  currentStepIndex: number;
  onStepClick: (index: number) => void;
  T: any;
}

export const WorkflowStepper: React.FC<WorkflowStepperProps> = ({ steps, currentStepIndex, onStepClick, T }): JSX.Element => {
  return (
    <nav aria-label="Workflow Progress" className="py-4 px-2 sm:px-0">
      <ol className="flex flex-wrap items-center justify-center space-y-2 sm:space-y-0 sm:space-x-2 md:space-x-4">
        {steps.map((step, index) => {
          const isCompleted = index < currentStepIndex;
          const isCurrent = index === currentStepIndex;
          const stepName = T.workflowSteps[step.id] || step.nameKey.substring(step.nameKey.lastIndexOf('.') + 1);

          return (
            <li key={step.id} className="flex-shrink-0 sm:flex-1 w-full sm:w-auto">
              <button
                onClick={() => onStepClick(index)}
                disabled={index > currentStepIndex + 1 && !isCompleted} // Allow navigation to current, previous, and next immediate step for demo
                className={`group flex flex-col items-center w-full text-center p-1 transition-colors duration-150 rounded-md ${
                  index > currentStepIndex + 1 && !isCompleted 
                    ? 'cursor-not-allowed' 
                    : 'hover:bg-accent/10 focus:outline-none focus:ring-2 focus:ring-accent/50 focus:ring-offset-2 focus:ring-offset-background'
                }`}
                aria-current={isCurrent ? 'step' : undefined}
              >
                <div className="flex items-center w-full">
                  <span
                    className={`flex items-center justify-center w-8 h-8 md:w-10 md:h-10 rounded-full border-2 font-semibold transition-all duration-200 ${
                      isCurrent 
                        ? 'bg-accent text-white border-accent scale-110' 
                        : isCompleted 
                        ? 'bg-green-500 text-white border-green-500' 
                        : 'bg-card text-text-secondary border-border-color group-hover:border-accent/70'
                    }`}
                  >
                    {isCompleted && !isCurrent ? <CheckCircleIcon className="w-5 h-5 md:w-6 md:h-6" /> : index + 1}
                  </span>
                  <span 
                    className={`hidden sm:block h-0.5 flex-grow mx-2 rounded transition-colors duration-200 ${
                      isCompleted || isCurrent ? 'bg-accent' : 'bg-border-color'
                    } ${
                      index === steps.length - 1 ? 'sm:hidden' : ''
                    }`}
                  ></span>
                  <span 
                    className={`sm:hidden h-4 w-0.5 mx-auto rounded transition-colors duration-200 ${
                      index === steps.length - 1 ? 'hidden' : 'block'
                    } ${
                      isCompleted || isCurrent ? 'bg-accent' : 'bg-border-color'
                    }`}
                  ></span>
                </div>
                <span 
                  className={`mt-1.5 text-xs md:text-sm font-medium transition-colors duration-200 ${
                    isCurrent 
                      ? 'text-accent font-bold' 
                      : isCompleted 
                      ? 'text-green-600' 
                      : 'text-text-secondary group-hover:text-accent'
                  }`}
                >
                  {stepName}
                </span>
              </button>
            </li>
          );
        })}
      </ol>
    </nav>
  );
};
