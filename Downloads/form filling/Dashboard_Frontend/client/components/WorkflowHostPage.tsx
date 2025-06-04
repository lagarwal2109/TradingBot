import React from 'react';
import { WorkflowType, WorkflowStepConfig, UserProfile } from '../types'; // Added UserProfile
import { WorkflowStepper } from './WorkflowStepper'; 
import ActionButton from './ActionButton';
import { ChevronLeftIcon, ChevronRightIcon, ArrowUturnLeftIcon } from './icons';

// Import placeholder step components
import WelcomeStepPage from './workflow/WelcomeStepPage';
import PersonalDataFormFillingPage from './workflow/PersonalDataFormFillingPage';
import PersonalDataJobMatchingPage from './workflow/PersonalDataJobMatchingPage';
import DocumentsPage from './workflow/DocumentsPage';
import TransmitFormPage from './workflow/TransmitFormPage';
import JobMatchingPage from './workflow/JobMatchingPage';
import TransmitApplicationPage from './workflow/TransmitApplicationPage';
import AmaPage from './workflow/AmaPage';
import CompletePage from './workflow/CompletePage';


interface WorkflowHostPageProps {
  workflowDetails: {
    type: WorkflowType;
    title: string;
    steps: WorkflowStepConfig[];
  };
  currentStepIndex: number;
  workflowData: Record<string, any>;
  userProfile: UserProfile; // Added userProfile
  onExitWorkflow: () => void;
  onGoToNextStep: () => void;
  onGoToPreviousStep: () => void;
  onJumpToStep: (index: number) => void;
  onUpdateWorkflowData: (data: Record<string, any>) => void;
  T: any;
}

const WorkflowHostPage: React.FC<WorkflowHostPageProps> = ({
  workflowDetails,
  currentStepIndex,
  workflowData,
  userProfile, // Added userProfile
  onExitWorkflow,
  onGoToNextStep,
  onGoToPreviousStep,
  onJumpToStep,
  onUpdateWorkflowData,
  T,
}) => {
  const currentStepConfig = workflowDetails.steps[currentStepIndex];

  const renderStepContent = () => {
    const stepProps = {
      workflowData,
      userProfile, // Pass userProfile to step components
      onUpdateData: onUpdateWorkflowData,
      onNextStep: onGoToNextStep,
      onPreviousStep: onGoToPreviousStep,
      T,
      onFinish: onExitWorkflow, 
    };

    switch (workflowDetails.type) {
      case 'formFilling':
        switch (currentStepConfig.id) {
          case 'welcome': return <WelcomeStepPage {...stepProps} />;
          case 'personalData': return <PersonalDataFormFillingPage {...stepProps} />;
          case 'documents': return <DocumentsPage {...stepProps} />;
          case 'transmitForm': return <TransmitFormPage {...stepProps} />;
          case 'ama': return <AmaPage {...stepProps} />;
          case 'complete': return <CompletePage {...stepProps} />;
          default: return <div>Unknown step for Form Filling</div>;
        }
      case 'jobMatching':
        switch (currentStepConfig.id) {
          case 'welcome': return <WelcomeStepPage {...stepProps} />;
          case 'personalData': return <PersonalDataJobMatchingPage {...stepProps} />;
          case 'jobMatching': return <JobMatchingPage {...stepProps} />;
          case 'transmitApplication': return <TransmitApplicationPage {...stepProps} />;
          case 'ama': return <AmaPage {...stepProps} />;
          case 'complete': return <CompletePage {...stepProps} />;
          default: return <div>Unknown step for Job Matching</div>;
        }
      default:
        return <div>Unknown workflow type</div>;
    }
  };

  const isFirstStep = currentStepIndex === 0;
  const isLastStep = currentStepIndex === workflowDetails.steps.length - 1;

  return (
    <div className="flex flex-col min-h-screen bg-background text-text-primary">
      <header className="p-4 border-b border-border-color bg-card shadow-sm">
        <div className="container mx-auto flex justify-center items-center"> {/* Centered title */}
          <h1 className="text-xl sm:text-2xl font-semibold text-accent">
            {workflowDetails.title}
          </h1>
          {/* "Back to Dashboard" button removed from here */}
        </div>
      </header>

      <main className="flex-grow container mx-auto px-2 py-2 sm:px-4 sm:py-6 flex flex-col">
        <WorkflowStepper
          steps={workflowDetails.steps}
          currentStepIndex={currentStepIndex}
          onStepClick={onJumpToStep}
          T={T}
        />
        <div className="flex-grow bg-card p-4 sm:p-6 md:p-8 rounded-lg shadow-lg border border-border-color mt-4">
          {renderStepContent()}
        </div>
         {/* "Back to Dashboard" button moved here */}
        <div className="mt-6 text-center">
          <ActionButton variant="outline" size="md" onClick={onExitWorkflow} icon={<ArrowUturnLeftIcon className="w-4 h-4"/>}>
            {T.workflowNav.backToDashboard}
          </ActionButton>
        </div>
      </main>

      <footer className="p-4 border-t border-border-color bg-card sticky bottom-0"> {/* Made footer sticky */}
        <div className="container mx-auto flex justify-between items-center">
          <ActionButton
            onClick={onGoToPreviousStep}
            disabled={isFirstStep}
            variant="secondary"
            icon={<ChevronLeftIcon className="w-5 h-5"/>}
          >
            {T.workflowNav.previous}
          </ActionButton>
          <ActionButton
            onClick={onGoToNextStep}
            variant="primary"
            icon={!isLastStep ? <ChevronRightIcon className="w-5 h-5"/> : undefined}
          >
            {isLastStep ? T.workflowNav.finish : T.workflowNav.next}
          </ActionButton>
        </div>
      </footer>
    </div>
  );
};

export default WorkflowHostPage;