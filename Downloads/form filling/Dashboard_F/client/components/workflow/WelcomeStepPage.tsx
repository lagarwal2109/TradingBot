
import React from 'react';
import ActionButton from '../ActionButton';
import { ChevronRightIcon } from '../icons';

interface WelcomeStepPageProps {
  onNextStep: () => void;
  T: any;
}

const WelcomeStepPage: React.FC<WelcomeStepPageProps> = ({ onNextStep, T }): JSX.Element => {
  const stepName = T.workflowSteps?.welcome || "Welcome";
  return (
    <div className="text-center">
      <h2 className="text-2xl font-semibold text-text-primary mb-4">{stepName}</h2>
      <p className="text-text-secondary mb-6">
        {T.workflowMessages?.welcomeIntro || "Welcome to this guided process. Let's get started!"}
      </p>
      <ActionButton onClick={onNextStep} icon={<ChevronRightIcon />} variant="primary" size="lg">
        {T.workflowNav?.next || "Start"}
      </ActionButton>
    </div>
  );
};

export default WelcomeStepPage;
