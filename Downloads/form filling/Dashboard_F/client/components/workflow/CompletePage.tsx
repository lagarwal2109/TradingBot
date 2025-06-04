
import React from 'react';
import ActionButton from '../ActionButton';
import { CheckCircleIcon } from '../icons';

interface CompletePageProps {
  onFinish: () => void; // This will call exitWorkflow
  T: any;
}

const CompletePage: React.FC<CompletePageProps> = ({ onFinish, T }) => {
  const stepName = T.workflowSteps?.complete || "Complete";
  return (
    <div className="text-center">
      <CheckCircleIcon className="w-16 h-16 text-green-500 mx-auto mb-4" />
      <h2 className="text-2xl font-semibold text-text-primary mb-4">{stepName}</h2>
      <p className="text-text-secondary mb-6">
        {T.workflowMessages?.completionMessage || "You have successfully completed this process. Maya will keep you updated on any next steps."}
      </p>
      <ActionButton onClick={onFinish} variant="primary" size="lg">
        {T.workflowNav?.finish || "Finish & Return to Dashboard"}
      </ActionButton>
    </div>
  );
};

export default CompletePage;
