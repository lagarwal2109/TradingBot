
import React from 'react';

interface JobMatchingPageProps {
  onNextStep: () => void;
  onPreviousStep: () => void;
  T: any;
}

const JobMatchingPage: React.FC<JobMatchingPageProps> = ({ T }) => {
  const stepName = T.workflowSteps?.jobMatching || "Job Matching";
  return (
    <div>
      <h2 className="text-xl font-semibold text-text-primary mb-4">{stepName}</h2>
      <p className="text-text-secondary">
        Placeholder for displaying job matches, filtering options, and selecting jobs.
      </p>
      {/* TODO: Add job matching UI */}
    </div>
  );
};

export default JobMatchingPage;
