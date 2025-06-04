import React from 'react';
import { UserProfile } from '../../types';

interface PersonalDataJobMatchingPageProps {
  workflowData: Record<string, any>;
  userProfile: UserProfile;
  onUpdateData: (data: Record<string, any>) => void;
  onNextStep: () => void;
  onPreviousStep: () => void;
  T: any;
}

// Updated SummaryItemCard to be a self-contained card
const SummaryItemCard: React.FC<{ label: string; value: string; valueColor?: string }> = ({ label, value, valueColor = "text-text-primary" }) => (
  <div className="bg-background p-3 rounded-lg mb-2 shadow-sm border border-border-color">
    <h4 className="text-xs text-text-secondary">{label}</h4>
    <p className={`text-sm font-medium ${valueColor}`}>{value}</p>
  </div>
);


const PersonalDataJobMatchingPage: React.FC<PersonalDataJobMatchingPageProps> = ({ userProfile, T }) => {
  const labels = T.workflowPageLabels.summaryLabels || {};
  const pageLabels = T.workflowPageLabels || {};
  const positionsFoundText = (labels.positionsFound || "{count} positions found").replace("{count}", "3");

  const summaryData = [
    { label: labels.skills || "Skills", value: "Software Development, IT Support" },
    { label: labels.experience || "Experience", value: labels.notCollected || "not collected yet", valueColor: "text-text-secondary italic" },
    { label: labels.education || "Education", value: labels.notCollected || "not collected yet", valueColor: "text-text-secondary italic" },
    { label: labels.languages || "Languages", value: "Ukrainian, English" },
    { label: labels.jobMatches || "Job Matches", value: positionsFoundText, valueColor: "text-status-green font-semibold" },
    { label: labels.application || "Application", value: labels.notStarted || "not started", valueColor: "text-text-secondary italic" },
  ];


  return (
    <div className="grid md:grid-cols-2 gap-8 md:items-start"> {/* Changed items-center to md:items-start */}
      {/* Left Column: Maya Avatar */}
      <div className="flex justify-center items-center h-full md:order-1 pt-4 md:pt-0">  {/* Added padding top for mobile, removed for md */}
        <div 
          className="w-56 h-56 sm:w-64 sm:h-64 bg-gradient-to-br from-purple-300 to-indigo-400 rounded-full flex items-center justify-center shadow-xl"
          aria-label={pageLabels.mayaAvatarText || "Maya Avatar"}
        >
          <span className="text-white text-lg font-medium opacity-80">{pageLabels.mayaAvatarText || "Maya Avatar"}</span>
        </div>
      </div>

      {/* Right Column: Summary Panel */}
      <div className="bg-white p-6 rounded-lg shadow-lg border border-border-color md:order-2">
        <div className="flex items-center mb-4">
           <div className="w-10 h-10 bg-accent text-white flex items-center justify-center rounded-full text-lg font-semibold mr-3">
             {userProfile.name.substring(0, 1).toUpperCase()}{(userProfile.name.split(' ')[1]?.substring(0,1) || '').toUpperCase()}
          </div>
          <div>
            <h3 className="text-lg font-semibold text-text-primary">{userProfile.name}</h3>
            <p className="text-sm text-text-secondary">{pageLabels.jobMatchingSummaryTitle || "Job Matching Profile"}</p>
          </div>
        </div>
        <div className="space-y-2"> {/* Container for SummaryItemCards */}
          {summaryData.map(item => (
            <SummaryItemCard key={item.label} label={item.label} value={item.value} valueColor={item.valueColor} />
          ))}
        </div>
      </div>
    </div>
  );
};

export default PersonalDataJobMatchingPage;