import React from 'react';
import { UserProfile } from '../../types';

interface PersonalDataFormFillingPageProps {
  workflowData: Record<string, any>;
  userProfile: UserProfile;
  onUpdateData: (data: Record<string, any>) => void;
  onNextStep: () => void;
  onPreviousStep: () => void;
  T: any;
}

// Updated SummaryItem to be a self-contained card
const SummaryItemCard: React.FC<{ label: string; value: string; valueColor?: string }> = ({ label, value, valueColor = "text-text-primary" }) => (
  <div className="bg-background p-3 rounded-lg mb-2 shadow-sm border border-border-color">
    <h4 className="text-xs text-text-secondary">{label}</h4>
    <p className={`text-sm font-medium ${valueColor}`}>{value}</p>
  </div>
);

const PersonalDataFormFillingPage: React.FC<PersonalDataFormFillingPageProps> = ({ userProfile, T }) => {
  const labels = T.workflowPageLabels.summaryLabels || {};
  const pageLabels = T.workflowPageLabels || {};

  const summaryData = [
    { label: labels.name || "Name", value: userProfile.name || "Alex Dmitri Doe" },
    { label: labels.dateOfBirth || "Date of Birth", value: labels.notCollected || "not collected yet", valueColor: "text-text-secondary italic" },
    { label: labels.nationality || "Nationality", value: "Ukraine" }, // Example static value
    { label: labels.placeOfBirth || "Place of Birth", value: labels.notCollected || "not collected yet", valueColor: "text-text-secondary italic" },
    { label: labels.maritalStatus || "Marital Status", value: labels.notCollected || "not collected yet", valueColor: "text-text-secondary italic" },
    { label: labels.vulnerabilities || "Vulnerabilities", value: labels.assessmentPending || "assessment pending", valueColor: "text-text-secondary italic" },
  ];

  return (
    <div className="grid md:grid-cols-2 gap-8 md:items-start"> {/* Changed items-center to md:items-start */}
      {/* Left Column: Maya Avatar */}
      <div className="flex justify-center items-center h-full md:order-1 pt-4 md:pt-0"> {/* Added padding top for mobile, removed for md */}
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
            <p className="text-sm text-text-secondary">{pageLabels.formFillingSummaryTitle || "Asylum Form Collection"}</p>
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

export default PersonalDataFormFillingPage;