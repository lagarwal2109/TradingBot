
import React from 'react';
import { UserProfile } from '../types';
import { UserCircleIcon } from './icons';

interface BasicInfoProps {
  user: UserProfile;
  onboardingCompletion: number; // Percentage 0-100
  T: any; // Translation object
}

const BasicInfo: React.FC<BasicInfoProps> = ({ user, onboardingCompletion, T }) => {
  const onboardingLabel = T.basicInfo.onboardingCompletion || `Onboarding {percentage}% Complete`;
  const formattedOnboardingLabel = onboardingLabel.replace('{percentage}', Math.round(onboardingCompletion).toString());

  return (
    <div className="bg-card p-4 sm:p-6 rounded-lg shadow-lg border border-border-color">
      <div className="flex flex-col sm:flex-row sm:items-center sm:space-x-6">
        <div className="flex items-center space-x-4 mb-4 sm:mb-0">
          <UserCircleIcon className="w-16 h-16 text-accent flex-shrink-0" />
          <div>
            <h2 className="text-2xl font-semibold text-text-primary">{T.basicInfo.welcomeMessage || "Welcome!"}</h2>
            <p className="text-sm text-text-secondary">

            </p>
          </div>
        </div>

        <div className="flex-grow mt-4 sm:mt-0">
          <div className="mb-1">
            <span className="text-sm font-medium text-text-primary">
              {formattedOnboardingLabel}
            </span>
          </div>
          <div className="w-full bg-border-color rounded-full h-2.5">
            <div
              className="bg-secondary-accent h-2.5 rounded-full transition-all duration-500 ease-out"
              style={{ width: `${onboardingCompletion}%` }}
              role="progressbar"
              aria-valuenow={onboardingCompletion}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-label={formattedOnboardingLabel}
            ></div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default BasicInfo;