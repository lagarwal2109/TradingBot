import React from 'react';
import { ActionItem, ProgressStatus } from '../types';
import { getTaskIcon } from './icons'; 

interface SuggestedActionItemProps {
  action: ActionItem;
  onClick: (action: ActionItem) => void; 
  T: any; // Translation object
}

const getStatusClasses = (status: ProgressStatus): { border: string; text: string; iconColor: string; badgeBg: string; badgeText: string } => {
  switch (status) {
    case ProgressStatus.NotStarted:
      return { border: 'border-status-red', text: 'text-status-red', iconColor: 'text-status-red', badgeBg: 'bg-status-red-badge', badgeText: 'text-white' };
    case ProgressStatus.InProgress:
      return { border: 'border-status-yellow', text: 'text-status-yellow', iconColor: 'text-status-yellow', badgeBg: 'bg-status-yellow-badge', badgeText: 'text-text-primary' };
    case ProgressStatus.Completed:
      return { border: 'border-status-green', text: 'text-status-green', iconColor: 'text-status-green', badgeBg: 'bg-status-green-badge', badgeText: 'text-white' };
    default:
      return { border: 'border-border-color', text: 'text-text-secondary', iconColor: 'text-text-secondary', badgeBg: 'bg-gray-400', badgeText: 'text-white' };
  }
};

const SuggestedActionItem: React.FC<SuggestedActionItemProps> = ({ action, onClick, T }) => {
  const statusClasses = getStatusClasses(action.status);
  const statusText = T.progressStatus[action.status] || action.status;
  
  const TaskIcon = getTaskIcon(action.icon);

  // Use translated title if available in T.actionTitles, otherwise use action.title directly
  const actionTitleKey = action.title; // This is the key from constants.ts or a direct title
  const translatedTitle = (T.actionTitles && T.actionTitles[actionTitleKey]) ? T.actionTitles[actionTitleKey] : action.title;

  return (
    <button
      onClick={() => onClick(action)}
      className={`w-full max-w-[330px] h-[280px] flex flex-col justify-between p-4 bg-card shadow-sm rounded-lg border-2 ${statusClasses.border} hover:shadow-md hover:border-accent hover:scale-105 transition-all duration-150 focus:outline-none focus:ring-2 focus:ring-accent focus:ring-offset-2 focus:ring-offset-background`}
      aria-label={`${translatedTitle}, ${statusText}`}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') onClick(action);}}
    >
      <div className="flex flex-col items-center justify-center flex-grow px-4 py-6 text-center">
        <TaskIcon className={`w-10 h-10 mb-3 ${statusClasses.iconColor}`} />
        <h3 className="text-base font-semibold text-text-primary leading-tight">
          {translatedTitle}
        </h3>
      </div>
      <div className="w-full">
        {action.status === ProgressStatus.InProgress && action.progressValue !== undefined && (
          <div className="mb-2">
            <div className="w-full bg-border-color rounded-full h-1.5">
              <div
                className={`h-1.5 rounded-full ${statusClasses.border.replace('border-', 'bg-')}`}
                style={{ width: `${action.progressValue}%` }}
                aria-hidden="true"
              ></div>
            </div>
            <div className="text-xs text-text-secondary text-center mt-1">
              {action.progressValue}% complete
            </div>
          </div>
        )}
        <span className={`text-xs font-medium px-2 py-1 rounded-full ${statusClasses.badgeBg} ${statusClasses.badgeText} block text-center`}>
          {statusText}
        </span>
      </div>
    </button>
  );
};

export default SuggestedActionItem;

// Helper for CSS if needed elsewhere (e.g. global CSS or styled-components)
// Ensure this class is available if using it, or remove Dtruncate-2-lines from h3 if not defined globally
// .truncate-2-lines {
//   overflow: hidden;
//   text-overflow: ellipsis;
//   display: -webkit-box;
//   -webkit-line-clamp: 2;
//   -webkit-box-orient: vertical;
// }
