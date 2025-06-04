import React from 'react';
import { ActionItem, ActiveTaskFilterType } from '../types';
import SuggestedActionItem from './SuggestedActionItem';

interface SuggestedActionsSectionProps {
  actions: ActionItem[]; 
  onOpenActionDetails: (action: ActionItem) => void; // This prop will now be handled by App.tsx to differentiate tasks
  activeFilter: ActiveTaskFilterType;
  T: any; // Translation object
  hasMore?: boolean;
  onLoadMore?: () => void;
  isLoadingMore?: boolean;
}

const SuggestedActionsSection: React.FC<SuggestedActionsSectionProps> = ({ 
  actions, 
  onOpenActionDetails, 
  activeFilter, 
  T, 
  hasMore = false, 
  onLoadMore, 
  isLoadingMore = false 
}) => {
  
  const getSectionTitle = () => {
    if (activeFilter === null) return T.suggestedActions.currentTasksTitle; 
    if (activeFilter === 'nextSteps') return T.suggestedActions.nextStepsTitle; 
    if (activeFilter === 'completed') return T.suggestedActions.completedTasksTitle;
    return T.suggestedActions.suggestedActionsTitle; // Fallback
  };

  const sectionTitle = getSectionTitle();
  const emptyDashboardMessages = T.emptyDashboard || {};

  if (actions.length === 0) {
    let emptyMessage = emptyDashboardMessages.generic || "No actions to display for this category.";
    if (activeFilter === null) { 
      emptyMessage = emptyDashboardMessages.noInProgressTasks || "No tasks are currently in progress or pending. Well done!";
    } else if (activeFilter === 'nextSteps') { 
      emptyMessage = emptyDashboardMessages.noNotStartedTasks || "You're all caught up with your next steps!";
    } else if (activeFilter === 'completed') { 
      emptyMessage = emptyDashboardMessages.noCompletedTasks || "No tasks have been completed yet.";
    }
    
    return (
      <div>
        <h2 className="text-2xl font-semibold text-text-primary mb-6">{sectionTitle}</h2>
        <p className="text-text-secondary">{emptyMessage}</p>
      </div>
    );
  }
  
  return (
    <div>
      <h2 className="text-2xl font-semibold text-text-primary mb-6">{sectionTitle}</h2>
      
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3 justify-center">
        {actions.map(action => (
          <SuggestedActionItem
            key={action.id}
            action={action}
            onClick={onOpenActionDetails}
            T={T}
          />
        ))}
      </div>

      {/* Load More Button */}
      {hasMore && onLoadMore && (
        <div className="mt-6 text-center">
          <button
            onClick={onLoadMore}
            disabled={isLoadingMore}
            className="px-6 py-2 bg-accent text-white rounded-md hover:bg-accent-dark disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isLoadingMore ? 'Loading...' : 'Load More'}
          </button>
        </div>
      )}
    </div>
  );
};

export default SuggestedActionsSection;
