
import React from 'react';
import { ActionItem, ActionItemLink, ProgressStatus } from '../types';
import { XMarkIcon, DocumentTextIcon, CheckCircleIcon, ArrowPathIcon } from './icons'; 
import Modal from './Modal'; 

interface ActionDetailModalProps {
  isOpen: boolean;
  onClose: () => void;
  action: ActionItem;
  T: any; // Translation object
}

const DetailSection: React.FC<{ title: string; children: React.ReactNode; className?: string }> = ({ title, children, className }) => (
  <div className={`mb-4 ${className}`}>
    <h4 className="text-sm font-semibold text-text-secondary uppercase tracking-wider mb-1">{title}</h4>
    <div className="text-sm text-text-primary bg-background p-3 rounded-md border border-border-color">{children}</div>
  </div>
);

const getStatusColorClasses = (status: ProgressStatus): { text: string; border: string; bg: string; icon: React.FC<{className?: string}> } => {
  switch (status) {
    case ProgressStatus.NotStarted:
      return { text: 'text-status-red', border: 'border-status-red', bg: 'bg-status-red/10', icon: XMarkIcon };
    case ProgressStatus.InProgress:
      return { text: 'text-status-yellow', border: 'border-status-yellow', bg: 'bg-status-yellow/10', icon: ArrowPathIcon };
    case ProgressStatus.Completed:
      return { text: 'text-status-green', border: 'border-status-green', bg: 'bg-status-green/10', icon: CheckCircleIcon };
    default:
      return { text: 'text-text-secondary', border: 'border-border-color', bg: 'bg-gray-500/10', icon: ArrowPathIcon };
  }
};

const ActionDetailModal: React.FC<ActionDetailModalProps> = ({ isOpen, onClose, action, T }) => {
  if (!isOpen || !action) return null;

  const detailsTitles = T.actionDetailModal || {
    title: "Task Details",
    what: "What this step involves:",
    why: "Why it's important:",
    preparedByMaya: "Prepared by Maya:",
    stillMissing: "What's still needed:",
    progress: "Current Progress:",
    helpfulLinks: "Helpful Links:",
    relevantDocuments: "Relevant Documents:",
    deadlines: "Deadlines:",
  };
  const statusText = T.progressStatus[action.status] || action.status;
  const statusClasses = getStatusColorClasses(action.status);
  const StatusIcon = statusClasses.icon;

  return (
    <Modal isOpen={isOpen} onClose={onClose} title={detailsTitles.title}>
      <div className="space-y-3">
        <div className={`p-4 rounded-lg border-l-4 ${statusClasses.border} ${statusClasses.bg} flex items-center space-x-3 mb-4`}>
          <StatusIcon className={`w-7 h-7 ${statusClasses.text}`} />
          <div>
            <h3 className="text-lg font-semibold text-text-primary">{action.title}</h3>
            <p className={`text-sm font-medium ${statusClasses.text}`}>{statusText}</p>
          </div>
        </div>
        
        {action.status === ProgressStatus.InProgress && action.progressValue !== undefined && (
           <DetailSection title={detailsTitles.progress}>
            <div className="w-full bg-border-color rounded-full h-2.5">
                <div 
                    className="bg-status-yellow h-2.5 rounded-full" 
                    style={{ width: `${action.progressValue}%`}}
                    aria-label={`${action.progressValue}% complete`}
                ></div>
            </div>
            <p className="text-xs text-text-secondary mt-1">{action.progressValue}% complete</p>
          </DetailSection>
        )}

        <DetailSection title={detailsTitles.what}><p>{action.details.what}</p></DetailSection>
        <DetailSection title={detailsTitles.why}><p>{action.details.why}</p></DetailSection>
        <DetailSection title={detailsTitles.preparedByMaya}><p>{action.details.preparedByMaya}</p></DetailSection>
        <DetailSection title={detailsTitles.stillMissing}><p>{action.details.stillMissing}</p></DetailSection>
        
        {action.details.links && action.details.links.length > 0 && (
          <DetailSection title={detailsTitles.helpfulLinks}>
            <ul className="list-disc list-inside space-y-1">
              {action.details.links.map((link: ActionItemLink, index: number) => (
                <li key={index}>
                  <a href={link.url} target="_blank" rel="noopener noreferrer" className="text-accent hover:underline">
                    {link.text}
                  </a>
                </li>
              ))}
            </ul>
          </DetailSection>
        )}

        {action.details.documents && action.details.documents.length > 0 && (
           <DetailSection title={detailsTitles.relevantDocuments}>
             <ul className="list-disc list-inside space-y-1">
              {action.details.documents.map((docName: string, index: number) => (
                <li key={index} className="flex items-center">
                  <DocumentTextIcon className="w-4 h-4 mr-2 text-text-secondary flex-shrink-0"/>
                  <span>{docName}</span>
                </li>
              ))}
            </ul>
          </DetailSection>
        )}

        {action.details.deadlines && (
          <DetailSection title={detailsTitles.deadlines}>
            <p className={`font-medium ${statusClasses.text}`}>{action.details.deadlines}</p>
          </DetailSection>
        )}
      </div>
    </Modal>
  );
};

export default ActionDetailModal;