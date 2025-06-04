
export enum ProgressStatus {
  NotStarted = 'Not Started',
  InProgress = 'In Progress',
  Completed = 'Completed',
}

export interface UserProfile {
  name: string;
  alias?: string;
  countryOfOrigin: string;
  dateOfRegistration: string;
  email: string;
  phone?: string;
  bio?: string;
  onboardingSummary?: string; 
  age?: number; 
  gender?: string; 
  dateOfBirth?: string; 
  challenges?: string[]; 
}

export interface ActionItemLink {
  text: string;
  url: string;
}

export interface ActionItemDetails {
  what: string;
  why: string;
  preparedByMaya: string;
  stillMissing: string;
  links?: ActionItemLink[];
  documents?: string[]; 
  deadlines?: string;
}

export interface ActionItem {
  id: string;
  title: string;
  status: ProgressStatus;
  details: ActionItemDetails;
  progressValue?: number; 
  icon?: string; 
}

export interface DocumentItem {
  id: string;
  name: string;
  type: 'PDF' | 'DOCX' | 'TXT' | 'Email' | 'Image';
  dateAdded: string;
  size: string; 
  url?: string | null; 
}

export type ModalType = 
  | 'fullProfile' 
  | 'documents' 
  | 'profilePanel' 
  | 'documentsPanel' 
  | 'actionDetail' 
  | null;

export type ActiveTaskFilterType = 'nextSteps' | 'completed' | null;

export type Language = 'en' | 'es' | 'de';

export interface LanguageOption {
  code: Language;
  name: string; 
}

export interface AiChatMessage {
  id: string;
  role: 'user' | 'model' | 'error';
  text: string;
  isLoading?: boolean;
}

// Workflow specific types
export type WorkflowType = 'formFilling' | 'jobMatching';

export interface WorkflowStepConfig {
  id: string; // e.g., 'welcome', 'personalData', 'documents'
  nameKey: string; // Translation key for the step name, e.g., 'workflowSteps.welcome'
}
