import { UserProfile, ActionItem, DocumentItem, ProgressStatus, LanguageOption, WorkflowStepConfig } from './types';

export const DEFAULT_USER_PROFILE: UserProfile = {
  name: 'Alex Doe',
  alias: 'MayaUser123',
  countryOfOrigin: 'Germany',
  dateOfRegistration: '2024-07-01',
  email: 'alex.doe@example.com',
  phone: '+49 123 4567890',
  bio: 'A proactive individual seeking new opportunities and settling into a new environment. Keen on technology and community engagement.',
  onboardingSummary: `
    During onboarding, Alex expressed a primary need for employment assistance, specifically in the tech sector. 
    They also highlighted the importance of understanding local healthcare options and managing official paperwork efficiently. 
    Alex has a background in software development and is eager to integrate into the local community.
    Key needs identified: Job search, Healthcare navigation, Document management, Local orientation.
  `,
  age: 34,
  gender: "Female", 
  dateOfBirth: "1990-05-15",
  challenges: ["Finding suitable housing", "Learning the local language", "Navigating bureaucracy"]
};

export const FORM_FILLING_WORKFLOW_ID = 'formFillingWorkflow';
export const JOB_MATCHING_WORKFLOW_ID = 'jobMatchingWorkflow';

export const FORM_FILLING_STEPS: WorkflowStepConfig[] = [
  { id: 'welcome', nameKey: 'workflowSteps.welcome' },
  { id: 'personalData', nameKey: 'workflowSteps.personalData' },
  { id: 'documents', nameKey: 'workflowSteps.documents' },
  { id: 'transmitForm', nameKey: 'workflowSteps.transmitForm' },
  { id: 'ama', nameKey: 'workflowSteps.ama' },
  { id: 'complete', nameKey: 'workflowSteps.complete' },
];

export const JOB_MATCHING_STEPS: WorkflowStepConfig[] = [
  { id: 'welcome', nameKey: 'workflowSteps.welcome' },
  { id: 'personalData', nameKey: 'workflowSteps.personalData' }, // Note: will render different content than formFilling
  { id: 'jobMatching', nameKey: 'workflowSteps.jobMatching' },
  { id: 'transmitApplication', nameKey: 'workflowSteps.transmitApplication' },
  { id: 'ama', nameKey: 'workflowSteps.ama' },
  { id: 'complete', nameKey: 'workflowSteps.complete' },
];


export const SUGGESTED_ACTIONS_DATA: ActionItem[] = [
  {
    id: 'action6',
    title: 'Attend BAMF Asylum Interview',
    status: ProgressStatus.NotStarted,
    details: {
      what: 'Attend your scheduled asylum interview at the Federal Office for Migration and Refugees (BAMF) to present your case and answer questions about your asylum application.',
      why: 'This is the most critical step in your asylum process. The interview determines whether you will be granted protection status in Germany. Your answers will directly impact the decision on your application.',
      preparedByMaya: 'Interview preparation materials provided. Practice questions and answers reviewed. Required documents organized and verified. Special needs communicated to BAMF.',
      stillMissing: 'Waiting for BAMF to schedule your interview appointment. You will receive official notification with date, time, and location. Bring all required documents and arrive on time.',
      documents: ['BAMF Interview Guidelines.pdf', 'Required Documents Checklist.pdf', 'Practice Questions and Answers.pdf', 'Your Rights During Interview.pdf'],
      deadlines: 'Monitor for BAMF correspondence. Review preparation materials. Ensure all documents are ready. Contact Maya if you have questions about the interview process.',
    },
    icon: 'BuildingOfficeIcon',
  },
  {
    id: 'action1',
    title: 'Register for Local Health Insurance', 
    status: ProgressStatus.NotStarted,
    details: {
      what: 'Complete and submit the application form for public health insurance.',
      why: 'Essential for accessing healthcare services in your new country.',
      preparedByMaya: 'Basic personal information pre-filled in the application form (Form H-123). Digital copy of passport prepared.',
      stillMissing: 'Proof of address (e.g., rental agreement), Bank details for premium payments.',
      links: [{ text: 'Official Health Insurance Portal', url: '#' }, { text: 'FAQ about Health Insurance', url: '#' }],
      documents: ['Form H-123 (pre-filled).pdf', 'Passport Scan.pdf'],
      deadlines: 'Within 3 months of registration.',
    },
    icon: 'ShieldCheckIcon',
  },
  {
    id: 'action2',
    title: 'Apply for a Tax Identification Number',
    status: ProgressStatus.InProgress,
    progressValue: 60,
    details: {
      what: 'Submit the request for a personal tax ID to the local tax office.',
      why: 'Required for employment and managing financial affairs.',
      preparedByMaya: 'Application form (Form T-45) partially filled. List of required documents compiled.',
      stillMissing: 'Certified copy of birth certificate (if applicable), Signature on Form T-45.',
      links: [{ text: 'Tax Office Information', url: '#' }],
      deadlines: 'As soon as possible, ideally before starting employment.',
    },
    icon: 'IdentificationIcon',
  },
   {
    id: 'action3',
    title: 'Open a Local Bank Account',
    status: ProgressStatus.Completed,
    details: {
      what: 'Visit a local bank branch or use an online service to open a current account.',
      why: 'Necessary for receiving salary, paying bills, and daily transactions.',
      preparedByMaya: 'Comparison of local banks provided. Checklist of typical documents needed.',
      stillMissing: 'Completed during previous interactions. Account details are now securely stored.',
      documents: ['Bank Comparison Sheet.pdf', 'Bank Account Opening Checklist.pdf'],
    },
    icon: 'CreditCardIcon',
  },
  {
    id: 'action5',
    title: 'Complete BAMF Asylum Interview Form',
    status: ProgressStatus.Completed,
    details: {
      what: 'Complete the official BAMF form for asylum interview preparation, including personal data and vulnerability assessment according to § 8 Abs. 1b AsylG.',
      why: 'Required for proper asylum interview preparation and ensures BAMF is aware of any special needs or vulnerabilities during the hearing process.',
      preparedByMaya: 'Form pre-filled with personal information. Vulnerability checklist completed based on user profile. Special circumstances documented.',
      stillMissing: 'Form has been completed and submitted to BAMF. Interview scheduling confirmation pending.',
      documents: ['BAMF Interview Preparation Form.pdf', 'Vulnerability Assessment Checklist.pdf', 'Special Circumstances Documentation.pdf'],
    },
    icon: 'ClipboardDocumentListIcon',
  },
  {
    id: 'action4',
    title: 'Prepare for Job Interview',
    status: ProgressStatus.InProgress,
    progressValue: 40,
    details: {
      what: 'Research common interview questions, practice answers, and prepare questions to ask the interviewer.',
      why: 'Increases confidence and chances of a successful interview outcome.',
      preparedByMaya: 'List of common behavioral questions compiled. Company research links provided.',
      stillMissing: 'Tailored answers to specific job descriptions. List of personal strengths and weaknesses.',
      links: [{ text: 'Tips for Job Interviews', url: '#' }],
      deadlines: 'Ongoing, intensify before scheduled interviews.',
    },
    icon: 'BriefcaseIcon', 
  },
  // Removed workflow tasks from here:
  // {
  //   id: FORM_FILLING_WORKFLOW_ID,
  //   title: 'formFillingWorkflowActionTitle', 
  //   status: ProgressStatus.InProgress, 
  //   details: {
  //     what: 'Begin the process of filling out the asylum registration form with Maya\'s assistance.',
  //     why: 'This is a crucial step for your application process.',
  //     preparedByMaya: 'Maya is ready to guide you step-by-step.',
  //     stillMissing: 'Your input for various sections of the form.',
  //   },
  //   icon: 'ClipboardDocumentListIcon', 
  // },
  // {
  //   id: JOB_MATCHING_WORKFLOW_ID,
  //   title: 'jobMatchingWorkflowActionTitle', 
  //   status: ProgressStatus.InProgress, 
  //   details: {
  //     what: 'Let Maya help you find job opportunities based on your profile and preferences.',
  //     why: 'Finding suitable employment is a key step towards integration.',
  //     preparedByMaya: 'Maya will ask for your skills and preferences to match you with jobs.',
  //     stillMissing: 'Information about your qualifications and job expectations.',
  //   },
  //   icon: 'BriefcaseIcon', 
  // }
];

export const DOCUMENTS_DATA: DocumentItem[] = [
  // Only one PDF document that will be dynamically loaded from backend
  { id: 'doc1', name: 'Application Document.pdf', type: 'PDF', dateAdded: '2024-07-10', size: 'Loading...', url: null },
];

export const AVAILABLE_LANGUAGES_RAW: LanguageOption[] = [
  { code: 'en', name: 'English' },
  { code: 'es', name: 'Español' },
  { code: 'de', name: 'Deutsch' },
];
