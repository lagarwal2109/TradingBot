import React, { useState, useEffect, useMemo } from 'react';
import { UserProfile, ModalType, ActionItem, ProgressStatus, ActiveTaskFilterType, DocumentItem, Language, WorkflowType, WorkflowStepConfig, AiChatMessage } from './types';
import { DEFAULT_USER_PROFILE, SUGGESTED_ACTIONS_DATA, DOCUMENTS_DATA, FORM_FILLING_WORKFLOW_ID, JOB_MATCHING_WORKFLOW_ID, FORM_FILLING_STEPS, JOB_MATCHING_STEPS } from './constants';
import { apiService, PaginationInfo } from './services/api';
import BasicInfo from './components/BasicInfo';
import SuggestedActionsSection from './components/SuggestedActionsSection';
import LeftSlidingPanel from './components/LeftSlidingPanel';
import FullProfileModalContent from './components/FullProfileModalContent';
import DocumentsModalContent from './components/DocumentsModalContent';
import ActionButton from './components/ActionButton';
import LanguageLearning from './components/LanguageLearning';
import LanguageAndExportPanel from './components/LanguageAndExportPanel';

import ChatHistorySection from './components/ChatHistory';
import ActionDetailModal from './components/ActionDetailModal'; 
import MayaAiAssistant from './components/MayaAiAssistant';
import WorkflowHostPage from './components/WorkflowHostPage'; // New Import

// Import the logo
const mayaCodeLogo = new URL('/mayacode-logo.png', import.meta.url).href;

import { 
  UserCircleIcon, 
  FolderIcon, 
  CheckCircleIcon,
  ClipboardDocumentListIcon,
  AcademicCapIcon
} from './components/icons';

// Texts for translation
const texts: Record<Language, any> = {
  en: {
    dashboardTitle: "MayaCode Dashboard",
    loadingMessage: "Loading Dashboard...",
    profilePanelTitle: "Full Profile",
    documentsModalTitle: "Documents & Forms",
    actionDetailModalTitle: "Task Details",
    viewFullProfile: "Profile",
    documentsAndForms: "Documents & Forms",
    languageLearning: "Language Learning",
    sidebarLabels: {
      nextSteps: "Not Started",
      completedTasks: "Completed Tasks",
      formFillingWorkflow: "",
      jobMatchingWorkflow: "", 
    },
    basicInfo: {
      onboardingCompletion: "Onboarding {percentage}% Complete",
      welcomeMessage: "Welcome!",
    },
    progressStatus: {
      [ProgressStatus.NotStarted]: "Not Started",
      [ProgressStatus.InProgress]: "In Progress",
      [ProgressStatus.Completed]: "Completed",
    },
    suggestedActions: {
      nextStepsTitle: "Not Started", 
      completedTasksTitle: "Completed Tasks",
      suggestedActionsTitle: "Suggested Actions",
      currentTasksTitle: "Current Tasks", 
    },
    actionTitles: { 
      formFillingWorkflowActionTitle: "",
      jobMatchingWorkflowActionTitle: "",
    },
    emptyDashboard: {
      generic: "No actions to display for this category.",
      noInProgressTasks: "No tasks are currently in progress. Well done!",
      noNotStartedTasks: "You're all caught up with your next steps!",
      noCompletedTasks: "No tasks have been completed yet.",
    },
    fullProfileContent: { 
        fullName: "Full Name",
        alias: "Alias",
        age: "Age",
        gender: "Gender",
        dateOfBirth: "Date of Birth",
        email: "Email Address",
        phone: "Phone Number",
        countryOfOrigin: "Country of Origin",
        dateOfRegistration: "Date of Registration",
        keyChallenges: "Key Challenges",
        bio: "Biography",
        onboardingSummary: "Onboarding Summary",
     },
    documentsContent: { 
        documentTypes: {
            'PDF': "PDF Document",
            'DOCX': "Word Document",
            'TXT': "Text File",
            'Email': "Email Message",
            'Image': "Image File",
        },
        previewNotAvailable: "Preview not available for this document.",
        downloadNotAvailable: "Download not available for this document.",
        simulatingSend: "Simulating send for: {docName}",
        noDocumentsFound: "No documents found.",
        docTypeLabel: "Type:",
        docAddedLabel: "Added:",
        docSizeLabel: "Size:",
        previewButton: "Preview",
        downloadButton: "Download",
        sendButton: "Send",
     },
    exportProfileOptions: { 
        pdf: "PDF", 
        json: "JSON", 
        text: "TXT" 
    },
    pdfExportAlert: "PDF export simulation: Downloading as TXT.", 
    languageNames: { en: "English", es: "Español", de: "Deutsch" },
    actionDetailModal: { 
        title: "Task Details",
        what: "What this step involves:",
        why: "Why it's important:",
        preparedByMaya: "Prepared by Maya:",
        stillMissing: "What's still needed:",
        progress: "Current Progress:",
        helpfulLinks: "Helpful Links:",
        relevantDocuments: "Relevant Documents:",
        deadlines: "Deadlines:",
    },
    sidebarCombinedMenu: { 
        triggerAriaLabel: "Open settings",
        exportSectionTitle: "Export Profile",
        languageSectionTitle: "Select Language"
    },
    mayaAiAssistant: { 
        greetingMessage: "Hello! I'm Maya, your AI assistant. How can I help you today?",
        errorMessage: "Sorry, I encountered an error. Please try again.",
        apiKeyMissingError: "Chat unavailable: API_KEY is not configured in the environment.", // New
        chatInputPlaceholder: "Ask Maya anything...",
        chatUnavailablePlaceholder: "Chat unavailable: Check configuration", // Kept as a more general fallback
        typingIndicator: "Maya is typing...",
        chatWindowTitle: "Maya AI Assistant",
        openChatLabel: "Open Maya AI Assistant",
        closeChatLabel: "Close chat",
        sendButtonLabel: "Send message",
    },
    workflowTitles: {
      formFilling: "",
      jobMatching: "Job Matching - Career Placement" 
    },
    workflowSteps: {
      welcome: "Welcome",
      personalData: "Personal Data",
      documents: "Documents",
      transmitForm: "Transmit Form",
      jobMatching: "Job Matching",
      transmitApplication: "Transmit Application",
      ama: "AMA (Ask Maya Anything)",
      complete: "Complete",
    },
    workflowNav: {
      backToDashboard: "Back to Dashboard",
      next: "Next",
      previous: "Previous",
      finish: "Finish",
    },
    workflowMessages: {
        welcomeIntro: "Welcome to this guided process. Let's get started!",
        completionMessage: "You have successfully completed this process. Maya will keep you updated on any next steps."
    },
    workflowPageLabels: {
      formFillingSummaryTitle: "Asylum Form Collection",
      jobMatchingSummaryTitle: "Job Matching Profile",
      mayaAvatarText: "Maya Avatar",
      summaryLabels: {
        name: "Name",
        dateOfBirth: "Date of Birth",
        nationality: "Nationality",
        placeOfBirth: "Place of Birth",
        maritalStatus: "Marital Status",
        vulnerabilities: "Vulnerabilities",
        skills: "Skills",
        experience: "Experience",
        education: "Education",
        languages: "Languages",
        jobMatches: "Job Matches",
        application: "Application",
        notCollected: "not collected yet",
        assessmentPending: "assessment pending",
        positionsFound: "{count} positions found",
        notStarted: "not started",
      }
    }
  },
  es: { 
    dashboardTitle: "Panel de MayaCode",
    loadingMessage: "Cargando Panel...",
    profilePanelTitle: "Perfil Completo",
    documentsModalTitle: "Documentos y Formularios",
    actionDetailModalTitle: "Detalles de la Tarea",
    viewFullProfile: "Perfil",
    documentsAndForms: "Documentos y Formularios",
    languageLearning: "Aprendizaje de Idiomas",
     sidebarLabels: {
      nextSteps: "No Iniciadas",
      completedTasks: "Completadas",
      formFillingWorkflow: "Registro de Asilo", 
      jobMatchingWorkflow: "Buscar Empleo", 
    },
     basicInfo: {
      registered: "Registrado",
      onboardingCompletion: "Incorporación {percentage}% Completa",
      welcomeMessage: "¡Bienvenido!",
    },
    progressStatus: {
      [ProgressStatus.NotStarted]: "No Iniciada",
      [ProgressStatus.InProgress]: "En Progreso",
      [ProgressStatus.Completed]: "Completada",
    },
    suggestedActions: {
      nextStepsTitle: "No Iniciadas",
      completedTasksTitle: "Tareas Completadas",
      suggestedActionsTitle: "Acciones Sugeridas",
      currentTasksTitle: "Tareas Actuales",
    },
    actionTitles: {
      formFillingWorkflowActionTitle: "Iniciar Formulario de Registro de Asilo",
      jobMatchingWorkflowActionTitle: "Buscar Oportunidades de Empleo",
    },
     emptyDashboard: {
      generic: "No hay acciones para mostrar en esta categoría.",
      noInProgressTasks: "No hay tareas actualmente en progreso. ¡Bien hecho!",
      noNotStartedTasks: "¡Estás al día con tus próximos pasos!",
      noCompletedTasks: "Aún no se han completado tareas.",
    },
    languageNames: { en: "Inglés", es: "Español", de: "Alemán" },
    mayaAiAssistant: { 
        greetingMessage: "¡Hola! Soy Maya, tu asistente de IA. ¿Cómo puedo ayudarte hoy?",
        errorMessage: "Lo siento, encontré un error. Por favor, inténtalo de nuevo.",
        apiKeyMissingError: "Chat no disponible: API_KEY no está configurada en el entorno.", // New
        chatInputPlaceholder: "Pregúntale a Maya lo que sea...",
        chatUnavailablePlaceholder: "Chat no disponible: Comprueba la configuración",
        typingIndicator: "Maya está escribiendo...",
        chatWindowTitle: "Asistente IA Maya",
        openChatLabel: "Abrir Asistente IA Maya",
        closeChatLabel: "Cerrar chat",
        sendButtonLabel: "Enviar mensaje",
    },
    workflowTitles: {
      formFilling: "Formulario de Registro de Asilo",
      jobMatching: "Búsqueda de Empleo - Colocación Profesional" 
    },
    workflowSteps: {
      welcome: "Bienvenida",
      personalData: "Datos Personales",
      documents: "Documentos",
      transmitForm: "Enviar Formulario",
      jobMatching: "Búsqueda de Empleo",
      transmitApplication: "Enviar Solicitud",
      ama: "Pregunta a Maya",
      complete: "Completado",
    },
    workflowNav: {
      backToDashboard: "Volver al Panel",
      next: "Siguiente",
      previous: "Anterior",
      finish: "Finalizar",
    },
     workflowMessages: {
        welcomeIntro: "Bienvenido a este proceso guiado. ¡Empecemos!",
        completionMessage: "Has completado este proceso con éxito. Maya te mantendrá informado sobre los próximos pasos."
    },
    workflowPageLabels: {
      formFillingSummaryTitle: "Recopilación Formulario Asilo",
      jobMatchingSummaryTitle: "Perfil de Búsqueda de Empleo",
      mayaAvatarText: "Avatar de Maya",
      summaryLabels: {
        name: "Nombre",
        dateOfBirth: "Fecha de Nacimiento",
        nationality: "Nacionalidad",
        placeOfBirth: "Lugar de Nacimiento",
        maritalStatus: "Estado Civil",
        vulnerabilities: "Vulnerabilidades",
        skills: "Habilidades",
        experience: "Experiencia",
        education: "Educación",
        languages: "Idiomas",
        jobMatches: "Coincidencias de Empleo",
        application: "Solicitud",
        notCollected: "aún no recopilado",
        assessmentPending: "evaluación pendiente",
        positionsFound: "{count} posiciones encontradas",
        notStarted: "no iniciada",
      }
    }
  },
  de: { 
    dashboardTitle: "MayaCode Dashboard",
    loadingMessage: "Dashboard wird geladen...",
    profilePanelTitle: "Vollständiges Profil",
    documentsModalTitle: "Dokumente & Formulare",
    actionDetailModalTitle: "Aufgabendetails",
    viewFullProfile: "Profil",
    documentsAndForms: "Dokumente & Formulare",
    languageLearning: "Sprachenlernen",
    basicInfo: {
      registered: "Registriert",
      onboardingCompletion: "Onboarding zu {percentage}% abgeschlossen",
      welcomeMessage: "Willkommen!",
    },
    sidebarLabels: {
      nextSteps: "Nicht Gestartet",
      completedTasks: "Abgeschlossen",
      formFillingWorkflow: "Asylregistrierung", 
      jobMatchingWorkflow: "Jobs finden", 
    },
    progressStatus: {
      [ProgressStatus.NotStarted]: "Nicht Gestartet",
      [ProgressStatus.InProgress]: "In Bearbeitung",
      [ProgressStatus.Completed]: "Abgeschlossen",
    },
    suggestedActions: {
      nextStepsTitle: "Nicht Gestartet",
      completedTasksTitle: "Abgeschlossene Aufgaben",
      suggestedActionsTitle: "Vorgeschlagene Aktionen",
      currentTasksTitle: "Aktuelle Aufgaben",
    },
    actionTitles: {
      formFillingWorkflowActionTitle: "Asylantragsformular starten",
      jobMatchingWorkflowActionTitle: "Arbeitsmöglichkeiten finden",
    },
    emptyDashboard: {
      generic: "Keine Aktionen für diese Kategorie vorhanden.",
      noInProgressTasks: "Derzeit sind keine Aufgaben in Bearbeitung. Gut gemacht!",
      noNotStartedTasks: "Sie sind mit Ihren nächsten Schritten auf dem Laufenden!",
      noCompletedTasks: "Es wurden noch keine Aufgaben abgeschlossen.",
    },
    languageNames: { en: "Englisch", es: "Spanisch", de: "Deutsch" },
    mayaAiAssistant: { 
        greetingMessage: "Hallo! Ich bin Maya, deine KI-Assistentin. Wie kann ich dir heute helfen?",
        errorMessage: "Entschuldigung, ein Fehler ist aufgetreten. Bitte versuche es erneut.",
        apiKeyMissingError: "Chat nicht verfügbar: API_KEY ist nicht in der Umgebung konfiguriert.", // New
        chatInputPlaceholder: "Frag Maya etwas...",
        chatUnavailablePlaceholder: "Chat nicht verfügbar: Konfiguration prüfen",
        typingIndicator: "Maya tippt...",
        chatWindowTitle: "Maya KI Assistent",
        openChatLabel: "Maya KI Assistent öffnen",
        closeChatLabel: "Chat schließen",
        sendButtonLabel: "Nachricht senden",
    },
    workflowTitles: {
      formFilling: "Asylantragsformular",
      jobMatching: "Jobsuche - Stellenvermittlung" 
    },
    workflowSteps: {
      welcome: "Willkommen",
      personalData: "Persönliche Daten",
      documents: "Dokumente",
      transmitForm: "Formular Übermitteln",
      jobMatching: "Jobsuche",
      transmitApplication: "Bewerbung Senden",
      ama: "Frag Maya",
      complete: "Abgeschlossen",
    },
    workflowNav: {
      backToDashboard: "Zurück zum Dashboard",
      next: "Weiter",
      previous: "Zurück",
      finish: "Abschließen",
    },
    workflowMessages: {
        welcomeIntro: "Willkommen zu diesem geführten Prozess. Lass uns anfangen!",
        completionMessage: "Sie haben diesen Vorgang erfolgreich abgeschlossen. Maya hält Sie über die nächsten Schritte auf dem Laufenden."
    },
    workflowPageLabels: {
      formFillingSummaryTitle: "Sammlung Asylformular",
      jobMatchingSummaryTitle: "Profil Jobsuche",
      mayaAvatarText: "Maya Avatar",
      summaryLabels: {
        name: "Name",
        dateOfBirth: "Geburtsdatum",
        nationality: "Nationalität",
        placeOfBirth: "Geburtsort",
        maritalStatus: "Familienstand",
        vulnerabilities: "Schwachstellen",
        skills: "Fähigkeiten",
        experience: "Erfahrung",
        education: "Ausbildung",
        languages: "Sprachen",
        jobMatches: "Job-Übereinstimmungen",
        application: "Bewerbung",
        notCollected: "noch nicht erfasst",
        assessmentPending: "Bewertung ausstehend",
        positionsFound: "{count} Stellen gefunden",
        notStarted: "nicht gestartet",
      }
    }
  }
};

// Improved translation merging logic
const T_EN = texts.en;
(['es', 'de'] as Language[]).forEach(langCode => {
  const T_LANG = texts[langCode];
  Object.keys(T_EN).forEach(key => {
    const enValue = T_EN[key];
    const langValue = T_LANG[key];

    if (typeof enValue === 'object' && enValue !== null && !Array.isArray(enValue)) {
      T_LANG[key] = { ...enValue, ...(typeof langValue === 'object' && langValue !== null ? langValue : {}) };
      Object.keys(enValue).forEach(nestedKey => {
        if (typeof enValue[nestedKey] === 'object' && enValue[nestedKey] !== null && !Array.isArray(enValue[nestedKey])) {
          T_LANG[key][nestedKey] = { 
            ...enValue[nestedKey], 
            ...(T_LANG[key][nestedKey] && typeof T_LANG[key][nestedKey] === 'object' ? T_LANG[key][nestedKey] : {}) 
          };
        } else if (T_LANG[key] && T_LANG[key][nestedKey] === undefined) {
           T_LANG[key][nestedKey] = enValue[nestedKey];
        }
      });
    } else if (langValue === undefined) {
      T_LANG[key] = enValue;
    }
  });
});


const initialProcessedActions = SUGGESTED_ACTIONS_DATA.map(action => {
  // Convert string status to ProgressStatus enum
  let status: ProgressStatus;
  switch (action.status) {
    case 'Not Started':
      status = ProgressStatus.NotStarted;
      break;
    case 'In Progress':
      status = ProgressStatus.InProgress;
      break;
    case 'Completed':
      status = ProgressStatus.Completed;
      break;
    default:
      status = ProgressStatus.NotStarted;
  }

  // Process icons and return transformed action
  if (action.id === 'action1' && !action.icon) return { ...action, status, icon: 'ShieldCheckIcon' };
  if (action.id === 'action2' && !action.icon) return { ...action, status, icon: 'IdentificationIcon', progressValue: action.progressValue || 60 };
  if (action.id === 'action3' && !action.icon) return { ...action, status, icon: 'CreditCardIcon' };
  return { ...action, status };
});

interface ActiveWorkflowDetails {
  type: WorkflowType;
  title: string;
  steps: WorkflowStepConfig[];
}

const App: React.FC = () => {
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);
  const [documentsList, setDocumentsList] = useState<DocumentItem[]>([]);
  const [actionsPagination, setActionsPagination] = useState<PaginationInfo | null>(null);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [activeModal, setActiveModal] = useState<ModalType | null>(null);
  const [selectedActionForDetails, setSelectedActionForDetails] = useState<ActionItem | null>(null);
  const [activeTaskFilter, setActiveTaskFilter] = useState<ActiveTaskFilterType>(null);
  const [currentLanguage, setCurrentLanguage] = useState<Language>('en');
  const [showLanguageLearning, setShowLanguageLearning] = useState(false);
  const [languageLearningKey, setLanguageLearningKey] = useState(Date.now().toString());
  const [currentWorkflowStepIndex, setCurrentWorkflowStepIndex] = useState(0);
  const [workflowData, setWorkflowData] = useState<Record<string, any>>({});
  const [activeWorkflowDetails, setActiveWorkflowDetails] = useState<ActiveWorkflowDetails | null>(null);
  const [conversationToLoad, setConversationToLoad] = useState<AiChatMessage[] | null>(null);

  const [suggestedActions, setSuggestedActions] = useState<ActionItem[]>([]); 
  const [onboardingCompletion, setOnboardingCompletion] = useState<number>(0);

  const T = useMemo(() => texts[currentLanguage], [currentLanguage]);

  const handleLanguageChange = (lang: Language) => {
    setCurrentLanguage(lang);
  };

  // Function to load more actions
  const loadMoreActions = async () => {
    if (!actionsPagination?.hasMore || isLoadingMore) return;
    
    setIsLoadingMore(true);
    try {
      const statusFilter = activeTaskFilter === 'nextSteps' ? ProgressStatus.NotStarted : 
                          activeTaskFilter === 'completed' ? ProgressStatus.Completed : 
                          activeTaskFilter === null ? ProgressStatus.InProgress : undefined;

      const response = await apiService.getUserActions({
        page: actionsPagination.nextPage || 1,
        loadMore: true,
        status: statusFilter
      });

      if (response.success) {
        setSuggestedActions(prev => [...prev, ...response.data.actions]);
        setActionsPagination(response.data.pagination);
      }
    } catch (error) {
      console.error('Failed to load more actions:', error);
    } finally {
      setIsLoadingMore(false);
    }
  };

  // Function to calculate onboarding completion percentage
  const calculateOnboardingCompletion = (actions: ActionItem[]): number => {
    if (actions.length === 0) return 0;
    
    let totalCompletionValue = 0;
    actions.forEach(action => {
      if (action.status === ProgressStatus.Completed) {
        totalCompletionValue += 100;
      } else if (action.status === ProgressStatus.InProgress && action.progressValue !== undefined) {
        totalCompletionValue += action.progressValue;
      }
      // NotStarted actions contribute 0
    });
    
    return Math.round(totalCompletionValue / actions.length);
  };

  useEffect(() => {
    const fetchInitialData = async () => {
      setIsLoading(true);
      console.log('🔄 Starting to fetch initial data...');
      
      try {
        // Fetch user data from backend
        console.log('📡 Calling apiService.getUser()...');
        const userResponse = await apiService.getUser();
        console.log('👤 User response:', userResponse);
        
        if (userResponse.success) {
          console.log('✅ User data received:', userResponse.data);
          
          // Transform PascalCase backend fields to camelCase for frontend
          const transformedData: UserProfile = {
            name: userResponse.data.Name || DEFAULT_USER_PROFILE.name,
            alias: userResponse.data.Alias || DEFAULT_USER_PROFILE.alias,
            countryOfOrigin: userResponse.data.OriginCountry || DEFAULT_USER_PROFILE.countryOfOrigin,
            dateOfRegistration: userResponse.data.ArrivalDate || '2024-07-01',
            email: userResponse.data.Email || DEFAULT_USER_PROFILE.email,
            phone: userResponse.data.Phone || DEFAULT_USER_PROFILE.phone,
            bio: userResponse.data.LongTermPlans || DEFAULT_USER_PROFILE.bio,
            onboardingSummary: userResponse.data.MigrationReason || DEFAULT_USER_PROFILE.onboardingSummary,
            age: userResponse.data.Age || DEFAULT_USER_PROFILE.age,
            gender: userResponse.data.Gender || DEFAULT_USER_PROFILE.gender,
            dateOfBirth: userResponse.data.DateOfBirth || DEFAULT_USER_PROFILE.dateOfBirth,
            challenges: userResponse.data.Challenges || DEFAULT_USER_PROFILE.challenges
          };
          
          setUserProfile(transformedData);
          console.log('🔄 Transformed user data:', transformedData);
        }
      } catch (error) {
        console.error('❌ Failed to fetch user data:', error);
        console.log('🔄 Using fallback static user data...');
        
        // Fallback to static user profile data if API fails
        setUserProfile(DEFAULT_USER_PROFILE); 
      } finally {
        // Always use hardcoded actions and documents from constants
        console.log('📋 Using hardcoded actions from constants');
        setSuggestedActions(initialProcessedActions);
        setDocumentsList(DOCUMENTS_DATA);
        
        // Calculate completion based on hardcoded actions
        const completion = calculateOnboardingCompletion(initialProcessedActions);
        console.log('📊 Calculated onboarding completion:', completion + '%');
        setOnboardingCompletion(completion);
        
        setIsLoading(false);
        console.log('✅ Initial data loading completed');
      }
    };

    fetchInitialData();
  }, []);

  // No longer polling for user data - now only fetched when profile panel is opened
  // Effect to refresh actions if needed
  useEffect(() => {
    // Only update filtered actions if not in a workflow and filter changes
    if (!activeWorkflowDetails) {
      fetchFilteredActions();
    }
  }, [activeWorkflowDetails, activeTaskFilter, userProfile]);
  

  // Function to filter actions from hardcoded constants (moved outside useEffect for reusability)
  const fetchFilteredActions = () => {
    if (userProfile) { // Only process if we have initial data
      const statusFilter = activeTaskFilter === 'nextSteps' ? ProgressStatus.NotStarted : 
                          activeTaskFilter === 'completed' ? ProgressStatus.Completed : 
                          activeTaskFilter === null ? ProgressStatus.InProgress : undefined;

      // Filter the hardcoded actions based on status filter
      const filteredActions = statusFilter !== undefined 
        ? initialProcessedActions.filter(action => action.status === statusFilter)
        : initialProcessedActions;
        
      // Update the actions list with filtered results
      setSuggestedActions(filteredActions);
      
      // Calculate completion based on hardcoded actions
      const completion = calculateOnboardingCompletion(filteredActions);
      setOnboardingCompletion(completion);
    }
  };

  // Effect to fetch filtered actions when filter changes
  useEffect(() => {
    fetchFilteredActions();
  }, [activeTaskFilter, userProfile]);

  const startWorkflow = (workflowType: WorkflowType) => {
    setWorkflowData({}); 
    setCurrentWorkflowStepIndex(0);
    setActiveTaskFilter(null); // Ensure dashboard shows "Current Tasks" when returning from a workflow started via sidebar
    if (workflowType === 'formFilling') {
      setActiveWorkflowDetails({ type: 'formFilling', title: T.workflowTitles.formFilling, steps: FORM_FILLING_STEPS });
    } else if (workflowType === 'jobMatching') {
      setActiveWorkflowDetails({ type: 'jobMatching', title: T.workflowTitles.jobMatching, steps: JOB_MATCHING_STEPS });
    }
  };

  const exitWorkflow = () => {
    setActiveWorkflowDetails(null);
    setCurrentWorkflowStepIndex(0);
  };

  const goToNextStep = () => {
    if (activeWorkflowDetails && currentWorkflowStepIndex < activeWorkflowDetails.steps.length - 1) {
      setCurrentWorkflowStepIndex(prev => prev + 1);
    } else if (activeWorkflowDetails && currentWorkflowStepIndex === activeWorkflowDetails.steps.length - 1) {
      exitWorkflow(); 
    }
  };

  const goToPreviousStep = () => {
    if (currentWorkflowStepIndex > 0) {
      setCurrentWorkflowStepIndex(prev => prev - 1);
    }
  };
  
  const jumpToStep = (index: number) => {
     if (activeWorkflowDetails && index >= 0 && index < activeWorkflowDetails.steps.length) {
      setCurrentWorkflowStepIndex(index);
    }
  };

  const updateWorkflowData = (data: Record<string, any>) => {
    setWorkflowData(prev => ({ ...prev, ...data }));
  };

  const handleTaskItemClick = (action: ActionItem) => {
    if (action.id === FORM_FILLING_WORKFLOW_ID) {
      startWorkflow('formFilling');
    } else if (action.id === JOB_MATCHING_WORKFLOW_ID) {
      startWorkflow('jobMatching');
    } else {
      setSelectedActionForDetails(action);
      setActiveModal('actionDetail');
    }
  };
  
  // Create a ref to track if profile data fetch should be cancelled
  const profileFetchAbortController = React.useRef<AbortController | null>(null);

  const openModal = async (modalType: ModalType) => {
    // Cancel any ongoing profile fetch if there is one
    if (profileFetchAbortController.current) {
      console.log('🚫 Cancelling previous profile data fetch');
      profileFetchAbortController.current.abort();
      profileFetchAbortController.current = null;
    }
    
    setActiveModal(modalType);
    
    // Fetch fresh user data only when opening the profile panel
    if (modalType === 'profilePanel') {
      console.log('📡 Fetching fresh user profile data for profile panel...');
      
      // Create a new abort controller for this fetch
      profileFetchAbortController.current = new AbortController();
      const signal = profileFetchAbortController.current.signal;
      
      try {
        // We'll simulate adding the abort signal to the API call
        // In a real implementation, you'd pass this signal to the fetch call
        const userResponse = await apiService.getUser();
        
        // Check if request was aborted before updating state
        if (signal.aborted) {
          console.log('🛂 Profile data fetch was aborted, not updating state');
          return;
        }
        
        if (userResponse.success) {
          console.log('✅ Fresh user data received for profile panel');
          setUserProfile(userResponse.data);
        }
      } catch (error) {
        // Ignore aborted fetch errors
        if (signal.aborted) {
          console.log('🛂 Profile data fetch was aborted');
          return;
        }
        console.error('❌ Failed to fetch fresh user data:', error);
      } finally {
        // Clear the abort controller reference if it's still this request's controller
        if (profileFetchAbortController.current?.signal === signal) {
          profileFetchAbortController.current = null;
        }
      }
    }
  };
  const closeModal = () => {
    // Cancel any ongoing profile fetch if there is one
    if (profileFetchAbortController.current) {
      console.log('🚫 Cancelling profile data fetch on modal close');
      profileFetchAbortController.current.abort();
      profileFetchAbortController.current = null;
    }
    
    setActiveModal(null);
    setSelectedActionForDetails(null); 
  };

  const navigateToHome = () => {
    setActiveTaskFilter(null);
    setShowLanguageLearning(false);
    if(activeWorkflowDetails) exitWorkflow(); 
  };
  
  const handleChatHistoryClick = (conversation: AiChatMessage[]) => {
    setConversationToLoad(conversation);
    // Reset after a small delay to ensure the effect in MayaAiAssistant can detect the change
    setTimeout(() => setConversationToLoad(null), 100);
  };

  if (isLoading || !userProfile) { 
    return (
      <div className="flex items-center justify-center min-h-screen bg-[#f4f0ff] text-text-primary text-lg">
        <img src={mayaCodeLogo} alt="MayaCode Logo" className="w-8 h-8 text-accent mr-3 animate-spin" />
        {T.loadingMessage || "Loading Dashboard..."}
      </div>
    );
  }

  if (activeWorkflowDetails) {
    return (
      <WorkflowHostPage
        workflowDetails={activeWorkflowDetails}
        currentStepIndex={currentWorkflowStepIndex}
        workflowData={workflowData}
        userProfile={userProfile} 
        onExitWorkflow={exitWorkflow}
        onGoToNextStep={goToNextStep}
        onGoToPreviousStep={goToPreviousStep}
        onJumpToStep={jumpToStep}
        onUpdateWorkflowData={updateWorkflowData}
        T={T}
      />
    );
  }

  return (
    <div className="flex min-h-screen bg-[#f4f0ff] text-text-primary">
      <aside className="w-72 bg-card text-text-primary flex flex-col fixed top-0 left-0 h-full shadow-xl z-30 border-r border-border-color overflow-y-auto">
        <div className="p-5 border-b border-border-color">
          <button 
            onClick={navigateToHome} 
            className="flex items-center space-x-2.5 focus:outline-none focus:ring-2 focus:ring-accent rounded-md p-1 -m-1 hover:bg-accent/10 transition-colors"
            aria-label="Go to dashboard home"
          >
            <img src={mayaCodeLogo} alt="MayaCode Logo" className="w-8 h-8" />
            <h1 className="text-lg font-semibold text-accent tracking-tight">
              {T.dashboardTitle}
            </h1>
          </button>
        </div>

        <nav className="flex-grow p-3 space-y-1.5">
          <ActionButton 
            onClick={() => {
              openModal('profilePanel');
              setShowLanguageLearning(false);
            }} 
            variant="sidebar"
            icon={<UserCircleIcon className="w-5 h-5" />}
             isActive={activeModal === 'profilePanel'}
          >
            {T.viewFullProfile}
          </ActionButton>
          <ActionButton 
            onClick={() => {
              openModal('documentsPanel');
              setShowLanguageLearning(false);
            }}
            variant="sidebar"
            icon={<FolderIcon className="w-5 h-5" />}
            isActive={activeModal === 'documentsPanel'}
          >
            {T.documentsAndForms}
          </ActionButton>
          
          <hr className="my-2 border-border-color" />

          <ActionButton
            onClick={() => { 
              setActiveTaskFilter('nextSteps'); 
              setShowLanguageLearning(false);
            }}
            variant="sidebar"
            icon={<ClipboardDocumentListIcon className="w-5 h-5" />}
            isActive={activeTaskFilter === 'nextSteps' && !showLanguageLearning}
          >
            {T.sidebarLabels.nextSteps}
          </ActionButton>
          <ActionButton
            onClick={() => { 
              setActiveTaskFilter('completed');
              setShowLanguageLearning(false);
            }}
            variant="sidebar"
            icon={<CheckCircleIcon className="w-5 h-5" />}
            isActive={activeTaskFilter === 'completed' && !showLanguageLearning}
          >
            {T.sidebarLabels.completedTasks}
          </ActionButton>

          <hr className="my-2 border-border-color" />

          <ActionButton
            onClick={() => { 
              // First, make sure any open panel is closed
              setActiveModal(null);
              
              // Cancel any ongoing profile data fetch
              if (profileFetchAbortController.current) {
                console.log('🚫 Cancelling profile data fetch');
                profileFetchAbortController.current.abort();
                profileFetchAbortController.current = null;
              }
              
              // Two-step approach: first hide any currently showing section
              setShowLanguageLearning(false);
              setActiveTaskFilter(null);
              
              // Then after a very brief delay, generate a new key and show language learning
              // This ensures a clean slate for mounting
              setTimeout(() => {
                const newKey = Date.now().toString();
                console.log(`📣 Showing Language Learning with fresh key: ${newKey}`);
                setLanguageLearningKey(newKey);
                setShowLanguageLearning(true);
              }, 10); // Very minimal delay, just enough to break the render cycle
            }}
            variant="sidebar"
            icon={<AcademicCapIcon className="w-5 h-5" />}
            isActive={showLanguageLearning}
          >
            {T.languageLearning}
          </ActionButton>

          {/* Chat history section below completed tasks */}
          <ChatHistorySection onChatHistoryClick={handleChatHistoryClick} />

          {/* Removed workflow buttons for formFilling and jobMatching as requested */}


        </nav>

        <div className="p-3 border-t border-border-color space-y-3">
          <LanguageAndExportPanel
            currentLanguage={currentLanguage}
            onLanguageChange={handleLanguageChange}
            userProfile={userProfile}
            T={T}
          />
        </div>
      </aside>

      <div className="ml-72 flex-grow flex flex-col">
        <main className="flex-grow p-4 sm:p-6 lg:p-8">
          {showLanguageLearning ? (
            // Completely isolate the Language Learning component from the rest of the app
            // This creates a boundary that prevents prop changes from causing freezes
            <div className="language-learning-container">
              {/* Create an internal shadow copy of userProfile to prevent prop changes during rendering */}
              {(() => {
                // Log mount information
                console.log(`📘 Mounting isolated Language Learning with key: ${languageLearningKey}`);
                
                // Create a clone of userProfile to isolate it from changes
                const isolatedUserProfile = JSON.parse(JSON.stringify(userProfile));
                
                return (
                  <LanguageLearning 
                    key={languageLearningKey} 
                    userProfile={isolatedUserProfile}
                    T={T} 
                  />
                );
              })()}
            </div>
          ) : (
            <div className="flex-grow">
              <div className="mb-6">
                 {userProfile && <BasicInfo user={userProfile} onboardingCompletion={onboardingCompletion} T={T} /> }
              </div>
              <SuggestedActionsSection 
                actions={
                  activeTaskFilter === null
                    ? suggestedActions.filter(action => action.status === ProgressStatus.InProgress)
                    : activeTaskFilter === 'nextSteps'
                    ? suggestedActions.filter(action => action.status === ProgressStatus.NotStarted)
                    : activeTaskFilter === 'completed'
                    ? suggestedActions.filter(action => action.status === ProgressStatus.Completed)
                    : suggestedActions
                }
                onOpenActionDetails={handleTaskItemClick} 
                activeFilter={activeTaskFilter} 
                T={T}
                hasMore={actionsPagination?.hasMore || false}
                onLoadMore={loadMoreActions}
                isLoadingMore={isLoadingMore}
              />
            </div>
          )}
        </main>
      </div>

      {userProfile && (
          <LeftSlidingPanel
            isOpen={activeModal === 'profilePanel'}
            onClose={closeModal}
            title={T.profilePanelTitle}
          >
            <FullProfileModalContent userProfile={userProfile} T={T} />
          </LeftSlidingPanel>
      )}
      
      <LeftSlidingPanel
        isOpen={activeModal === 'documentsPanel'}
        onClose={closeModal}
        title={T.documentsModalTitle}
      >
        <DocumentsModalContent documents={documentsList} T={T} />
      </LeftSlidingPanel>

      {selectedActionForDetails && (
        <ActionDetailModal
          isOpen={activeModal === 'actionDetail'}
          onClose={closeModal}
          action={selectedActionForDetails}
          T={T}
        />
      )}
      <MayaAiAssistant T={T} loadConversation={conversationToLoad} />
    </div>
  );
};

export default App;
