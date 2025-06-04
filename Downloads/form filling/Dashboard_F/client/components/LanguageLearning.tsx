import React, { useMemo, useState, useEffect } from 'react';
import { UserProfile } from '../types';
import ActionButton from './ActionButton';
import { SparklesIcon } from './icons';

interface LanguageLearningProps {
  userProfile: UserProfile;
  T: any; // Translation object
}

interface LessonCard {
  id: string;
  level: string;
  title: string;
  description: string;
  status: 'completed' | 'inProgress' | 'available' | 'locked';
  progress: string;
  totalExercises: string;
  buttonText: string;
}

const LanguageLearning: React.FC<LanguageLearningProps> = React.memo(({ userProfile, T }) => {
  // Track component mounting and stabilization
  const [isStabilized, setIsStabilized] = useState(false);
  const [mountTime] = useState(Date.now());
  
  // Use ref to track if there were prop changes during initialization
  const userProfileRef = React.useRef(userProfile);
  const initialRenderComplete = React.useRef(false);
  
  // Track whether userProfile has changed since mount
  const [hasUserProfileChanged, setHasUserProfileChanged] = useState(false);
  
  // Detect userProfile changes after mount and handle them gracefully
  useEffect(() => {
    // Skip the first render effect
    if (!initialRenderComplete.current) {
      initialRenderComplete.current = true;
      userProfileRef.current = userProfile;
      return;
    }
    
    // Detect if userProfile changed after initial render
    if (userProfile !== userProfileRef.current) {
      console.log('📊 UserProfile changed after Language Learning mounted');
      setHasUserProfileChanged(true);
      userProfileRef.current = userProfile;
    }
  }, [userProfile]);
  
  // Stabilization effect - wait until component and data are stable
  useEffect(() => {
    // Allow component to stabilize briefly
    const timer = setTimeout(() => {
      setIsStabilized(true);
    }, 50); // Short delay for stability
    
    // Log mount information for debugging
    console.log(`🔄 Language Learning component mounted at ${new Date(mountTime).toISOString()}`);
    
    return () => {
      clearTimeout(timer);
      console.log('🛑 Language Learning component unmounted');
    };
  }, [mountTime]);

  // Memoize hardcoded lesson data
  const lessons: LessonCard[] = useMemo(() => [
    {
      id: 'a1-1',
      level: 'A1.1',
      title: 'Basic Greetings & Introductions',
      description: 'Learn essential German greetings and how to introduce yourself in various situations.',
      status: 'completed',
      progress: '10/10',
      totalExercises: '10/10 exercises',
      buttonText: 'Review Lesson'
    },
    {
      id: 'a1-2',
      level: 'A1.2',
      title: 'Numbers & Basic Counting',
      description: 'Master German numbers from 1-100 and learn to count in everyday situations.',
      status: 'inProgress',
      progress: '10/10',
      totalExercises: '10/10 exercises',
      buttonText: 'Continue Lesson'
    },
    {
      id: 'a1-3',
      level: 'A1.3',
      title: 'Shopping & Daily Activities',
      description: 'Learn vocabulary for shopping, asking for prices, and describing daily routines.',
      status: 'available',
      progress: '0/12',
      totalExercises: '0/12 exercises',
      buttonText: 'Start Lesson'
    },
    {
      id: 'a2-1',
      level: 'A2.1',
      title: 'Past Tense & Storytelling',
      description: 'Learn to talk about past events and tell simple stories using German past tense.',
      status: 'locked',
      progress: '0/15',
      totalExercises: '0/15 exercises',
      buttonText: 'Complete A1 first'
    },
    {
      id: 'a2-2',
      level: 'A2.2',
      title: 'Job Interview German',
      description: 'Professional German for job interviews, workplace communication, and career development.',
      status: 'locked',
      progress: '0/18',
      totalExercises: '0/18 exercises',
      buttonText: 'Unlock at A2 level'
    },
    {
      id: 'a2-3',
      level: 'A2.3',
      title: 'Advanced Conversations',
      description: 'Complex discussions about opinions, plans, and cultural topics in German.',
      status: 'locked',
      progress: '0/20',
      totalExercises: '0/20 exercises',
      buttonText: 'Complete A2.2 first'
    }
  ], []);

  // Memoize helper functions
  const getStatusStyles = useMemo(() => (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-green-100 text-green-800 border-green-200';
      case 'inProgress':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'available':
        return 'bg-blue-100 text-blue-800 border-blue-200';
      case 'locked':
        return 'bg-gray-100 text-gray-600 border-gray-200';
      default:
        return 'bg-gray-100 text-gray-600 border-gray-200';
    }
  }, []);

  const getButtonStyles = useMemo(() => (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-purple-600 hover:bg-purple-700 text-white';
      case 'inProgress':
        return 'bg-purple-600 hover:bg-purple-700 text-white';
      case 'available':
        return 'bg-purple-600 hover:bg-purple-700 text-white';
      case 'locked':
        return 'bg-gray-400 text-gray-600 cursor-not-allowed';
      default:
        return 'bg-gray-400 text-gray-600 cursor-not-allowed';
    }
  }, []);

  const getStatusLabel = useMemo(() => (status: string) => {
    switch (status) {
      case 'completed':
        return 'Completed';
      case 'inProgress':
        return 'In Progress';
      case 'available':
        return 'Available';
      case 'locked':
        return 'Locked';
      default:
        return 'Locked';
    }
  }, []);

  const getInitials = useMemo(() => (name: string) => {
    return name.split(' ').map(word => word[0]).join('').toUpperCase();
  }, []);

  // Enhanced loading placeholder with userProfile stability check
  if (!isStabilized || hasUserProfileChanged) {
    // Show a more detailed loading placeholder that indicates we're stabilizing data
    return (
      <div className="p-6 bg-white rounded-xl shadow-sm animate-pulse">
        <div className="h-6 bg-gray-200 rounded w-1/3 mb-4"></div>
        <div className="h-4 bg-gray-200 rounded w-2/3 mb-2"></div>
        <div className="h-4 bg-gray-200 rounded w-1/2 mb-4"></div>
        {hasUserProfileChanged && (
          <div className="flex items-center justify-center py-2 text-sm text-gray-500">
            <svg className="animate-spin h-4 w-4 mr-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Updating lesson data...
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="bg-[#f4f0ff]">
      {/* User Profile Section */}
      <div className="max-w-6xl mx-auto mb-8">
        <div className="bg-white rounded-xl p-6 shadow-sm border border-border-color">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="w-12 h-12 bg-purple-600 rounded-full flex items-center justify-center text-white font-semibold text-lg">
                {getInitials(userProfile.name)}
              </div>
              <div>
                <h2 className="text-xl font-semibold text-text-primary">{userProfile.name}</h2>
                <p className="text-text-secondary">German Language Progress: 40% Complete</p>
                <div className="mt-2 w-64 bg-gray-200 rounded-full h-2">
                  <div className="bg-purple-600 h-2 rounded-full" style={{ width: '40%' }}></div>
                </div>
              </div>
            </div>
            <div className="flex items-center space-x-3">
              <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm font-medium">
                A1 Basic Level Complete
              </span>
              <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm font-medium">
                5 Conversations with Maya
              </span>
              <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium">
                A2 Elementary Level
              </span>
              <span className="px-3 py-1 bg-orange-100 text-orange-800 rounded-full text-sm font-medium">
                7 Day Streak
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-6xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-3xl font-bold text-text-primary">German Language Learning</h1>
          <div className="flex items-center space-x-2 px-4 py-2 bg-purple-100 rounded-lg">
            <SparklesIcon className="w-5 h-5 text-purple-600" />
            <span className="text-purple-600 font-medium">Powered by Maya AI</span>
          </div>
        </div>

        {/* Lessons Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8 items-start">
          {lessons.map((lesson) => (
            <div
              key={lesson.id}
              className={`bg-white rounded-xl p-6 shadow-sm border transition-all duration-200 flex flex-col h-full ${
                lesson.status === 'locked' ? 'opacity-75' : 'hover:shadow-md'
              } ${getStatusStyles(lesson.status)}`}
            >
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-purple-600">{lesson.level}</h3>
                <span className={`px-3 py-1 rounded-full text-xs font-medium ${getStatusStyles(lesson.status)}`}>
                  {getStatusLabel(lesson.status)}
                </span>
              </div>
              
              <h4 className="text-xl font-semibold text-text-primary mb-3">{lesson.title}</h4>
              <p className="text-text-secondary text-sm mb-4 leading-relaxed flex-grow">{lesson.description}</p>
              
              <div className="mb-4 mt-auto">
                <div className="flex justify-between text-sm text-text-secondary mb-2">
                  <span>Progress</span>
                  <span>{lesson.totalExercises}</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className={`h-2 rounded-full ${
                      lesson.status === 'completed' ? 'bg-green-500' :
                      lesson.status === 'inProgress' ? 'bg-yellow-500' :
                      'bg-gray-300'
                    }`}
                    style={{ 
                      width: lesson.status === 'completed' ? '100%' :
                             lesson.status === 'inProgress' ? '70%' :
                             '0%'
                    }}
                  ></div>
                </div>
              </div>

              <button
                className={`w-full py-3 px-4 rounded-lg font-medium transition-colors ${getButtonStyles(lesson.status)}`}
                disabled={lesson.status === 'locked'}
                onClick={() => {
                  if (lesson.status !== 'locked') {
                    console.log(`Opening lesson: ${lesson.id}`);
                  }
                }}
              >
                {lesson.buttonText}
              </button>
            </div>
          ))}
        </div>

        {/* Practice with Maya Section */}
        <div className="bg-white rounded-xl p-6 shadow-sm border border-border-color">
          <div className="flex items-center space-x-4 mb-4">
            <div className="w-12 h-12 bg-purple-600 rounded-full flex items-center justify-center">
              <SparklesIcon className="w-6 h-6 text-white" />
            </div>
            <div>
              <h3 className="text-xl font-semibold text-text-primary">Practice with Maya</h3>
              <p className="text-text-secondary">AI-powered conversation practice in German</p>
            </div>
          </div>
          
          <div className="bg-purple-50 rounded-lg p-4 mb-4">
            <p className="text-purple-900 italic">
              <strong>Maya:</strong> Hallo Alex! Wie geht es dir heute? Let's practice some basic German conversation. 
              When you click "Continue Lesson", I'll speak only German with you for immersive learning!
            </p>
          </div>
          
          <ActionButton
            onClick={() => window.open('https://teaching.mayacode.io', '_blank')}
            variant="primary"
            className="bg-purple-600 hover:bg-purple-700"
          >
            Start Conversation with Maya
          </ActionButton>
        </div>
      </div>
    </div>
  );
});

export default LanguageLearning;