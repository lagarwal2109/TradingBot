import React from 'react';
import { AiChatMessage } from '../types';

interface ChatHistorySectionProps {
  T?: any;
  onChatHistoryClick?: (conversation: AiChatMessage[]) => void;
}

const mockConversations: Record<string, AiChatMessage[]> = {
  'Asylum Registration Help': [
    { id: '1', role: 'model', text: "Hi! I'm Maya, your MayaCode dashboard assistant. I can help you navigate your profile, workflows, documents, and job applications. What would you like to explore?" },
    { id: '2', role: 'user', text: 'I need help with my asylum registration. What documents do I need?' },
    { id: '3', role: 'model', text: 'For asylum registration, you\'ll typically need: 1) A valid passport or ID document, 2) Your asylum application form, 3) Supporting evidence for your claim, 4) Medical records if applicable. Check your Documents section for templates and guidance.' },
    { id: '4', role: 'user', text: 'Where can I find the asylum application form?' },
    { id: '5', role: 'model', text: 'You can find the asylum application form in your Documents section. Look for "BAMF Forms" or check your suggested actions - there should be a task for "Complete BAMF Asylum Interview Form" that will guide you through the process.' },
    { id: '6', role: 'user', text: 'What happens after I submit my application?' },
    { id: '7', role: 'model', text: 'After submission, BAMF will schedule an interview with you. You\'ll receive a notification with the date and location. I see you have an "Attend BAMF Asylum Interview" task in your dashboard that will help you prepare for this step.' }
  ],
  'Software Developer Jobs': [
    { id: '1', role: 'model', text: "Hi! I'm Maya, your MayaCode dashboard assistant. I can help you navigate your profile, workflows, documents, and job applications. What would you like to explore?" },
    { id: '2', role: 'user', text: 'I\'m looking for software developer positions. Can you help me?' },
    { id: '3', role: 'model', text: 'Absolutely! I can help you with software developer job opportunities. First, let\'s make sure your profile is optimized. Do you have your technical skills and programming languages listed in your profile?' },
    { id: '4', role: 'user', text: 'I have experience with React, Node.js, and Python. How do I add these to my profile?' },
    { id: '5', role: 'model', text: 'Great skills! Go to your Profile section and look for the "Skills" or "Technical Expertise" area. Add React, Node.js, and Python there. Also, make sure to upload an updated resume highlighting these technologies.' },
    { id: '6', role: 'user', text: 'Should I include my international work experience?' },
    { id: '7', role: 'model', text: 'Yes, definitely! International experience is valuable. Include all relevant work experience, and consider getting your credentials evaluated if you have formal education from outside Germany. Check the "Prepare for Job Interview" task in your dashboard for more tips.' }
  ],
  'German Language Courses': [
    { id: '1', role: 'model', text: "Hi! I'm Maya, your MayaCode dashboard assistant. I can help you navigate your profile, workflows, documents, and job applications. What would you like to explore?" },
    { id: '2', role: 'user', text: 'I want to improve my German. What language courses are available?' },
    { id: '3', role: 'model', text: 'Great choice! I see you have access to our Language Learning section. Click on "Language Learning" in your sidebar to see available German courses from A1.1 to A2.3 levels.' },
    { id: '4', role: 'user', text: 'What level should I start with if I\'m a complete beginner?' },
    { id: '5', role: 'model', text: 'As a complete beginner, start with A1.1. This covers basic greetings, introductions, and essential vocabulary. Each level builds on the previous one, so it\'s important to progress systematically.' },
    { id: '6', role: 'user', text: 'How long does each level take to complete?' },
    { id: '7', role: 'model', text: 'Typically, each level takes 4-6 weeks with regular practice. The courses are self-paced, so you can learn at your own speed. Don\'t forget to use the "Practice with Maya" feature for additional conversational practice!' }
  ],
  'Health Insurance Setup': [
    { id: '1', role: 'model', text: "Hi! I'm Maya, your MayaCode dashboard assistant. I can help you navigate your profile, workflows, documents, and job applications. What would you like to explore?" },
    { id: '2', role: 'user', text: 'I need to set up health insurance in Germany. Where do I start?' },
    { id: '3', role: 'model', text: 'Health insurance is mandatory in Germany. You\'ll need to choose between public (gesetzliche) or private (private) insurance. For most newcomers, public insurance through providers like AOK, TK, or Barmer is recommended.' },
    { id: '4', role: 'user', text: 'What documents do I need to apply?' },
    { id: '5', role: 'model', text: 'You\'ll need: 1) Your passport/ID, 2) Residence permit, 3) Employment contract (if working), 4) Bank account details, 5) Previous insurance certificate (if applicable). Check your Documents section for application templates.' },
    { id: '6', role: 'user', text: 'How much does it cost?' },
    { id: '7', role: 'model', text: 'Public health insurance costs about 14.6% of your gross income (split between you and employer if employed). The minimum is around €160/month for non-working individuals. Contact your chosen provider for exact rates.' }
  ],
  'Housing & WBS Application': [
    { id: '1', role: 'model', text: "Hi! I'm Maya, your MayaCode dashboard assistant. I can help you navigate your profile, workflows, documents, and job applications. What would you like to explore?" },
    { id: '2', role: 'user', text: 'I need help finding housing and applying for WBS. What is WBS?' },
    { id: '3', role: 'model', text: 'WBS (Wohnberechtigungsschein) is a housing eligibility certificate that allows you to rent subsidized apartments in Germany. It\'s available if your income is below certain thresholds.' },
    { id: '4', role: 'user', text: 'How do I apply for WBS?' },
    { id: '5', role: 'model', text: 'Apply at your local housing authority (Wohnungsamt). You\'ll need: income proof, residence permit, family composition details, and bank statements. The process takes 2-4 weeks typically.' },
    { id: '6', role: 'user', text: 'What are good websites for finding apartments?' },
    { id: '7', role: 'model', text: 'Popular housing websites include: ImmobilienScout24, WG-Gesucht (for shared housing), Immowelt, and eBay Kleinanzeigen. Also check local housing cooperatives and municipal housing companies in your city.' }
  ],
  'Document Translation': [
    { id: '1', role: 'model', text: "Hi! I'm Maya, your MayaCode dashboard assistant. I can help you navigate your profile, workflows, documents, and job applications. What would you like to explore?" },
    { id: '2', role: 'user', text: 'I need to translate my documents for various applications. What\'s the best approach?' },
    { id: '3', role: 'model', text: 'For official purposes, you\'ll need certified translations (beglaubigte Übersetzungen). These can only be done by sworn translators (vereidigte Übersetzer) recognized by German courts.' },
    { id: '4', role: 'user', text: 'Which documents typically need certified translation?' },
    { id: '5', role: 'model', text: 'Common documents: birth certificates, marriage certificates, diplomas, transcripts, employment certificates, and medical records. Check with the specific authority requiring the documents for their exact requirements.' },
    { id: '6', role: 'user', text: 'How much do certified translations cost?' },
    { id: '7', role: 'model', text: 'Certified translations typically cost €1.50-2.50 per line, with a minimum fee of around €30-50 per document. Complex documents or urgent requests may cost more. Get quotes from multiple certified translators.' }
  ],
  'Integration Services': [
    { id: '1', role: 'model', text: "Hi! I'm Maya, your MayaCode dashboard assistant. I can help you navigate your profile, workflows, documents, and job applications. What would you like to explore?" },
    { id: '2', role: 'user', text: 'What integration services are available to help me settle in Germany?' },
    { id: '3', role: 'model', text: 'Germany offers comprehensive integration programs including: Integration courses (language + orientation), job counseling through Arbeitsagentur, social counseling, and recognition of foreign qualifications.' },
    { id: '4', role: 'user', text: 'How do I register for an integration course?' },
    { id: '5', role: 'model', text: 'Contact your local BAMF office or Ausländerbehörde to apply. You\'ll receive a participation entitlement (Berechtigung) or obligation (Verpflichtung). The course includes 600 hours of German language and 100 hours of orientation about German society.' },
    { id: '6', role: 'user', text: 'Are there support services for finding employment?' },
    { id: '7', role: 'model', text: 'Yes! The Arbeitsagentur (Job Center) provides job placement services, career counseling, and skills assessment. There are also specialized programs for migrants and refugees through organizations like IQ Network.' }
  ],
  'Legal Rights & Advice': [
    { id: '1', role: 'model', text: "Hi! I'm Maya, your MayaCode dashboard assistant. I can help you navigate your profile, workflows, documents, and job applications. What would you like to explore?" },
    { id: '2', role: 'user', text: 'I have questions about my legal rights in Germany. Where can I get advice?' },
    { id: '3', role: 'model', text: 'For legal advice, you can contact: Legal aid services (Rechtshilfe), refugee counseling centers, lawyers specializing in migration law, or legal clinics at universities. Many offer free initial consultations.' },
    { id: '4', role: 'user', text: 'What are my basic rights as an asylum seeker?' },
    { id: '5', role: 'model', text: 'Your rights include: access to accommodation and basic needs, medical care, legal representation during asylum procedures, education for children, and protection from deportation to countries where you face persecution.' },
    { id: '6', role: 'user', text: 'Can I work while my asylum application is being processed?' },
    { id: '7', role: 'model', text: 'Generally, you can work after 3 months if you have a residence permit (Aufenthaltsgestattung) and get approval from the Ausländerbehörde and Arbeitsagentur. Some restrictions may apply depending on your specific situation.' }
  ]
};

const chatHistoryData = [
  {
    date: 'Today',
    items: [
      'Asylum Registration Help',
      'Software Developer Jobs',
    ]
  },
  {
    date: 'Yesterday',
    items: [
      'German Language Courses',
      'Health Insurance Setup',
    ]
  },
  {
    date: 'Previous 7 Days',
    items: [
      'Housing & WBS Application',
      'Document Translation',
      'Integration Services',
      'Legal Rights & Advice',
    ]
  }
];

const ChatHistorySection: React.FC<ChatHistorySectionProps> = ({ onChatHistoryClick }) => {
  const handleChatClick = (chatTitle: string) => {
    const conversation = mockConversations[chatTitle];
    if (conversation && onChatHistoryClick) {
      onChatHistoryClick(conversation);
    }
  };

  return (
    <div className="px-1 pt-3 mt-2 border-t border-border-color max-h-80 overflow-y-auto">
      <div className="flex justify-between items-center mb-2 pt-1">
        <h3 className="text-xs font-semibold text-text-secondary uppercase tracking-wider">Chat History</h3>
        <button type="button" className="font-medium rounded-md focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-background transition-all duration-150 ease-in-out inline-flex items-center px-3 py-1.5 text-sm bg-border-color text-text-primary hover:bg-card-header focus:ring-border-color justify-center px-2 py-0.5 text-xs !min-h-0 !h-auto !leading-none">
          <span className="inline-flex items-center ">+ New</span>
        </button>
      </div>
      {chatHistoryData.map(section => (
        <React.Fragment key={section.date}>
          <p className="text-[11px] text-text-secondary mb-1">{section.date}</p>
          <ul className="space-y-0.5 mb-3">
            {section.items.map(item => (
              <li key={item}>
                <button 
                  onClick={() => handleChatClick(item)}
                  className="w-full text-left text-sm text-text-primary hover:text-accent hover:bg-accent/10 rounded-md p-1.5 cursor-pointer truncate focus:outline-none focus:ring-1 focus:ring-accent"
                >
                  {item}
                </button>
              </li>
            ))}
          </ul>
        </React.Fragment>
      ))}
    </div>
  );
};

export default ChatHistorySection;
