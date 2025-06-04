import React from 'react';

interface ChatHistorySectionProps {
  T?: any;
}

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

const ChatHistorySection: React.FC<ChatHistorySectionProps> = () => {
  return (
    <div className="px-1 pt-3 mt-2 border-t border-border-color max-h-64 overflow-y-auto">
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
                <button className="w-full text-left text-sm text-text-primary hover:text-accent hover:bg-accent/10 rounded-md p-1.5 cursor-pointer truncate focus:outline-none focus:ring-1 focus:ring-accent">
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
