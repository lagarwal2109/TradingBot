
import React from 'react';

interface DocumentsPageProps {
  onNextStep: () => void;
  onPreviousStep: () => void;
  T: any;
}

const DocumentsPage: React.FC<DocumentsPageProps> = ({ T }) => {
  const stepName = T.workflowSteps?.documents || "Documents";
  // Example mock data for demonstration
  const documents = [
    { id: 1, name: 'Passport Scan', type: 'PDF', added: '2024-05-01', size: '1.2 MB' },
    { id: 2, name: 'Birth Certificate', type: 'Image', added: '2024-04-20', size: '500 KB' },
    { id: 3, name: 'Refugee Status Letter', type: 'DOCX', added: '2024-03-15', size: '200 KB' },
  ];

  return (
    <div>
      <h2 className="text-xl font-semibold text-text-primary mb-4">{stepName}</h2>
      <div className="space-y-3">
        {documents.map((doc) => (
          <div key={doc.id} className="bg-background p-3 sm:p-4 rounded-lg shadow-sm border border-border-color">
            <h4 className="text-sm font-semibold text-text-primary mb-0.5">{doc.name}</h4>
            <div className="text-xs text-text-secondary">
              Type: {doc.type} | Added: {doc.added} | Size: {doc.size}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default DocumentsPage;
