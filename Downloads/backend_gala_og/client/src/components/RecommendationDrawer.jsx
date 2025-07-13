import { useState, useEffect, useRef, forwardRef } from "react";

// Convert to forwardRef to accept ref from UI component
export const RecommendationDrawer = forwardRef(({ 
  isOpen = false, 
  setIsOpen = () => {}, 
  recommendation = "", 
  isLoading = false, 
  error = null,
}, ref) => {
  // Use component ref or create our own
  const localRef = useRef(null);
  const drawerRef = ref || localRef;

  // Define button style to match the purple theme
  const buttonStyle = "bg-purple-400 hover:bg-purple-500 text-white p-4 rounded-md flex items-center justify-center";
  
  // Utility to convert literal \n sequences into actual line breaks
  const normalizeContent = (raw) =>
    raw.replace(/\\n\\n/g, "\n\n").replace(/\\n/g, "\n");

  // Close drawer when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (drawerRef.current && !drawerRef.current.contains(event.target) && isOpen) {
        setIsOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [isOpen]);

  // Job Recommendation Card Component
  const JobRecommendationCard = ({ title, description, applyText }) => {
    // Extract company name (assuming format "Position - Company" or "Position at Company")
    const company = title.includes(" - ") 
      ? title.split(" - ")[1]
      : (title.includes(" at ") ? title.split(" at ")[1] : "");
    
    // Extract job position
    const position = title.includes(" - ")
      ? title.split(" - ")[0]
      : (title.includes(" at ") ? title.split(" at ")[0] : title);
    
    // Extract URL from apply text if available - Enhanced to handle various formats
    const extractUrl = () => {
      if (!applyText) return null;
      
      // Try markdown link format first: [text](url)
      const markdownMatch = applyText.match(/\[([^\]]+)\]\(([^)]+)\)/);
      if (markdownMatch) return markdownMatch[2];
      
      // Try direct URL detection
      const urlMatch = applyText.match(/(https?:\/\/[^\s]+)/);
      if (urlMatch) return urlMatch[1];
      
      // Try to find URL patterns even without http
      const domainMatch = applyText.match(/([a-zA-Z0-9-]+\.[a-zA-Z]{2,}[^\s]*)/);
      if (domainMatch && !domainMatch[1].includes('@')) {
        return `https://${domainMatch[1]}`;
      }
      
      return null;
    };
    
    const applyUrl = extractUrl();
    
    // Check if apply text contains job-related keywords
    const isJobRelated = applyText && (
      applyText.toLowerCase().includes('apply') ||
      applyText.toLowerCase().includes('job') ||
      applyText.toLowerCase().includes('career') ||
      applyText.toLowerCase().includes('position') ||
      applyUrl
    );
    
    // Process the description to create properly spaced paragraphs
    const formatDescriptionWithParagraphs = (text) => {
      if (!text) return null;
      
      // Split by double newlines to create paragraphs
      const paragraphs = text.split(/\n\n+/);
      
      return (
        <div className="space-y-3">
          {paragraphs.map((paragraph, idx) => {
            // Check if paragraph contains sections with bold headers
            if (paragraph.includes('**') && paragraph.includes(':')) {
              // Process sections with headers like "**Description**:"
              const sections = parseDescriptionSections(paragraph);
              return (
                <div key={idx} className="space-y-2">
                  {sections.map((section, sectionIdx) => (
                    <div key={`${idx}-${sectionIdx}`} className="mb-2">
                      {section.label && (
                        <div className="font-semibold text-gray-700 mb-1">{section.label}:</div>
                      )}
                      <div className="text-gray-600 ml-1">{processParagraph(section.content)}</div>
                    </div>
                  ))}
                </div>
              );
            } else {
              // Regular paragraph
              return (
                <div key={idx} className="text-gray-600">
                  {processParagraph(paragraph)}
                </div>
              );
            }
          })}
        </div>
      );
    };
    
    return (
      <div className="group relative bg-white rounded-xl shadow-lg border border-gray-100 overflow-hidden hover:shadow-xl hover:scale-[1.02] transition-all duration-300 mb-6">
        {/* Gradient top border */}
        <div className="h-1 bg-gradient-to-r from-purple-400 via-pink-400 to-purple-500"></div>
        
        {/* Card content */}
        <div className="p-6">
          {/* Header Section */}
          <div className="mb-4">
            {/* Title with enhanced styling */}
            <h3 className="font-bold text-xl text-gray-800 mb-2 group-hover:text-purple-700 transition-colors duration-200">
              {position}
            </h3>
            
            {/* Company badge */}
            {company && (
              <div className="inline-flex items-center px-3 py-1 bg-gradient-to-r from-purple-50 to-pink-50 text-purple-700 text-sm font-medium rounded-full border border-purple-100">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
                </svg>
                <span>{company}</span>
              </div>
            )}
          </div>
          
          {/* Job Description with enhanced styling */}
          <div className="mb-4 bg-gray-50 rounded-lg p-4 border-l-4 border-purple-200">
            {formatDescriptionWithParagraphs(description)}
          </div>
          
          {/* Contact Information Card */}
          {applyText && !description.includes(applyText) && (
            <div className="mb-4 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-4 border border-blue-100">
              <div className="flex items-center mb-2">
                <svg className="w-5 h-5 text-blue-600 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 4.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                </svg>
                <div className="font-semibold text-blue-700">Contact Information</div>
              </div>
              <div className="text-gray-700 ml-7">{processParagraph(applyText)}</div>
            </div>
          )}
          
          {/* Apply Button - Enhanced subtle design */}
          {(applyUrl || isJobRelated) && (
            <div className="mt-6 pt-4 border-t border-gray-200 flex justify-end">
              {applyUrl ? (
                <a 
                  href={applyUrl} 
                  target="_blank" 
                  rel="noopener noreferrer" 
                  className="group/btn inline-flex items-center px-6 py-3 text-sm font-semibold text-white bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 rounded-lg shadow-md hover:shadow-lg transform hover:scale-105 transition-all duration-200"
                >
                  <svg className="w-4 h-4 mr-2 group-hover/btn:animate-pulse" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                  </svg>
                  Apply Now
                </a>
              ) : (
                <div className="inline-flex items-center px-4 py-2 text-sm text-purple-600 bg-purple-50 rounded-lg border border-purple-200">
                  <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  See contact details above
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    );
  };
  
  // More flexible parser for job description sections
  const parseDescriptionSections = (description) => {
    if (!description) return [];
    
    const sections = [];
    const lines = description.split('\n');
    let currentSection = { label: null, content: '' };
    
    // For job sections that start with a hyphen and have a bold header
    const hyphenBoldPattern = /^-\s*\*\*([^*]+)\*\*:\s*(.+)/g;
    
    // For sections that just have a bold header without hyphen
    const boldHeaderPattern = /^\*\*([^:*]+)\*\*:\s*(.+)/g;
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line) continue; // Skip empty lines
      
      // Check if this is a line with a hyphen and a bold header
      const hyphenBoldMatch = line.match(hyphenBoldPattern);
      
      // Check if this is a line with just a bold header
      const boldHeaderMatch = !hyphenBoldMatch && line.match(boldHeaderPattern);
      
      if (hyphenBoldMatch) {
        // If we have content in the current section, save it
        if (currentSection.content) {
          sections.push({ ...currentSection });
        }
        
        // Start a new section with the label from the hyphen-bold pattern
        const label = hyphenBoldMatch[1].trim();
        const content = hyphenBoldMatch[2].trim();
        currentSection = { label, content };
      }
      else if (boldHeaderMatch) {
        // If we have content in the current section, save it
        if (currentSection.content) {
          sections.push({ ...currentSection });
        }
        
        // Start a new section with the label from the bold header pattern
        const label = boldHeaderMatch[1].trim();
        const content = boldHeaderMatch[2].trim();
        currentSection = { label, content };
      }
      // If it's just a bullet point (starts with "-") but not a section header
      else if (line.startsWith('-') && !line.includes('**:')) {
        // Add as part of the current section if we have one
        if (currentSection.label) {
          currentSection.content += (currentSection.content ? ' ' : '') + line.substring(1).trim();
        } 
        // Otherwise start a new unlabeled section
        else {
          if (currentSection.content) {
            sections.push({ ...currentSection });
          }
          currentSection = { label: null, content: line };
        }
      }
      // Regular content - add to current section
      else {
        if (currentSection.content) {
          currentSection.content += ' ' + line;
        } else {
          currentSection.content = line;
        }
      }
    }
    
    // Add the last section if it has content
    if (currentSection.content) {
      sections.push(currentSection);
    }
    
    // If no sections were found, add the entire description as one section
    if (sections.length === 0 && description) {
      sections.push({ label: "Description", content: description });
    }
    
    return sections;
  };

  // Simplified flexible parser that focuses on working cards
  const parseStructuredContent = (content) => {
    if (!content || typeof content !== 'string') return [];

    const normalizedContent = normalizeContent(content);
    const lines = normalizedContent.split('\n').filter(line => line.trim() !== '');
    const sections = [];
    let currentItem = null;

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line) continue;

      // Detect section headers
      if (line.match(/^\*\*[^*]+\*\*:?\s*$/)) {
        sections.push({
          type: 'header',
          content: line.replace(/^\*\*/, '').replace(/\*\*:?\s*$/, '')
        });
        continue;
      }

      // Detect numbered items (job cards)
      const numberedMatch = line.match(/^(\*\*)?(\d+[\.\)]\s*.+?)(\*\*)?$/) || 
                           line.match(/^\*\*([^*]+)\*\*$/);
      
      if (numberedMatch) {
        // Save previous item
        if (currentItem) {
          sections.push(currentItem);
        }

        let title = '';
        if (numberedMatch[2]) {
          // Numbered item - remove number prefix
          title = numberedMatch[2].replace(/^\d+[\.\)]\s*/, '');
        } else {
          // Bold item
          title = numberedMatch[1] || numberedMatch[0];
        }

        currentItem = {
          type: 'job_card',
          title: title.replace(/^\*\*/, '').replace(/\*\*$/, ''),
          fields: {},
          bullets: []
        };
        continue;
      }

      // Detect field-value pairs with enhanced patterns
      const fieldMatch = line.match(/^[-•*]\s*\*\*([^*]+)\*\*:\s*(.*)$/) || 
                        line.match(/^[-•*]\s*([A-Za-z][A-Za-z\s]+):\s*(.*)$/) ||
                        line.match(/^\s*\*\*([^*]+)\*\*:\s*(.*)$/) ||
                        line.match(/^\s*([A-Za-z][A-Za-z\s]+):\s*(.+)$/);
      
      if (fieldMatch && currentItem) {
        let fieldName = fieldMatch[1].trim().toLowerCase();
        const fieldValue = fieldMatch[2] ? fieldMatch[2].trim() : '';
        
        // Normalize common field variations
        if (fieldName.includes('experience') || fieldName.includes('exp')) {
          fieldName = 'experience';
        } else if (fieldName.includes('position') || fieldName.includes('role') || fieldName.includes('title')) {
          fieldName = 'position';
        } else if (fieldName.includes('location') || fieldName.includes('loc')) {
          fieldName = 'location';
        } else if (fieldName.includes('salary') || fieldName.includes('pay') || fieldName.includes('compensation')) {
          fieldName = 'salary';
        } else if (fieldName.includes('skill') || fieldName.includes('requirement')) {
          fieldName = 'skills';
        } else if (fieldName.includes('benefit') || fieldName.includes('perk')) {
          fieldName = 'benefits';
        } else if (fieldName.includes('contact') || fieldName.includes('website') || 
                   fieldName.includes('link') || fieldName.includes('apply') || 
                   fieldName.includes('url')) {
          fieldName = 'contact_info';
        }
        
        currentItem.fields[fieldName] = fieldValue;
        continue;
      }

      // Regular bullet points
      if (line.match(/^[-•*]\s+/) && currentItem) {
        const bulletContent = line.replace(/^[-•*]\s+/, '');
        currentItem.bullets.push(bulletContent);
        continue;
      }

      // Continuation content
      if (currentItem && !line.match(/^[-•*]/)) {
        // Add to last field or as bullet
        const fieldNames = Object.keys(currentItem.fields);
        if (fieldNames.length > 0) {
          const lastField = fieldNames[fieldNames.length - 1];
          currentItem.fields[lastField] += ' ' + line;
        } else {
          currentItem.bullets.push(line);
        }
      } else {
        // Standalone content
        sections.push({
          type: 'paragraph',
          content: line
        });
      }
    }

    // Add the last item
    if (currentItem) {
      sections.push(currentItem);
    }

    return sections;
  };

  // Convert text to JSX with paragraphs and line breaks - Enhanced version with job cards
  const formatContent = (content) => {
    if (!content) return null;

    try {
      const sections = parseStructuredContent(content);
      
      // Debug logging
      console.log('Parsed sections:', sections);
      
      if (sections.length === 0) {
        // Fallback to simple rendering
        return (
          <div className="space-y-4">
            <div className="text-gray-700">{processParagraph(content)}</div>
          </div>
        );
      }

      return (
        <div className="space-y-6">
          {sections.map((section, idx) => {
            console.log(`Rendering section ${idx}:`, section.type, section);
            if (section.type === 'job_card') {
              // Create a clean, properly formatted description
              const descriptionParts = [];
              let contactInfo = '';
              
              // Build description from all non-contact fields with clean formatting
              Object.entries(section.fields).forEach(([fieldName, fieldValue]) => {
                if (fieldValue) {
                  const isContactField = fieldName === 'contact_info' || 
                                       fieldName.includes('contact') || 
                                       fieldName.includes('website') || 
                                       fieldName.includes('link') ||
                                       fieldName.includes('apply');
                  
                  if (isContactField) {
                    contactInfo = fieldValue;
                  } else {
                    // Add field with clean formatting
                    const displayName = fieldName.charAt(0).toUpperCase() + fieldName.slice(1).replace(/[_-]/g, ' ');
                    descriptionParts.push(`${displayName}: ${fieldValue}`);
                  }
                }
              });
              
              // Add bullets to description with clean formatting
              if (section.bullets.length > 0) {
                descriptionParts.push(''); // Add spacing before bullets
                descriptionParts.push(...section.bullets.map(bullet => `• ${bullet}`));
              }
              
              const cleanDescription = descriptionParts.join('\n\n'); // Use double line breaks for spacing
              
              return (
                <JobRecommendationCard 
                  key={idx} 
                  title={section.title} 
                  description={cleanDescription} 
                  applyText={contactInfo}
                />
              );
            }
            else if (section.type === 'header') {
              return (
                <div key={idx} className="border-b-2 border-purple-200 pb-2 mb-4">
                  <h3 className="text-xl font-bold text-purple-700">{section.content}</h3>
                </div>
              );
            } else if (section.type === 'paragraph') {
              return (
                <div key={idx} className="text-gray-700">
                  {processParagraph(section.content)}
                </div>
              );
            }
            return null;
          })}
        </div>
      );
    } catch (error) {
      console.error('Error formatting content:', error);
      // Fallback to simple rendering if parsing fails
      return (
        <div className="space-y-4">
          <div className="text-gray-700">{processParagraph(content)}</div>
        </div>
      );
    }
  };

  // Process bold and links
  const processParagraph = (text) => {
    if (!text) return null;
    
    // First clean up the text by replacing double asterisks used for emphasis
    // This needs to happen before we split into segments to avoid breaking hyperlinks
    let cleanedText = text;
    
    // Replace **Section**: with styled bold section headers but preserve the content after it
    cleanedText = cleanedText.replace(/\*\*([^*:]+)\*\*:\s*(.+)/g, (match, p1, p2) => {
      return `<section-header>${p1}</section-header>${p2}`;
    });
    
    // Replace remaining **bold text** with styled bold
    cleanedText = cleanedText.replace(/\*\*([^*]+)\*\*/g, '<bold>$1</bold>');
    
    // Now process links and our custom markers
    const segments = [];
    let remaining = cleanedText;
    
    // Regex patterns for different formatting elements
    const patterns = [
      { type: 'link', regex: /\[([^\]]+)\]\(([^)]+)\)/ },
      { type: 'section-header', regex: /<section-header>([^<]+)<\/section-header>/ },
      { type: 'bold', regex: /<bold>([^<]+)<\/bold>/ }
    ];
    
    while (remaining) {
      // Find the earliest match of any pattern
      let earliestMatch = null;
      let earliestType = null;
      let earliestIndex = Infinity;
      
      for (const pattern of patterns) {
        const match = remaining.match(pattern.regex);
        if (match && match.index < earliestIndex) {
          earliestMatch = match;
          earliestType = pattern.type;
          earliestIndex = match.index;
        }
      }
      
      if (!earliestMatch) {
        // No more matches, add the remaining text and break
        segments.push({ type: 'text', content: remaining });
        break;
      }
      
      // Add text before the match
      if (earliestIndex > 0) {
        segments.push({ type: 'text', content: remaining.slice(0, earliestIndex) });
      }
      
      // Add the matched element
      if (earliestType === 'link') {
        segments.push({ 
          type: 'link', 
          text: earliestMatch[1], 
          url: earliestMatch[2]
        });
      } else if (earliestType === 'section-header') {
        segments.push({ 
          type: 'section-header', 
          content: earliestMatch[1]
        });
      } else if (earliestType === 'bold') {
        segments.push({ 
          type: 'bold', 
          content: earliestMatch[1]
        });
      }
      
      // Update remaining text
      remaining = remaining.slice(earliestIndex + earliestMatch[0].length);
    }
    
    // Convert segments to React elements
    return segments.map((seg, idx) => {
      if (seg.type === 'text') return seg.content;
      if (seg.type === 'link') {
        return (
          <a key={idx} href={seg.url} target="_blank" rel="noopener noreferrer" className="text-pink-600 hover:text-pink-800 underline">
            {seg.text}
          </a>
        );
      }
      if (seg.type === 'bold') return (
        <strong key={idx} className="font-bold">{seg.content}</strong>
      );
      if (seg.type === 'section-header') return (
        <span key={idx} className="font-semibold text-purple-700">{seg.content}</span>
      );
      return null;
    });
  };

  return (
    <>
      {/* Drawer opens from the left */}
      <div
        className={`fixed inset-y-0 left-0 w-full sm:w-96 bg-white shadow-xl transform transition-transform duration-300 ease-in-out z-50 pointer-events-auto ${isOpen ? 'translate-x-0' : '-translate-x-full'}`}
        ref={drawerRef}
      >
        <div className="flex justify-between items-center px-6 py-4 border-b">
          <h2 className="text-xl font-bold">Recommendations</h2>
          <button onClick={() => setIsOpen(false)} className="text-gray-500 hover:text-gray-700">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        <div className="p-6 overflow-y-auto h-[calc(100%-8rem)]">
          {error ? (
            <div className="text-red-500 p-4 bg-red-50 rounded-md">
              <p className="font-medium">Error</p><p>{error}</p>
            </div>
          ) : isLoading ? (
            <div className="flex flex-col items-center justify-center h-full">
              {/* Elegant loading animation */}
              <div className="relative mb-8">
                {/* Document stack animation */}
                <div className="w-20 h-24 bg-purple-100 rounded-md shadow-sm absolute top-4 right-2 transform rotate-6 animate-pulse"></div>
                <div className="w-20 h-24 bg-purple-200 rounded-md shadow-sm absolute top-2 right-1 transform rotate-3 animate-pulse animation-delay-150"></div>
                <div className="w-20 h-24 bg-white border-2 border-purple-300 rounded-md shadow-md relative z-10">
                  <div className="absolute top-3 left-2 right-2 h-2 bg-purple-200 rounded-sm"></div>
                  <div className="absolute top-7 left-2 right-2 h-2 bg-purple-200 rounded-sm"></div>
                  <div className="absolute top-11 left-2 right-6 h-2 bg-purple-200 rounded-sm"></div>
                  <div className="absolute top-15 left-2 right-4 h-2 bg-purple-200 rounded-sm"></div>
                </div>
                
                {/* Sparkle animation */}
                <div className="absolute -top-2 -right-2 w-4 h-4 bg-purple-300 rounded-full animate-ping opacity-75"></div>
                <div className="absolute bottom-0 -right-1 w-3 h-3 bg-purple-400 rounded-full animate-ping opacity-75 animation-delay-300"></div>
                <div className="absolute top-1/2 -left-2 w-3 h-3 bg-purple-200 rounded-full animate-ping opacity-75 animation-delay-700"></div>
                
                {/* Search magnifying glass */}
                <div className="absolute -bottom-3 -left-3 w-10 h-10 bg-purple-400 rounded-full flex items-center justify-center transform -rotate-12 animate-bounce">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                </div>
              </div>
              <div className="text-lg font-medium text-purple-700 mb-2">Finding Best Matches</div>
              <div className="text-sm text-purple-500 flex items-center">
                <span>Analyzing your profile</span>
                <span className="ml-2 flex">
                  <span className="animate-bounce mx-px delay-0">.</span>
                  <span className="animate-bounce mx-px delay-150">.</span>
                  <span className="animate-bounce mx-px delay-300">.</span>
                </span>
              </div>
            </div>
          ) : (
            <div className="recommendation-content">{formatContent(recommendation)}</div>
          )}
        </div>
        <div className="border-t p-4 flex justify-end">
          <button onClick={() => setIsOpen(false)} className="bg-purple-400 hover:bg-purple-500 text-white px-4 py-2 rounded transition-colors">
            Close
          </button>
        </div>
      </div>
      {isOpen && <div className="fixed inset-0 bg-black bg-opacity-50 z-40 pointer-events-auto" onClick={() => setIsOpen(false)} />}
    </>
  );
});

export default RecommendationDrawer;
