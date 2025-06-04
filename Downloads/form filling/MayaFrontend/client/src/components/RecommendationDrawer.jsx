import { useState, useEffect, useRef, forwardRef } from "react";
const API_ENDPOINT = "https://test.mayacode.io/api/api"

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
    
    // Extract URL from apply text if available
    // Example: "For more details and to apply, visit: [Apply Here](https://example.com/job)"
    const extractUrl = () => {
      if (!applyText) return null;
      const urlMatch = applyText.match(/\[([^\]]+)\]\(([^)]+)\)/);
      return urlMatch ? urlMatch[2] : null;
    };
    
    const applyUrl = extractUrl();
    
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
      <div className="bg-white rounded-lg shadow-md border-l-4 border-l-purple-400 overflow-hidden hover:shadow-lg transition-all mb-5">
        <div className="p-5">
          {/* Title and Company */}
          <h3 className="font-bold text-lg text-gray-800 mb-2">{position}</h3>
          {company && (
            <div className="flex items-center text-sm text-purple-600 mb-4">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
              </svg>
              <span className="font-medium">{company}</span>
            </div>
          )}
          
          {/* Job Description with improved paragraph spacing */}
          <div className="mb-4">
            {formatDescriptionWithParagraphs(description)}
          </div>
          
          {/* Add the apply text if it's not already included in description */}
          {applyText && !description.includes(applyText) && (
            <div className="mb-3 mt-3 pt-2 border-t border-gray-100">
              <div className="font-semibold text-gray-700 mb-1.5">Contact Information:</div>
              <div className="text-gray-600 ml-1">{processParagraph(applyText)}</div>
            </div>
          )}
          
          {/* Apply Button - Only show if we have a URL */}
          {applyUrl && (
            <div className="mt-4 flex justify-end">
              <a 
                href={applyUrl} 
                target="_blank" 
                rel="noopener noreferrer" 
                className="bg-purple-400 hover:bg-purple-500 text-white py-1.5 px-4 rounded text-sm font-medium transition-colors inline-block"
              >
                Apply Now
              </a>
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
    const hyphenBoldPattern = /^-\s*\*\*([^:*]+)\*\*:\s*(.+)$/;
    
    // For sections that just have a bold header without hyphen
    const boldHeaderPattern = /^\*\*([^:*]+)\*\*:\s*(.+)$/;
    
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

  // Convert text to JSX with paragraphs and line breaks - Enhanced version with job cards
  const formatContent = (content) => {
    if (!content) return null;
    const normalized = normalizeContent(content);
    
    // Determine if content has a standard format with job listings marked by numbered headings or asterisks
    const hasNumberedJobs = /\*\*\d+\.\s+[\w\s]+/.test(normalized) || /^\d+\.\s+[\w\s]+/m.test(normalized);
    
    // If we have numbered job listings, parse them in a structured way
    if (hasNumberedJobs) {
      return parseNumberedJobListings(normalized);
    }
    
    // Otherwise use a more flexible approach for unstructured text
    return parseUnstructuredContent(normalized);
  };
  
  // Parse content where job listings are clearly marked with numbered headings
  const parseNumberedJobListings = (content) => {
    // Let's use a different approach to parse job listings more reliably
    // This will handle multiple jobs with headers like "**1. Job Title**"
    
    // Split the content by job headers (numbered with asterisks)
    const jobHeaderRegex = /\*\*\d+\.\s+[^*\n]+\*\*/g;
    
    // Find all job headers
    const jobHeaders = content.match(jobHeaderRegex) || [];
    
    // If no job headers found, treat as regular content
    if (jobHeaders.length === 0) {
      return (
        <div className="space-y-4">
          <div className="text-gray-700">{processParagraph(content)}</div>
        </div>
      );
    }
    
    // Otherwise split content into introduction, jobs, and conclusion
    
    // Get introduction - text before the first job header
    const firstHeaderIndex = content.indexOf(jobHeaders[0]);
    const introBlock = firstHeaderIndex > 0 ? content.substring(0, firstHeaderIndex).trim() : "";
    
    // Get conclusion - text after the last job
    let conclusionBlock = "";
    const lastHeaderIndex = content.lastIndexOf(jobHeaders[jobHeaders.length - 1]);
    if (lastHeaderIndex >= 0) {
      // Find the next job header after this one
      const nextJobHeaderIndex = content.indexOf("**", lastHeaderIndex + jobHeaders[jobHeaders.length - 1].length);
      
      // If there's text after all jobs but no more job headers, it's a conclusion
      if (nextJobHeaderIndex === -1) {
        // Find end of the last job content
        const lastJobSectionEndIndex = findLastJobEnd(content, lastHeaderIndex);
        if (lastJobSectionEndIndex < content.length - 1) {
          conclusionBlock = content.substring(lastJobSectionEndIndex).trim();
        }
      }
    }
    
    // Extract each job's content
    const jobBlocks = [];
    
    for (let i = 0; i < jobHeaders.length; i++) {
      const currentHeader = jobHeaders[i];
      const headerIndex = content.indexOf(currentHeader);
      
      let jobEndIndex;
      if (i < jobHeaders.length - 1) {
        // If not the last job, the job ends where the next job begins
        jobEndIndex = content.indexOf(jobHeaders[i + 1]);
      } else {
        // For the last job, find where it ends
        jobEndIndex = findLastJobEnd(content, headerIndex);
      }
      
      // Get the job content including its header
      const jobContent = content.substring(headerIndex, jobEndIndex).trim();
      
      // Process the job content to extract title and details
      const title = currentHeader.replace(/^\*\*|\*\*$/g, '').trim();
      const descriptionContent = jobContent.substring(currentHeader.length).trim();
      
      // Split the description to find contact info (usually the last part with a link)
      let description = descriptionContent;
      let applyText = null;
      
      // Look for contact info or a URL in the description
      const lines = descriptionContent.split('\n');
      for (let j = 0; j < lines.length; j++) {
        if (
          lines[j].includes('**Contact**') || 
          lines[j].toLowerCase().includes('contact information') ||
          (lines[j].includes('[') && lines[j].includes(']') && lines[j].includes('('))
        ) {
          applyText = lines[j];
          break;
        }
      }
      
      jobBlocks.push({ title, description, applyText });
    }
    
    return (
      <div className="space-y-6">
        {/* Introduction */}
        {introBlock && <p className="text-gray-700 mb-4">{processParagraph(introBlock)}</p>}
        
        {/* Job Cards */}
        <div className="space-y-6">
          {jobBlocks.map((job, index) => (
            <JobRecommendationCard 
              key={`job-${index}`}
              title={job.title}
              description={job.description}
              applyText={job.applyText}
            />
          ))}
        </div>
        
        {/* Conclusion */}
        {conclusionBlock && (
          <p className="text-gray-700 mt-4">{processParagraph(conclusionBlock)}</p>
        )}
      </div>
    );
  };
  
  // Helper function to find where a job section ends
  const findLastJobEnd = (content, startIndex) => {
    // Find the next job header after this position
    const nextHeaderIndex = content.indexOf("**", startIndex + 1);
    if (nextHeaderIndex === -1) {
      // If no more headers, job ends at the end of content
      return content.length;
    }
    
    // Check if this is actually a job header or just bold text
    const potentialHeader = content.substring(nextHeaderIndex, content.indexOf("\n", nextHeaderIndex) > -1 ? 
                                            content.indexOf("\n", nextHeaderIndex) : content.length);
    
    if (potentialHeader.match(/\*\*\d+\.\s+[^*\n]+\*\*/)) {
      // It's a job header, so the previous job ends here
      return nextHeaderIndex;
    } else {
      // It's just bold text, keep looking
      return findLastJobEnd(content, nextHeaderIndex + 2);
    }
  };
  
  // Parse unstructured content as a fallback
  const parseUnstructuredContent = (content) => {
    // Just render the entire content as a single block if we can't detect structure
    return (
      <div className="space-y-4">
        <div className="text-gray-700">{processParagraph(content)}</div>
      </div>
    );
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
      if (seg.type === 'link') return (
        <a key={idx} href={seg.url} target="_blank" rel="noopener noreferrer" className="text-pink-600 hover:text-pink-800 underline">
          {seg.text}
        </a>
      );
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
