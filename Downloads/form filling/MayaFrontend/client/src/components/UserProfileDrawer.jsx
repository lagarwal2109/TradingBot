import { useState, useEffect, useRef } from "react";
import { LanguageSelector } from "./LanguageSelector";

export const UserProfileDrawer = ({ currentLanguage, setCurrentLanguage }) => {
  const [userPersona, setUserPersona] = useState(null);
  const [editablePersona, setEditablePersona] = useState(null);
  const [isEditing, setIsEditing] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [saveStatus, setSaveStatus] = useState(null); // null | 'saving' | 'saved' | 'error' | 'auto-saving' | 'auto-save-error'
  const [locationDenied, setLocationDenied] = useState(false);
  const [locationStatus, setLocationStatus] = useState('idle'); // Keep internal status for logic
  
  // Add a ref to track if a location update is pending
  const pendingLocationUpdateRef = useRef(false);
  const chatEndTimeoutRef = useRef(null);
  
  // Add useEffect to listen for chat-ended event
  useEffect(() => {
    // Define the event handler function
    const handleChatEnded = (event) => {
      // Clear any existing timeout to prevent multiple calls
      if (chatEndTimeoutRef.current) {
        clearTimeout(chatEndTimeoutRef.current);
      }
      
      // Set the pending flag to true
      pendingLocationUpdateRef.current = true;
      
      // Wait 10 seconds before refreshing location data
      chatEndTimeoutRef.current = setTimeout(() => {
        if (pendingLocationUpdateRef.current) {
          fetchUserPersonaAndLocation();
          pendingLocationUpdateRef.current = false;
        }
      }, 10000);
    };
    
    // Add event listener
    window.addEventListener('chat-ended', handleChatEnded);
    
    // Call fetchUserPersonaAndLocation on component mount
    fetchUserPersonaAndLocation();
    
    // Clean up event listener on component unmount
    return () => {
      window.removeEventListener('chat-ended', handleChatEnded);
      if (chatEndTimeoutRef.current) {
        clearTimeout(chatEndTimeoutRef.current);
      }
    };
  }, []);  // Empty dependency array means this runs once on mount

  const API_ENDPOINT = "https://test.mayacode.io/api/api";

  // --- Refactored Backend Update Function ---
  const updatePersonaOnBackend = (personaData, isAutoSave = false) => {
    // Use specific status for auto-saves if needed, or just reuse 'saving'
    const currentSaveStatus = isAutoSave ? 'auto-saving' : 'saving';
    setSaveStatus(currentSaveStatus);

    // Ensure Lat/Lon are numbers before sending
    const dataToSend = {
        ...personaData,
        Latitude: Number(personaData.Latitude) || 0,
        Longitude: Number(personaData.Longitude) || 0,
    };

    return fetch(`${API_ENDPOINT}/update-user-persona`, { // Return the promise
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ data: dataToSend }) // Send the validated data
    })
    .then(response => {
      if (!response.ok) {
        // Throw an error object with status for better handling
        throw { status: response.status, message: `Network response was not ok: ${response.status}` };
      }
      return response.json();
    })
    .then(data => {
      if (!data.success) {
        // Throw an error object with backend message
        throw { message: data.message || 'Unknown error from backend update' };
      }
      // Success!
      setSaveStatus(isAutoSave ? null : 'saved'); // Clear auto-save status, show 'saved' for manual
      if (!isAutoSave) {
          setTimeout(() => setSaveStatus(null), 2000); // Clear manual save status after delay
      }
      setLastUpdated(new Date()); // Update timestamp on successful save
      return data; // Return success data
    })
    .catch(err => {
      const errorStatus = isAutoSave ? 'auto-save-error' : 'error';
      setSaveStatus(errorStatus);
      // Optionally clear error status after a delay for auto-save errors too
       if (isAutoSave) {
            setTimeout(() => setSaveStatus(null), 3000);
       } else {
            // Keep manual save error visible longer or until user acts
            setTimeout(() => setSaveStatus(null), 3000);
       }
      // Re-throw the error so the calling context knows it failed
      throw err;
    });
  };
  // --- End Refactored Function ---

  // Function to fetch user persona and then try to get location
  const fetchUserPersonaAndLocation = () => {
    setLocationStatus('idle');
    setLocationDenied(false);
    setSaveStatus(null);

    fetch(`${API_ENDPOINT}/get-user-persona`)
      .then(response => {
        if (!response.ok) {
          throw new Error(`Network response was not ok: ${response.status}`);
        }
        return response.json();
      })
      .then(initialPersonaData => {
        if (navigator.geolocation) {
          setLocationStatus('fetching');
          navigator.geolocation.getCurrentPosition(
            (position) => {
              // --- Location Success ---
              const { latitude, longitude } = position.coords;
              setLocationStatus('success');
              setLocationDenied(false);

              const updatedPersona = {
                ...initialPersonaData,
                Latitude: parseFloat(latitude.toFixed(6)),
                Longitude: parseFloat(longitude.toFixed(6))
              };

              // 1. Update local state immediately for UI responsiveness
              setUserPersona(updatedPersona);
              if (!isEditing) {
                setEditablePersona(JSON.parse(JSON.stringify(updatedPersona)));
              }
               setLastUpdated(new Date()); // Reflect local update time initially

              // 2. Trigger automatic save to backend
              updatePersonaOnBackend(updatedPersona, true)
                .then(() => {
                    // Auto-save successful - state is already updated
                    // lastUpdated is set within updatePersonaOnBackend on success
                })
                .catch(error => {
                    // Auto-save failed. State still reflects fetched location locally.
                    // Error is logged within updatePersonaOnBackend.
                    // User might need to manually save later if they make other edits.
                });
                // --- End Auto Save ---

            },
            (error) => {
              // Location Error
              if (error.code === error.PERMISSION_DENIED) {
                setLocationStatus('denied');
                setLocationDenied(true);
              } else {
               setLocationStatus('error');
               setLocationDenied(false);
              }

              // Update state with fetched persona data (without new location)
              const finalPersona = {
                 ...initialPersonaData,
                 Latitude: initialPersonaData.Latitude ?? 0,
                 Longitude: initialPersonaData.Longitude ?? 0
               };
              setUserPersona(currentPersona => {
                 if (JSON.stringify(finalPersona) !== JSON.stringify(currentPersona)) {
                    setLastUpdated(new Date());
                    return finalPersona;
                 } return currentPersona; });
              if (!isEditing) {
                setEditablePersona(JSON.parse(JSON.stringify(finalPersona)));
              }
            },
            { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
          );
        } else {
          // Geolocation not supported
          setLocationStatus('unsupported');
          setLocationDenied(false);
          const finalPersona = { ...initialPersonaData, Latitude: initialPersonaData.Latitude ?? 0, Longitude: initialPersonaData.Longitude ?? 0 };
          setUserPersona(currentPersona => {
              if (JSON.stringify(finalPersona) !== JSON.stringify(currentPersona)) {
                  setLastUpdated(new Date());
                  return finalPersona;
              } return currentPersona; });
          if (!isEditing) {
            setEditablePersona(JSON.parse(JSON.stringify(finalPersona)));
          }
        }
      })
      .catch(error => {
        setUserPersona(null);
        setEditablePersona(null);
        setLocationStatus('error');
        setLocationDenied(false);
      });
  };

  // Manual Save Changes (uses the refactored backend update function)
  const saveChanges = () => {
    // Don't save if no changes or already saving
    if (!hasUnsavedChanges || saveStatus === 'saving' || saveStatus === 'auto-saving') return;

    const personaToSave = { ...editablePersona }; // Use the current editable state

    updatePersonaOnBackend(personaToSave, false) // false indicates manual save
        .then(() => {
            // On successful MANUAL save:
            setUserPersona(personaToSave); // Update base persona
            // editablePersona is already personaToSave
            setIsEditing(false); // Exit edit mode
            // 'saved' status and timeout are handled within updatePersonaOnBackend
        })
        .catch(error => {
            // Manual save failed.
            // Keep isEditing true so user can retry or cancel.
            // 'error' status and timeout are handled within updatePersonaOnBackend
            console.error("Manual save failed.");
        });
  };

  // Format different value types for display
  const formatValue = (value, key) => {
    if (value === null || value === undefined) return " ";
    if (Array.isArray(value)) return value.length > 0 ? value.join(", ") : " ";
    if (typeof value === 'object' && value !== null) return Object.entries(value).map(([k, v]) => `${k}: ${v}`).join(", ");
    if ((key === 'Latitude' || key === 'Longitude') && typeof value === 'number') return value.toFixed(6);
    return value.toString();
  };

  // Handle field changes
  const handleFieldChange = (key, value) => {
    const updatedPersona = {...editablePersona};
    const originalValueType = typeof userPersona[key];

    if (Array.isArray(userPersona[key])) {
      updatedPersona[key] = value.split(",").map(item => item.trim()).filter(item => item !== "");
    } else if (originalValueType === 'object' && userPersona[key] !== null) {
      const obj = {};
      value.split(",").forEach(pair => {
        const [k, v] = pair.split(":").map(part => part.trim());
        if (k && v) obj[k] = v;
      });
      updatedPersona[key] = obj;
    } else if (key === 'Latitude' || key === 'Longitude' || originalValueType === 'number') {
       const num = parseFloat(value);
       updatedPersona[key] = isNaN(num) ? '' : num; // Allow empty string for temporary invalid input, but backend saves 0
    } else if (originalValueType === 'boolean') {
      updatedPersona[key] = value === 'true';
    } else {
      updatedPersona[key] = value;
    }
    setEditablePersona(updatedPersona);
  };

  // Initial load
  useEffect(() => {
    fetchUserPersonaAndLocation();
  }, []);

   // Polling (unchanged)
   useEffect(() => {
    if (isEditing || !userPersona || saveStatus === 'auto-saving') return; // Avoid polling during auto-save

    const intervalId = setInterval(() => {
      console.log("Polling for user persona updates...");
      fetch(`${API_ENDPOINT}/get-user-persona`)
        .then(response => { if (!response.ok) throw new Error('Polling failed'); return response.json(); })
        .then(polledData => {
          setUserPersona(currentPersona => {
             // Avoid overwriting if an auto-save is happening or just finished
             if (saveStatus === 'auto-saving') return currentPersona;

             const currentRelevant = { ...currentPersona }; delete currentRelevant.Latitude; delete currentRelevant.Longitude;
             const polledRelevant = { ...polledData }; delete polledRelevant.Latitude; delete polledRelevant.Longitude;

             if (JSON.stringify(polledRelevant) !== JSON.stringify(currentRelevant)) {
               console.log("Persona data changed on backend, updating (preserving location)...");
               const mergedData = { ...polledData, Latitude: polledData.Latitude ?? currentPersona.Latitude ?? 0, Longitude: polledData.Longitude ?? currentPersona.Longitude ?? 0 };
               if (!isEditing) { setEditablePersona(JSON.parse(JSON.stringify(mergedData))); }
               setLastUpdated(new Date());
               return mergedData;
             } return currentPersona;
          });
        })
        .catch(error => console.error("Error polling user persona:", error));
    }, 15000);

    return () => clearInterval(intervalId);
  }, [isEditing, userPersona, saveStatus]); // Add saveStatus to dependencies

  // Toggle edit mode
  const toggleEditMode = () => {
    if (saveStatus === 'auto-saving') return; // Prevent mode change during auto-save
    if (isEditing) {
      setEditablePersona(JSON.parse(JSON.stringify(userPersona)));
      setSaveStatus(null);
    } else {
      setEditablePersona(JSON.parse(JSON.stringify(userPersona)));
    }
    setIsEditing(!isEditing);
  };



  // Cancel editing and revert changes
  const cancelEditing = () => {
    if (saveStatus === 'auto-saving') return; // Prevent cancel during auto-save
    setEditablePersona(JSON.parse(JSON.stringify(userPersona)));
    setIsEditing(false);
    setSaveStatus(null);
  };

  // Determine if there are unsaved changes
  const hasUnsavedChanges = isEditing && JSON.stringify(editablePersona) !== JSON.stringify(userPersona);

  return (
    <div className="w-56 max-h-[26rem] bg-white/90 backdrop-blur-sm rounded-lg border border-purple-200 shadow-lg overflow-hidden">
      {/* Header */}
      <div className="p-2 bg-gradient-to-r from-[#d2bfef] via-[#c9b3ea] to-[#ffffff] text-purple-800 flex-shrink-0">
        <div className="grid grid-cols-3 items-center">
          {/* Left - Avatar */}
          <div className="flex justify-start">
            <div className="w-6 h-6 bg-gradient-to-br from-purple-300 to-purple-500 rounded-full flex items-center justify-center text-white font-bold shadow-inner text-xs">
              {userPersona?.Name ? userPersona.Name.charAt(0).toUpperCase() : 'U'}
            </div>
          </div>
          
          {/* Center - Name */}
          <div className="text-center">
            {isEditing ? (
              <input 
                type="text" 
                className="w-full p-0.5 bg-white/40 border border-purple-300 rounded text-xs text-purple-800 text-center focus:ring-1 focus:ring-purple-400 focus:border-purple-400" 
                value={editablePersona?.Name || ''} 
                onChange={(e) => handleFieldChange('Name', e.target.value)} 
                disabled={saveStatus === 'auto-saving'}
              />
            ) : (
              <h2 className="font-bold text-sm truncate">{userPersona?.Name || 'Profile'}</h2>
            )}
          </div>
          
          {/* Right - Edit Button */}
          <div className="flex justify-end">
            <button 
              onClick={toggleEditMode} 
              className="text-purple-800 hover:bg-white/30 rounded p-1.5 transition-all duration-200" 
              title={isEditing ? "View Mode" : "Edit Mode"} 
              disabled={saveStatus === 'auto-saving'}
              style={{ minWidth: '28px', minHeight: '28px' }}
            >
              {isEditing ? (
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M2.036 12.322a1.012 1.012 0 010-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178z" />
                  <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
              ) : (
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M16.862 4.487l1.687-1.688a1.875 1.875 0 112.652 2.652L10.582 16.07a4.5 4.5 0 01-1.897 1.13L6 18l.8-2.685a4.5 4.5 0 011.13-1.897l8.932-8.931zm0 0L19.5 7.125M18 14v4.75A2.25 2.25 0 0115.75 21H5.25A2.25 2.25 0 013 18.75V8.25A2.25 2.25 0 015.25 6H10" />
                </svg>
              )}
            </button>
          </div>
        </div>

        {/* Status Indicators */}
        {saveStatus === 'auto-saving' && (
          <div className="mt-1 bg-blue-500/20 text-blue-100 px-2 py-1 rounded text-xs flex items-center gap-1.5">
            <svg className="animate-spin h-3 w-3" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Saving location...
          </div>
        )}
        
        {locationDenied && !isEditing && (
          <div className="mt-1 bg-yellow-500/20 text-yellow-100 px-2 py-1 rounded text-xs flex items-center gap-1.5">
            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
            Location access denied
          </div>
        )}
      </div>

      {/* Edit Mode Controls */}
      {isEditing && (
        <div className="bg-blue-50 px-3 py-2 flex justify-between items-center border-b border-blue-100 flex-shrink-0">
          <span className="text-xs text-blue-700 font-medium flex items-center gap-1.5">
            <div className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-pulse"></div>
            Edit {hasUnsavedChanges ? <span className="text-orange-600 font-semibold">(Unsaved)</span> : ''}
          </span>
          <div className="flex gap-1.5">
            <button 
              onClick={saveChanges} 
              className={`px-2 py-1 rounded text-xs text-white font-medium transition-all duration-200 ${
                saveStatus === "saving" ? 'bg-gray-400 cursor-not-allowed' : 'bg-green-500 hover:bg-green-600'
              } ${!hasUnsavedChanges && saveStatus !== "saving" ? 'opacity-50 cursor-not-allowed' : ''}`} 
              disabled={saveStatus === "saving" || !hasUnsavedChanges}
            > 
              {saveStatus === "saving" ? "Saving..." : "Save"} 
            </button>
            <button 
              onClick={cancelEditing} 
              className="bg-gray-500 hover:bg-gray-600 text-white px-2 py-1 rounded text-xs font-medium transition-all duration-200"
            > 
              Cancel 
            </button>
          </div>
        </div>
      )}

      {/* Save Status Messages */}
      {saveStatus === "saved" && (
        <div className="bg-green-100 text-green-800 py-1 px-2 text-center text-xs font-medium border-b border-green-200 flex-shrink-0"> 
          ✓ Saved successfully! 
        </div>
      )}
      {saveStatus === "error" && (
        <div className="bg-red-100 text-red-800 py-1 px-2 text-center text-xs font-medium border-b border-red-200 flex-shrink-0"> 
          ✗ Error saving changes 
        </div>
      )}
      {saveStatus === "auto-save-error" && (
        <div className="bg-red-100 text-red-800 py-1 px-2 text-center text-xs font-medium border-b border-red-200 flex-shrink-0"> 
          ⚠️ Auto-save failed 
        </div>
      )}

      {/* Language Selector */}
      {currentLanguage && (
        <div className="px-2 py-2 border-b border-purple-100 bg-gradient-to-r from-purple-50 to-white flex-shrink-0">
          <div className="flex items-center mb-1">
            <div className="w-3 h-3 bg-purple-300 rounded-full mr-1.5"></div>
            <div className="text-xs text-purple-700 font-medium">Language</div>
          </div>
          <LanguageSelector
            currentLanguage={currentLanguage}
            setCurrentLanguage={setCurrentLanguage}
            buttonClassName="w-full bg-white/80 hover:bg-white text-gray-700 py-1.5 px-2 rounded border border-purple-100 flex items-center justify-between text-xs font-medium"
          />
        </div>
      )}
      
      {/* Content Area */}
      <div className="flex-1 overflow-y-auto p-2 text-sm max-h-72 custom-scrollbar">
        {/* Loading/Error States */}
        {!userPersona && (locationStatus === 'fetching' || locationStatus === 'idle') ? (
          <div className="text-center p-4">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-600 mx-auto mb-2"></div>
            <p className="text-gray-600 text-xs">Loading profile...</p>
          </div>
        )
        : !userPersona && locationStatus === 'error' ? (
          <div className="text-center p-4">
            <div className="text-red-500 text-2xl mb-2">⚠️</div>
            <p className="text-red-600 text-xs font-medium">Failed to load profile</p>
          </div>
        )
        : userPersona && (
          <div className="space-y-2">
            {Object.entries(isEditing ? editablePersona : userPersona)
              .filter(([key, _]) => key !== 'Name')
              .map(([key, value]) => (
              <div key={key} className="bg-white/60 backdrop-blur-sm rounded-lg p-2 border border-purple-100 shadow-sm">
                <label className="font-medium capitalize text-gray-700 block mb-1 text-xs"> 
                  {key.replace(/([A-Z])/g, ' $1').trim()} 
                </label>
                {isEditing ? (
                  <div>
                    {key === 'Latitude' || key === 'Longitude' ? (
                      <input 
                        type="number" 
                        step="any" 
                        className="w-full p-1.5 border border-purple-200 rounded text-xs focus:ring-1 focus:ring-purple-500 focus:border-purple-500 bg-white/80" 
                        value={editablePersona[key]} 
                        onChange={(e) => handleFieldChange(key, e.target.value)} 
                        disabled={saveStatus === 'auto-saving'} 
                      />
                    )
                    : Array.isArray(userPersona[key]) ? (
                      <textarea 
                        className="w-full p-1.5 border border-purple-200 rounded text-xs focus:ring-1 focus:ring-purple-500 focus:border-purple-500 bg-white/80 resize-none" 
                        value={formatValue(editablePersona[key], key)} 
                        onChange={(e) => handleFieldChange(key, e.target.value)} 
                        rows={1} 
                        placeholder="Comma-separated list" 
                        disabled={saveStatus === 'auto-saving'}
                      />
                    )
                    : typeof userPersona[key] === 'object' && userPersona[key] !== null ? (
                      <textarea 
                        className="w-full p-1.5 border border-purple-200 rounded text-xs focus:ring-1 focus:ring-purple-500 focus:border-purple-500 bg-white/80 resize-none" 
                        value={formatValue(editablePersona[key], key)} 
                        onChange={(e) => handleFieldChange(key, e.target.value)} 
                        rows={1} 
                        placeholder="key: value, key2: value2" 
                        disabled={saveStatus === 'auto-saving'}
                      />
                    )
                    : typeof userPersona[key] === 'boolean' ? (
                      <select 
                        className="w-full p-1.5 border border-purple-200 rounded bg-white/80 text-xs focus:ring-1 focus:ring-purple-500 focus:border-purple-500" 
                        value={editablePersona[key]?.toString()} 
                        onChange={(e) => handleFieldChange(key, e.target.value)} 
                        disabled={saveStatus === 'auto-saving'}
                      > 
                        <option value="true">True</option> 
                        <option value="false">False</option> 
                      </select>
                    )
                    : typeof userPersona[key] === 'number' ? (
                      <input 
                        type="number" 
                        className="w-full p-1.5 border border-purple-200 rounded text-xs focus:ring-1 focus:ring-purple-500 focus:border-purple-500 bg-white/80" 
                        value={editablePersona[key]} 
                        onChange={(e) => handleFieldChange(key, e.target.value)} 
                        disabled={saveStatus === 'auto-saving'}
                      />
                    )
                    : (
                      <input 
                        type="text" 
                        className="w-full p-1.5 border border-purple-200 rounded text-xs focus:ring-1 focus:ring-purple-500 focus:border-purple-500 bg-white/80" 
                        value={editablePersona[key]} 
                        onChange={(e) => handleFieldChange(key, e.target.value)} 
                        disabled={saveStatus === 'auto-saving'}
                      />
                    )}
                  </div>
                ) : (
                  <div className="text-gray-800 text-xs bg-white/80 p-1.5 rounded break-words min-h-[24px] border border-purple-100/50"> 
                    {formatValue(value, key)} 
                  </div>
                )}
              </div>
            ))}
            

            
            {/* No timestamp at the bottom */}
          </div>
        )}
      </div>
    </div>
  );
};