import os
import json
import base64
import subprocess
import time
import logging
import asyncio
# Import authentication and user routers
from auth_routes import router as auth_router
from user_routes import user_router
from redis_session import SessionManager
from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import openai
from elevenlabs import ElevenLabs
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
import tempfile
import wave
import copy
from opencage.geocoder import OpenCageGeocode  # New reliable geocoding
from utils import (
    exec_command, 
    read_json_transcript, 
    audio_file_to_base64,
    flatten_json,
    save_json_to_excel,
    clear_json_structure,
    fill_pdf,
    send_email_with_pdf,
    convert_pdf_with_libreoffice,
    translate_pdf_with_deepl
)
from language_mapping import LANGUAGE_MAPPING_Deepgram, LANGUAGE_MAPPING_Groq
from groq_client import transcribe_audio_groq
from openai_client import process_gpt_response, call_chatgpt, pdf_maker,get_async_http_client
from form_template import data_dict
from deepgram_client import transcribe_audio_file, initialize_deepgram
from auth_middleware import get_current_user_id, get_current_user
initialize_deepgram("b44dcad772560c54fbd0206b9ebcf96feb3e2f54")
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

# Initialize services on startup
@app.on_event("startup")
async def startup_event():
    # Initialize Redis session manager
    try:
        session_mgr = await SessionManager.get_instance()
        logger.info("Redis session manager initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Redis: {str(e)}")
        logger.warning("Continuing without Redis - session isolation will be limited")

# Include auth and user routers
app.include_router(auth_router, prefix='/api')
app.include_router(user_router, prefix='/api')

# Configure CORS for authentication with cookies
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://test.mayacode.io", "https://dashboard.mayacode.io", "http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173", "http://localhost:8080","https://teaching.mayacode.io"],
    allow_credentials=True,  # Important for cookies
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
    expose_headers=["Set-Cookie"],
    max_age=600  # Cache preflight requests for 10 minutes
)

# Load API keys from environment
openai.api_key = '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'
ELEVEN_LABS_API_KEY = 'sk_458ab78174e06ab982e64303812896fc64f912a704063222'
VOICE_ID =  "9BWtsMINqrJLrRacOk9x"
Language =  None


# Hardcoded path for JSON data storage
User_messages=[]
Bot_messages=[]
JSON_FILE_PATH = "./userpersona.json"
EXCEL_FILE_PATH = "./User_Data.xlsx"
Temp_Dict = data_dict.copy()

# Lip sync configuration
if os.path.exists(EXCEL_FILE_PATH):
        print(f"Excel file already exists at: {EXCEL_FILE_PATH}")
else:
    try:
        # Create an empty DataFrame
        df = pd.DataFrame()
        
        # Save as Excel file
        df.to_excel(EXCEL_FILE_PATH, index=False)
        
        print(f"Excel file created at: {EXCEL_FILE_PATH}")
    except Exception as e:
        print(f"Error creating Excel file: {e}")
        
class ChatRequest(BaseModel):
    message: Optional[str] = None

class UserPersonaUpdate(BaseModel):
    data: Dict[str, Any]

class LanguageRequest(BaseModel):
    language: str

# Language mapping has been moved to language_mapping.py




async def get_or_generate_lipsync_by_words(text, message_index, audio_file, user_id=None):
    """Generate lip sync data for the given audio file"""
    try:
        # Generate new lip sync data
        if user_id:
            output_json = f"audios/{user_id}_message_{message_index}.json"
        else:
            output_json = f"audios/message_{message_index}.json"
        
        # Ensure audios directory exists
        os.makedirs("audios", exist_ok=True)
        
        current_dir = os.getcwd()
        rhubarb_dir = os.path.join(current_dir, "rhubarb")
        
        # Construct relative paths
        relative_audio = os.path.join("..", audio_file)
        relative_json = os.path.join("..", output_json)
        
        # Generate lip sync
        start_time = time.time()
        os.chdir(rhubarb_dir)
        exec_command(f'./rhubarb -q --threads 2 -f json -o {relative_json} {relative_audio} -r phonetic')
        os.chdir(current_dir)
        
        # Read the result
        with open(output_json, 'r') as f:
            lip_sync_data = json.load(f)
        
        logger.info(f"Generated lip sync for message {message_index} (generated in {int((time.time() - start_time) * 1000)}ms)")
        
        return lip_sync_data
        
    except Exception as e:
        logger.error(f"Error in lip sync processing: {str(e)}")
        raise

async def lip_sync_message(message_index: int, user_id: str = None) -> None:
    """Convert the generated MP3 to WAV and create a lipsync JSON."""
    try:
        start_time = time.time()
        if user_id:
            output_wav = f"audios/{user_id}_message_{message_index}.wav"
            output_json = f"audios/{user_id}_message_{message_index}.json"
        else:
            output_wav = f"audios/message_{message_index}.wav"
            output_json = f"audios/message_{message_index}.json"
        
        # Ensure audios directory exists
        os.makedirs("audios", exist_ok=True)

        current_dir = os.getcwd()
        rhubarb_dir = os.path.join(current_dir, "rhubarb")
        
        # Construct relative paths from rhubarb directory to the wav and json files
        relative_wav = os.path.join("..", output_wav)
        relative_json = os.path.join("..", output_json)
        
        os.chdir(rhubarb_dir)
        # Optimize Rhubarb by using phonetic recognition only (faster)
        # Add -q flag for quiet mode to reduce logging output
        exec_command(f'./rhubarb -q --threads 2 -f json -o {relative_json} {relative_wav} -r phonetic')
        os.chdir(current_dir)
                
        logger.info(f"Lip sync done in {int((time.time() - start_time) * 1000)}ms")
    except Exception as e:
        logger.error(f"Error in lip sync processing: {str(e)}")
        raise

async def generate_audio_responses(messages: List[Dict[str, Any]], user_id: str = None) -> List[Dict[str, Any]]:
    """Generate audio for each message using ElevenLabs with word-count based lip sync caching"""
    try:
        client = ElevenLabs(api_key=ELEVEN_LABS_API_KEY)
        
        # Get user's voice preference from Redis or fallback to default
        voice_id_to_use = VOICE_ID  # Default fallback
        if user_id:
            try:
                session_mgr = await SessionManager.get_instance()
                user_voice_id = await session_mgr.get_user_voice_id(user_id)
                if user_voice_id:
                    voice_id_to_use = user_voice_id
                    logger.info(f"Using user-specific voice ID for user {user_id}: {voice_id_to_use}")
                else:
                    logger.info(f"No user-specific voice found for user {user_id}, using default: {voice_id_to_use}")
            except Exception as e:
                logger.warning(f"Could not get user voice preference for {user_id}, using default: {str(e)}")
        
        for i, message in enumerate(messages):
            text_input = message.get("text")
            if not text_input:
                continue

            # Create user-specific filename to prevent race conditions
            if user_id:
                file_name = f"audios/{user_id}_message_{i}.wav"
            else:
                # Fallback for backward compatibility (though this could still have race conditions)
                file_name = f"audios/message_{i}.wav"
            
            # Generate audio using ElevenLabs with user-specific voice
            audio_bytes = b"".join(client.text_to_speech.convert(
                voice_id=voice_id_to_use,
                output_format="pcm_44100",
                text=text_input,
                model_id="eleven_flash_v2_5", 
                optimize_streaming_latency=4,
            ))
            
            # Save the audio file
            sample_rate = 44100  # Must match the 'pcm_44100' request
            num_channels = 1     # Assuming mono audio from ElevenLabs PCM
            sample_width = 2     
            with wave.open(file_name, "wb") as wf:
                wf.setnchannels(num_channels)
                wf.setsampwidth(sample_width)  # Sets sample width in bytes (e.g., 2 for 16-bit)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_bytes)
            logger.info(f"Audio saved to {file_name} for user {user_id if user_id else 'legacy'}")
            
            # Get or generate lip sync data based on word count (pass user_id for race condition fix)
            lip_sync_data = await get_or_generate_lipsync_by_words(text_input, i, file_name, user_id)
            
            # Add audio and lip sync to message
            message["audio"] = await audio_file_to_base64(file_name)
            message["lipsync"] = lip_sync_data
        
        return messages
    except Exception as e:
        logger.error(f"Error generating audio responses: {str(e)}")
        raise

async def process_json_and_call_api(user_message, user_id):
    """
    Load user persona from Redis, send it to the ChatGPT API with a user message,
    and update the user's persona data in Redis with the response.
    
    Args:
        user_message (str): User message (transcribed text) to send to the API
        user_id (str): User ID for Redis key scoping
        
    Returns:
        dict: Updated JSON data with API response

    """
    logger.info(f"Processing persona update for user {user_id} with message: {user_message}")
    try:
        session_mgr = await SessionManager.get_instance()
        
        # Get user's current persona data from Redis
        persona_data = await session_mgr.get_user_persona(user_id)
        
        # If no persona exists, initialize with default structure
        if not persona_data:
            persona_data = {
                "Name": "",
                "Age": "",
                "DateOfBirth": "",
                "Gender": "",
                "OriginCountry": "",
                "MigrationReason": "",
                "ArrivalDate": "",
                "IntendedDestination": "",
                "Education": "",
                "GermanLanguageLevel": "",
                "Languages": [],
                "PreviousOccupation": "",
                "CurrentEmployment": "",
                "ProfessionalSkills": [],
                "GeneralHealth": "",
                "MedicalConditions": "",
                "StressLevel": "",
                "LongTermPlans": "",
                "FiveYearGoals": "",
                "DesiredProfession": "",
                "Latitude": "",
                "Longitude": ""
            }
            logger.info(f"Initialized new persona for user {user_id}")
        
        logger.info(f"Loaded persona data for user {user_id}: {persona_data}")
        
        # Create prompt with persona content and user message
        prompt = f"""Current User Persona:
{json.dumps(persona_data, indent=2)}

User Message: {user_message}"""
        
        # Call the API
        response = await call_chatgpt(prompt)
        
        # Parse the response and update the user's persona in Redis
        try:
            parsed_response = json.loads(response)
            
            # Update user's persona in Redis
            await session_mgr.set_user_persona(user_id, parsed_response)
            
            logger.info(f"Updated persona in Redis for user {user_id}")
            return parsed_response
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing ChatGPT response as JSON for user {user_id}: {str(e)}")
            logger.error(f"Raw response was: {response}")
            return persona_data  # Return original data if parsing fails
        
    except Exception as e:
        logger.error(f"Error processing persona update for user {user_id}: {e}")
        return None

@app.get("/api")
async def root():
    return {"message": "Hello World!"}

@app.post("/api/update-user-persona")
async def update_user_persona(
    data: UserPersonaUpdate,
    user_id: str = Depends(get_current_user_id)
):
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        logger.info(f"Attempting to update user persona for user: {user_id}")
        logger.info(f"Received persona data keys: {list(data.data.keys())}")
        
        session_mgr = await SessionManager.get_instance()
        
        # Update user persona in Redis instead of JSON file
        await session_mgr.update_user_persona(user_id, data.data)
        
        logger.info(f"Successfully updated user persona for user {user_id}")
        return {"success": True, "message": "User persona updated successfully"}
            
    except Exception as e:
        logger.error(f"Error updating user persona for user {user_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update user persona: {str(e)}")

@app.get("/api/get-user-persona")
async def get_user_persona(user_id: str = Depends(get_current_user_id)):
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        session_mgr = await SessionManager.get_instance()
        
        # Get user persona from Redis instead of JSON file
        persona_data = await session_mgr.get_user_persona(user_id)
        
        if persona_data is None:
            logger.info(f"No persona found for user {user_id}, returning empty object")
            return {}
        
        logger.info(f"Retrieved persona for user {user_id}")
        return persona_data
                
    except Exception as e:
        logger.error(f"Error reading user persona: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to read user persona: {str(e)}")

@app.post("/api/end-chat")
async def end_chat(
    request: Request,
    user_id: str = Depends(get_current_user_id)
):
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        session_mgr = await SessionManager.get_instance()
        
        
        # Get user's persona data from Redis for Excel export
        persona_data = await session_mgr.get_user_persona(user_id)
        
        if persona_data:
            # Save persona data to Excel using our pandas-based function
            save_json_to_excel(persona_data, f"User_Data.xlsx")
            logger.info(f"Saved chat data to Excel for user {user_id}")
        
        # Clear only specific session data (NOT persona, language, voice settings)
        await session_mgr.set_user_data(user_id, "user_messages", [])
           # Clear user messages
        await session_mgr.set_user_data(user_id, "bot_messages", [])     # Clear bot messages
        
        # Reset form data to template default
        from form_template import data_dict
        await session_mgr.update_user_data_dict(user_id, data_dict.copy())
        
        # Clean up user-specific PDF file if it exists
        user_pdf_path = f"filled_{user_id}.pdf"
        if os.path.exists(user_pdf_path):
            print(f"Attempting to delete PDF file: {user_pdf_path}")
            try:
                os.remove(user_pdf_path)
                logger.info(f"Deleted PDF file: {user_pdf_path}")
            except Exception as pdf_error:
                logger.warning(f"Failed to delete PDF file {user_pdf_path}: {pdf_error}")
        
        logger.info(f"Cleared messages, form data, and PDF for user {user_id} (kept persona, language, voice settings)")
        
        # Return response
        return {
            "status": "success", 
            "message": "Chat ended and conversation data cleared"
        }
    
    except Exception as e:
        logger.error(f"Error in end_chat for user {user_id}: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/api/get-pdf")
async def get_pdf_base64(
    action: str,
    user_id: str = Depends(get_current_user_id)
):
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Validate action parameter
    if action not in ["show", "send", "translate"]:
        return {
            "status": "error",
            "message": "Invalid action specified"
        }
    
    try:
        session_mgr = await SessionManager.get_instance()
        
        # Get user's form data from Redis instead of global Temp_Dict
        user_form_data = await session_mgr.get_user_data_dict(user_id)
        print(f"User form data for user {user_id}: {user_form_data}")
        
        # Define user-specific PDF path
        pdf_path = f"filled_{user_id}.pdf"
            
        # For "show" action, generate the PDF and return it as base64
        if action == "show":
            logger.info(f"Generating PDF form with collected data for user {user_id}")
            
            # Generate the PDF using user-specific form data
            pdf_path = fill_pdf(user_form_data, "editable5.pdf", user_id)
            logger.info(f"PDF generated successfully: {pdf_path}")
            
            # Read and encode the PDF to base64
            with open(pdf_path, "rb") as pdf_file:
                pdf_bytes = pdf_file.read()
                pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
                logger.info(f"PDF encoded to base64 for download for user {user_id}")
            
            # Save PDF to Redis database when generated
            try:
                pdf_filename = f"filled_form_{user_id}.pdf"
                await session_mgr.save_user_pdf(user_id, pdf_bytes, pdf_filename)
            except Exception as save_error:
                logger.warning(f"Failed to save PDF to Redis for user {user_id}: {save_error}")
                # Don't fail the PDF generation if saving fails
            
            # Return response with PDF data
            return {
                "status": "success",
                "pdf_data": pdf_base64,
                "pdf_filename": f"filled_form_{user_id}.pdf"
            }
        
        # For "send" action, just email the PDF
        elif action == "send":
            # Check if PDF exists
            if not os.path.exists(pdf_path):
                return {
                    "status": "error",
                    "message": "No PDF has been generated yet. Please view the PDF first."
                }
                
            # Send the PDF via email
            try:
                email_sent = await send_email_with_pdf(pdf_path)
                if email_sent:
                    logger.info(f"Email sent successfully for user {user_id}")
                    return {
                        "status": "success",
                        "message": "PDF sent via email",
                        "email_sent": True
                    }
            except Exception as email_error:
                logger.error(f"Error sending email for user {user_id}: {email_error}")
                return {
                    "status": "error",
                    "message": f"Failed to send email: {str(email_error)}",
                    "email_sent": False
                }

        # For "translate" action, flatten, translate, and return the PDF
        elif action == "translate":
            # Define paths for the flattened and translated PDFs
            flattened_pdf_path = f"flattened_{user_id}.pdf"
            translated_pdf_path = f"translated_{user_id}.pdf"

            try:
                # Step 1: Flatten the PDF
                convert_pdf_with_libreoffice(pdf_path, flattened_pdf_path)

                # Step 2: Translate the flattened PDF
                translate_pdf_with_deepl(flattened_pdf_path, translated_pdf_path)

                # Step 3: Read and encode the translated PDF to base64
                with open(translated_pdf_path, "rb") as pdf_file:
                    pdf_bytes = pdf_file.read()
                    pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
                
                return {
                    "status": "success",
                    "pdf_data": pdf_base64,
                    "pdf_filename": os.path.basename(translated_pdf_path)
                }

            except Exception as e:
                logger.error(f"Error during PDF translation process for user {user_id}: {e}")
                return {"status": "error", "message": f"Failed to translate PDF: {str(e)}"}
        
    except Exception as e:
        logger.error(f"Error in PDF operation for user {user_id}: {e}")
        return {"status": "error", "message": f"Failed to process PDF: {str(e)}"}

@app.post("/api/recommendation")
async def process_geodata(user_id: str = Depends(get_current_user_id)):
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        session_mgr = await SessionManager.get_instance()
        
        # Get user's form data from Redis instead of JSON file
        data = await session_mgr.get_user_persona(user_id)
        
        if not data:
            return {"error": "No user data found. Please complete the form first."}
        
        logger.info(f"Processing recommendation for user {user_id}")
        print("loaded data from Redis")
        print(data)
        
        # Check if required fields are present and not empty/null
        missing_fields = []
        
        # Check latitude
        if "Latitude" not in data or data["Latitude"] is None:
            missing_fields.append("Latitude")
        
        # Check longitude
        if "Longitude" not in data or data["Longitude"] is None:
            missing_fields.append("Longitude")
        
        # Check name
        if "Name" not in data or not data["Name"] or data["Name"] == "":
            missing_fields.append("name")
        
        # Check languages
        if "Languages" not in data or not data["Languages"] or (isinstance(data["Languages"], list) and len(data["Languages"]) == 0):
            missing_fields.append("Languages")
        
        # Return error if any required fields are missing or empty
        if missing_fields:
            missing_fields_str = ", ".join(missing_fields)
            return {"error": f" Please make sure you have given location access to maya and prodvided basic details like name and the languages you speak {missing_fields_str}"}
        
        # Create a copy of the data to modify
        processed_data = copy.deepcopy(data)
        logger.info(f"Processing user data for recommendations for user {user_id}")
        print("processed data")
        print(processed_data)
        # Get geolocation data
        geolocator = OpenCageGeocode('4d01a25c76d34889b024278e67896c2c')  # Replace with your actual OpenCage API key
        
        try:
            # OpenCage uses different method and response format
            results = geolocator.reverse_geocode(data["Latitude"], data["Longitude"])
            
            if not results or len(results) == 0:
                return {"error": "Could not retrieve location data"}
            
            location = results[0]  # Get first result
            address = location['components']  # OpenCage uses 'components' instead of 'address'
            
        except Exception as e:
            logger.error(f"Geocoding error: {e}")
            return {"error": "Could not retrieve location data"}
        
        # Add country and state to the data
        processed_data["Country"] = address.get("country")
        processed_data["State"] = address.get("state") or address.get("province") or address.get("county")
        
        # Remove latitude and longitude
        del processed_data["Latitude"]
        del processed_data["Longitude"]
        print("processed data after removing lat long")
        print(processed_data)
        # Prepare message for ChatGPT
        messages = [
            {"role": "system", "content": """ 
## INPUT DATA UNDERSTANDING:
- Current Location (Country and City): This is where the user is currently located
- Origin Country (if provided): This is the user's country of origin/birth, which may differ from current location
- Skills (if provided): Professional skills the user possesses fields from json that will be associated skill , currentemployment , previousoccupation
- Languages (if provided): Languages the user speaks fields from json that will be associated language 
- Medical Conditions (if provided): Any health conditions requiring care fields from json that will be associated medicalcondition and generalhealth

## RECOMMENDATION GUIDELINES:

IMPORTANT: ONLY provide recommendations for fields that are explicitly provided. Skip any sections entirely where no relevant user data exists.

1. IF SKILLS ARE PROVIDED:
   - Provide up to 3 relevant job opportunities matching their listed skills in their current location
   - Each recommendation must include: company name, job title, brief description (1-2 sentences), and contact information/website link

2. IF LANGUAGES ARE PROVIDED:
   - Provide up to 2 language-based jobs (translation, interpretation, auditing, etc.) in their current location
   - Focus on languages they speak that may be valuable in their current location
   - Each recommendation must include: organization name, position type, brief description (1-2 sentences), and contact information/website link

3. IF MEDICAL CONDITIONS ARE PROVIDED:
   - Provide up to 3 appropriate healthcare providers/facilities in their current location that specialize in their conditions
   - Each recommendation must include: facility name, specialty area, brief description (1-2 sentences), address, and contact information/website link
## OUTPUT GUIDELINES:
- Your output should only conating the link  and their samll description

## GENERIC GUIDELINES:
- If no skills are provided, completely omit the "SKILLS-BASED JOB OPPORTUNITIES" section
- If no languages are provided, completely omit the "LANGUAGE-BASED OPPORTUNITIES" section
- If no medical conditions are provided, completely omit the "HEALTHCARE RECOMMENDATIONS" section
- Dont recommend internships
- Avoid results from the same website or organization
- Dont miss any recommendations in case info is provided them to you only skip if no info is provided in json
- Dont ask for more details just give recommendations thats all
- If current location information is insufficient, state "Insufficient location data to provide specific recommendations"
- Only provide real, searchable results that exist in their current location
- Always have Links in your output never miss the links
- Do not include generic suggestions or placeholders
- Always output things in english no other language
- VERY IMPORTANT ALWAYS SEARCH FOR RECOMMENDATIONS NEVER MAKE UP ANYTHING ALWAYS SEARCH IT UP BY FORMULATING APPROPRIATE QUERIES
"""},
            {"role": "user", "content": f"{processed_data}"}
        ]
        
        # Call OpenAI API
        client = openai.AsyncOpenAI(
            api_key=openai.api_key,
            http_client=get_async_http_client()
        )
        response = await client.chat.completions.create(
            model="gpt-4o-mini-search-preview",
            web_search_options={"search_context_size": "high"},
            messages=messages,
        )
        
        # Extract and return only the GPT response
        gpt_response = response.choices[0].message.content
        logger.info(f"Generated recommendations for user {user_id}")
        print(gpt_response)
        return gpt_response
        
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in user data for user {user_id}")
        return {"error": "Invalid user data format. Please complete the form again."}
        
    except Exception as e:
        logger.error(f"Error generating recommendations for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
@app.post("/api/set-language")
async def set_language(
    request: LanguageRequest,
    user_id: str = Depends(get_current_user_id)
):
    """Endpoint to set the language code and voice ID based on frontend language request"""
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        session_mgr = await SessionManager.get_instance()
        
        # Convert to lowercase for case-insensitive comparison
        requested_language = request.language.lower()
        
        # Check if the requested language is supported
        if requested_language not in LANGUAGE_MAPPING_Groq and requested_language not in LANGUAGE_MAPPING_Deepgram:
            raise HTTPException(
                status_code=400, 
                detail=f"Language '{request.language}' is not supported. Supported languages are: {', '.join(LANGUAGE_MAPPING_Groq.keys())}"
            )
        
        # Get language and voice ID settings
        if requested_language in LANGUAGE_MAPPING_Groq:
            language_code = LANGUAGE_MAPPING_Groq[requested_language]["code"]
            voice_id = LANGUAGE_MAPPING_Groq[requested_language]["voice_id"]
        elif requested_language in LANGUAGE_MAPPING_Deepgram:
            language_code = LANGUAGE_MAPPING_Deepgram[requested_language]["code"]
            voice_id = LANGUAGE_MAPPING_Deepgram[requested_language]["voice_id"]
        
        # Store user-specific language and voice settings in Redis
        await session_mgr.set_user_language(user_id, language_code)
        await session_mgr.set_user_voice_id(user_id, voice_id)
        
        logger.info(f"Language set to {language_code} and voice ID to {voice_id} for user {user_id}")
        
        return {
            "status": "success", 
            "language": language_code, 
            "voice_id": voice_id,
            "message": f"Language set to {request.language}"
        }
        
    except Exception as e:
        logger.error(f"Error setting language: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id)
):
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        session_mgr = await SessionManager.get_instance()
        logger.info(f"Starting audio transcription pipeline for user {user_id}")
        
        # Get user's language setting from Redis
        user_language = await session_mgr.get_user_language(user_id)
        
        # Create a temporary file for the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_audio:
            try:
                # Write the uploaded file content
                content = await audio.read()
                temp_audio.write(content)
                temp_audio.flush()
                
                # Step 1: Transcribe with Deepgram or Groq based on user's language
                with open(temp_audio.name, "rb") as audio_file:
                    logger.info(f"Transcribing audio with user language {user_language} for user {user_id}")
                    audio_data = audio_file.read()
                    if any(info["code"] == user_language for info in LANGUAGE_MAPPING_Deepgram.values()):
                        transcript = await transcribe_audio_file(
                        file_obj=audio_data,  # Pass file object directly
                        language=user_language,  # Pass user-specific language
                        model="nova-2-general"     # Using Deepgram's latest model
                        )
                        print("deepgram transcript",transcript)
                        # Get transcribed text from result
                        transcribed_text = transcript.get("text", "Unable to transcribe audio. Please try again.")
                    elif any(info["code"] == user_language for info in LANGUAGE_MAPPING_Groq.values()):
                        transcript = await transcribe_audio_groq(audio_data, language=user_language)
                        print("groq transcript",transcript)
                        # Get transcribed text from result
                        transcribed_text = transcript
                    else:
                        # Handle unsupported language
                        logger.error(f"Language {user_language} not found in any transcription provider mapping for user {user_id}")
                        transcribed_text = ""
                    
                    # Extract the transcribed text from Deepgram response
                    logger.info(f"Transcription received for user {user_id}: {transcribed_text}")
                
                # Get user's message history from Redis

                
                # Start tasks that can run in parallel
                # Step 2: Process through GPT (this needs to finish before audio generation)
                logger.info(f"Processing transcription through GPT for user {user_id}")
                messages_task = asyncio.create_task(process_gpt_response(transcribed_text, user_id=user_id))
                
                
                # Step 3: Update JSON file (can run in parallel with everything else)
                logger.info(f"Updating JSON file with conversation data (parallel) for user {user_id}")
                json_task = asyncio.create_task(process_json_and_call_api(transcribed_text, user_id))
                
                # Wait for GPT processing to complete
                messages = await messages_task
                print('Gpt ka output is',messages)
                print(type(messages))
                print(messages[0]['text'])
                bot_message = messages[0]['text']
                
                # Update Redis with new messages
                await session_mgr.store_user_message(user_id, transcribed_text)
                await session_mgr.store_bot_message(user_id, bot_message)
                user_messages = await session_mgr.get_user_messages(user_id)
                bot_messages = await session_mgr.get_bot_messages(user_id)
                
                # Step 4: Generate audio responses and lip sync (this will handle parallelization internally)
                logger.info(f"Generating audio responses and lip sync data for user {user_id}")
                final_messages = await generate_audio_responses(messages, user_id)
                
                # We don't need to await the JSON task since it's running in parallel
                # and doesn't affect the response
                
                # Run pdf_maker in background to process form data
                async def run_pdf_maker_background():
                    try:
                        # Get the bot question (second-to-last bot message, or empty string if not enough messages)
                        print("bot messages",bot_messages)
                        bot_question = bot_messages[-2] if len(bot_messages) >= 2 else ""
                        
                        # Get the user response (latest user message)
                        print("user messages",user_messages)
                        client_response = user_messages[-1]  # User messages will always have at least one element
                        
                        # Get user's form data from Redis
                        temp_dict = await session_mgr.get_user_data_dict(user_id)
                        
                        logger.info(f"Processing form data with pdf_maker in background for user {user_id}")
                        print("temporary dict =", temp_dict)
                        # Call the pdf_maker function asynchronously
                        pdf_result = await pdf_maker(
                            current_dict=temp_dict,
                            client_response=client_response,
                            bot_question=bot_question
                        )
                        
                        # Print the result for debugging
                        logger.info(f"PDF Maker Result for user {user_id}: {pdf_result}")
                        
                        # Parse the result as JSON and update user's form data
                        try:
                            updated_dict = json.loads(pdf_result)
                            print("updated_dict =", updated_dict)
                            
                            # Update only non-empty fields from the result
                            for key, value in updated_dict.items():
                                if value and key in temp_dict:
                                    temp_dict[key] = value
                            logger.info(f"Updated form data with pdf_maker result for user {user_id}: {temp_dict}")
                            
                            # Save updated form data back to Redis
                            await session_mgr.update_user_data_dict(user_id, temp_dict)
                            
                            logger.info(f"Updated form data in Redis for user {user_id}")
                        except json.JSONDecodeError as e:
                            logger.error(f"Error parsing pdf_maker result as JSON for user {user_id}: {str(e)}")
                    except Exception as e:
                        logger.error(f"Error in pdf_maker processing for user {user_id}: {str(e)}")
                
                # Create background task without awaiting it
                # This allows it to run without blocking the response
                asyncio.create_task(run_pdf_maker_background())
                
                return {"messages": final_messages}
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_audio.name)
                except Exception as e:
                    logger.error(f"Error deleting temporary file: {str(e)}")
                    
    except Exception as e:
        logger.error(f"Error in transcribe pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)