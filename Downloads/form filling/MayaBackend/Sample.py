import os
import json
import base64
import subprocess
import time
import logging
import asyncio
from fastapi import FastAPI, HTTPException, Request, File, UploadFile
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
from geopy.geocoders import Nominatim
from utils import (
    exec_command, 
    read_json_transcript, 
    audio_file_to_base64,
    flatten_json,
    save_json_to_excel,
    clear_json_structure,
    fill_pdf,
    send_email_with_pdf
)
from language_mapping import LANGUAGE_MAPPING_Deepgram, LANGUAGE_MAPPING_Groq
from groq_client import transcribe_audio_groq
from openai_client import process_gpt_response, call_chatgpt, pdf_maker,get_async_http_client
from form_template import data_dict
from deepgram_client import transcribe_audio_file, initialize_deepgram
initialize_deepgram("b44dcad772560c54fbd0206b9ebcf96feb3e2f54")
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

# Allow all origins (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://test.mayacode.io", "http://localhost:3000", "http://localhost:5173"],  # Adjust to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load API keys from environment
openai.api_key = '<OPENAI_API_KEY>'
ELEVEN_LABS_API_KEY = '<ELEVEN_LABS_API_KEY>'
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




async def get_or_generate_lipsync_by_words(text, message_index, audio_file):
    """Generate lip sync data for the given audio file"""
    try:
        # Generate new lip sync data
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

async def lip_sync_message(message_index: int) -> None:
    """Convert the generated MP3 to WAV and create a lipsync JSON."""
    try:
        start_time = time.time()
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

async def generate_audio_responses(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate audio for each message using ElevenLabs with word-count based lip sync caching"""
    try:
        client = ElevenLabs(api_key=ELEVEN_LABS_API_KEY)
        
        for i, message in enumerate(messages):
            text_input = message.get("text")
            if not text_input:
                continue

            file_name = f"audios/message_{i}.wav"
            
            # Generate audio using ElevenLabs
            audio_bytes = b"".join(client.text_to_speech.convert(
                voice_id=VOICE_ID,
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
            logger.info(f"Audio saved to {file_name}")
            
            # Get or generate lip sync data based on word count
            lip_sync_data = await get_or_generate_lipsync_by_words(text_input, i, file_name)
            
            # Add audio and lip sync to message
            message["audio"] = await audio_file_to_base64(file_name)
            message["lipsync"] = lip_sync_data
        
        return messages
    except Exception as e:
        logger.error(f"Error generating audio responses: {str(e)}")
        raise

async def process_json_and_call_api(user_message):
    """
    Load JSON from a file, send it to the ChatGPT API with a user message,
    and update the original JSON file with the response.
    
    Args:
        user_message (str): User message (transcribed text) to send to the API
        
    Returns:
        dict: Updated JSON data with API response

    """
    logger.info(f"Processing JSON and calling API with user message: {user_message}")
    try:
        # Ensure the JSON file exists, create it if it doesn't
        if not os.path.exists(JSON_FILE_PATH):
            logger.info(f"Creating new JSON file at {JSON_FILE_PATH}")
            with open(JSON_FILE_PATH, 'w') as file:
                json.dump({"conversations": []}, file)
        logger.info(f"Reading JSON file from {JSON_FILE_PATH}")        
        # Load JSON from file
        with open(JSON_FILE_PATH, 'r') as file:
            try:
                json_data = json.load(file)
            except json.JSONDecodeError:
                # If the file exists but is empty or invalid, initialize it
                json_data = {"conversations": []}
        logger.info(f"Loaded JSON data: {json_data}")
        # Create prompt with JSON content and user message
        prompt = f"""Current JSON:
{json.dumps(json_data, indent=2)}

User Message: {user_message}"""
        
        # Call the API
        response = await call_chatgpt(prompt)
        
        # Parse the response and update the JSON file
        parsed_response = json.loads(response)
        
        # Write the updated JSON back to the file
        with open(JSON_FILE_PATH, 'w') as file:
            json.dump(parsed_response, file, indent=2)
            
        logger.info(f"Updated JSON file with new conversation entry")
        return json_data
        
    except Exception as e:
        logger.error(f"Error processing JSON and calling API: {e}")
        return None

@app.get("/api")
async def root():
    return {"message": "Hello World!"}
@app.post("/api/update-user-persona")
async def update_user_persona(data: UserPersonaUpdate):
    try:
        logger.info(f"Updating user persona at {JSON_FILE_PATH}")
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(JSON_FILE_PATH), exist_ok=True)
        
        # Write the updated data to the file
        with open(JSON_FILE_PATH, 'w') as file:
            json.dump(data.data, file, indent=2)
            
        logger.info(f"Successfully updated user persona")
        return {"success": True, "message": "User persona updated successfully"}
        
    except Exception as e:
        logger.error(f"Error updating user persona: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update user persona: {str(e)}")

@app.get("/api/get-user-persona")
async def get_user_persona():
    try:
        logger.info(f"Reading user persona from {JSON_FILE_PATH}")
        
        # Check if file exists
        if not os.path.exists(JSON_FILE_PATH):
            logger.info(f"User persona file not found, returning empty object")
            return {}
        
        # Read the file
        with open(JSON_FILE_PATH, 'r') as file:
            try:
                data = json.load(file)
                return data
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in user persona file")
                return {}
                
    except Exception as e:
        logger.error(f"Error reading user persona: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to read user persona: {str(e)}")

@app.post("/api/end-chat")
async def end_chat(request: Request):
    # Get data from request
    global Temp_Dict
    data = await request.json()
    json_file_path = JSON_FILE_PATH
    
    if not json_file_path or not os.path.exists(json_file_path):
        return {"status": "error", "message": "Invalid or missing JSON file path"}
    
    try:
        # Read the JSON file
        with open(json_file_path, 'r') as file:
            chat_data = json.load(file)
        
        # Save to Excel using our pandas-based function
        save_json_to_excel(chat_data, EXCEL_FILE_PATH)
            
        # Clear the JSON but maintain structure
        cleared_data = clear_json_structure(chat_data)
        User_messages.clear()
        Bot_messages.clear()
        Temp_Dict = data_dict.copy()
        
        # Write cleared data back to the JSON file
        with open(json_file_path, 'w') as file:
            json.dump(cleared_data, file, indent=2)
                
        # Return response without PDF data
        return {
            "status": "success", 
            "message": "Chat ended and data saved to Excel"
        }
    
    except Exception as e:
        logger.error(f"Error in end_chat: {e}")
        return {"status": "error", "message": str(e)}
@app.get("/api/get-pdf")
async def get_pdf_base64(action: str):
    global Temp_Dict
    # Validate action parameter
    if action not in ["show", "send"]:
        return {
            "status": "error"
        }
    
    # Define expected PDF path (adjust based on what fill_pdf returns)
    pdf_path = "filled.pdf"
        
    try:
        # For "show" action, generate the PDF and return it as base64
        if action == "show":
            logger.info("Generating PDF form with collected data")
            
            # Generate the PDF
            pdf_path = fill_pdf(Temp_Dict, "editable5.pdf")
            logger.info(f"PDF generated successfully: {pdf_path}")
            
            # Read and encode the PDF to base64
            with open(pdf_path, "rb") as pdf_file:
                pdf_bytes = pdf_file.read()
                pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
                logger.info("PDF encoded to base64 for download")
            
            # Return response with PDF data
            return {
                "status": "success",
                "pdf_data": pdf_base64,
                "pdf_filename": "filled_form.pdf"
            }
        
        # For "send" action, just email the PDF
        elif action == "send":
            # Check if PDF exists
            print(pdf_path)
            if not os.path.exists(pdf_path):
                return {
                    "status": "error",
                    "message": "No PDF has been generated yet. Please view the PDF first."
                }
                
            # Send the PDF via email
            try:
                email_sent = send_email_with_pdf(pdf_path)
                if email_sent:
                    logger.info("Email sent successfully")
                    return {
                        "status": "success",
                        "message": "PDF sent via email",
                        "email_sent": True
                    }
                else:
                    logger.warning("Failed to send email")
                    return {
                        "status": "error",
                        "message": "Failed to send PDF via email",
                        "email_sent": False
                    }
            except Exception as email_error:
                logger.error(f"Error sending email: {email_error}")
                return {
                    "status": "error",
                    "message": f"Failed to send email: {str(email_error)}",
                    "email_sent": False
                }
        
    except Exception as e:
        logger.error(f"Error in PDF operation: {e}")
        return {"status": "error", "message": f"Failed to process PDF: {str(e)}"}
@app.post("/api/recommendation")
async def process_geodata():
    # Check if file exists
    if not os.path.exists(JSON_FILE_PATH):
        return {"error": f"File not found: {JSON_FILE_PATH}"}
    
    try:
        # Read JSON data from file
        with open(JSON_FILE_PATH, 'r') as file:
            data = json.load(file)
        print("loaded data")
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
            return {"error": f" Please make sure you have given location access to maya and prodvided basic details like name and the languages you speak"}
        
        # Create a copy of the data to modify
        processed_data = copy.deepcopy(data)
        print("processed data")
        print(processed_data)
        # Get geolocation data
        geolocator = Nominatim(user_agent="my_app")
        location = geolocator.reverse((data["Latitude"], data["Longitude"]), language='en')
        
        if not location or not location.raw or 'address' not in location.raw:
            return {"error": "Could not retrieve location data"}
        
        address = location.raw['address']
        
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
- Avoid Usage of * in your outputs

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
        print(gpt_response)
        return gpt_response
        
    except json.JSONDecodeError:
        return {"error": f"Invalid JSON format in file: {JSON_FILE_PATH}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
@app.post("/api/set-language")
async def set_language(request: LanguageRequest):
    """Endpoint to set the language code and voice ID based on frontend language request"""
    global Language, VOICE_ID
    
    # Convert to lowercase for case-insensitive comparison
    requested_language = request.language.lower()
    
    # Check if the requested language is supported
    if requested_language not in LANGUAGE_MAPPING_Groq and requested_language not in LANGUAGE_MAPPING_Deepgram:
        raise HTTPException(
            status_code=400, 
            detail=f"Language '{request.language}' is not supported. Supported languages are: {', '.join(LANGUAGE_MAPPING.keys())}"
        )
    if requested_language in LANGUAGE_MAPPING_Groq:
        Language = LANGUAGE_MAPPING_Groq[requested_language]["code"]
        logger.info(f"Language set to {Language}")
        VOICE_ID = LANGUAGE_MAPPING_Groq[requested_language]["voice_id"]
    elif requested_language in LANGUAGE_MAPPING_Deepgram:
        Language = LANGUAGE_MAPPING_Deepgram[requested_language]["code"]
        logger.info(f"Language set to {Language}")
        VOICE_ID = LANGUAGE_MAPPING_Deepgram[requested_language]["voice_id"]
    
    # Set the global variables
    
    
    
    # Simple confirmation response
    return {"status": "success"}

@app.post("/api/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    
    try:
        logger.info("Starting audio transcription pipeline")
        
        # Create a temporary file for the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_audio:
            try:
                # Write the uploaded file content
                content = await audio.read()
                temp_audio.write(content)
                temp_audio.flush()
                
                # Step 1: Transcribe with Deepgram
                with open(temp_audio.name, "rb") as audio_file:
                    audio_data = audio_file.read()
                    if any(info["code"] == Language for info in LANGUAGE_MAPPING_Deepgram.values()):
                        logger.info("Transcribing audio with Deepgram")
                        transcript = await transcribe_audio_file(
                        file_obj=audio_data,  # Pass file object directly
                        language=Language,  # Pass language if specified
                        model="nova-2-general"     # Using Deepgram's latest model
                        )
                        print("deepgram transcript",transcript)
                        # Get transcribed text from result
                        transcribed_text = transcript.get("text", "Unable to transcribe audio. Please try again.")
                    elif any(info["code"] == Language for info in LANGUAGE_MAPPING_Groq.values()):
                        logger.info("Transcribing audio with Groq")
                        transcript = await transcribe_audio_groq(audio_data, language=Language)
                        print("groq transcript",transcript)
                        # Get transcribed text from result
                        transcribed_text = transcript
                    else:
                        # Handle unsupported language
                        logger.error(f"Language {Language} not found in any transcription provider mapping")
                        transcribed_text = ""
                    
                    # Use the pre-initialized Deepgram client

                    
                    # Extract the transcribed text from Deepgram response
                    logger.info(f"Transcription received: {transcribed_text}")

                

                logger.info("Processing transcription through GPT")
                User_messages.append(transcribed_text)
                messages_task = asyncio.create_task(process_gpt_response(transcribed_text , User_messages, Bot_messages))
                
                
                # Step 3: Update JSON file (can run in parallel with everything else)
                logger.info("Updating JSON file with conversation data (parallel)")
                json_task = asyncio.create_task(process_json_and_call_api(transcribed_text))
                
                # Wait for GPT processing to complete
                messages = await messages_task
                print('Gpt ka output is',messages)
                print(type(messages))
                print(messages[0]['text'])
                Bot_messages.append(messages[0]['text'])
                
                # Step 4: Generate audio responses and lip sync (this will handle parallelization internally)
                logger.info("Generating audio responses and lip sync data")
                final_messages = await generate_audio_responses(messages)
                
                # We don't need to await the JSON task since it's running in parallel
                # and doesn't affect the response
                
                # Run pdf_maker in background to process form data
                async def run_pdf_maker_background():
                    try:
                        # Get the bot question (second-to-last bot message, or empty string if not enough messages)
                        bot_question = Bot_messages[-2] if len(Bot_messages) >= 2 else ""
                        
                        # Get the user response (latest user message)
                        client_response = User_messages[-1]  # User messages will always have at least one element
                        
                        logger.info("Processing form data with pdf_maker in background")
                        print("temporary dict =", Temp_Dict)
                        # Call the pdf_maker function asynchronously
                        pdf_result = await pdf_maker(
                            current_dict=Temp_Dict,
                            client_response=client_response,
                            bot_question=bot_question
                        )
                        
                        # Print the result for debugging
                        logger.info(f"PDF Maker Result: {pdf_result}")
                        
                        # Parse the result as JSON and update Temp_Dict
                        try:
                            updated_dict = json.loads(pdf_result)
                            print("updated dict",updated_dict)
                            print("Temp_Dict before update",Temp_Dict)
                            
                            # Update only non-empty fields from the result
                            for key, value in updated_dict.items():
                                if value and key in Temp_Dict:
                                    Temp_Dict[key] = value
                            print("Temp_Dict after update",Temp_Dict)
                            
                            logger.info(f"Updated Temp_Dict with new values")
                        except json.JSONDecodeError as e:
                            logger.error(f"Error parsing pdf_maker result as JSON: {str(e)}")
                    except Exception as e:
                        logger.error(f"Error in pdf_maker processing: {str(e)}")
                
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