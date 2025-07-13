import json
import logging
from time import perf_counter
from typing import List, Dict, Any, Optional

import openai
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global HTTP client for connection pooling
_http_client = None

def get_async_http_client():
    """Returns a reusable httpx.AsyncClient with optimized connection settings"""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100),
            timeout=60.0
        )
    return _http_client

async def call_chatgpt(prompt, model="gpt-4.1", api_key=None):
    """
    Call the ChatGPT API with a prompt, using optimized connection pooling.
    
    Args:
        prompt (str): Prompt to send to the API
        model (str): Model to use for the API call
        api_key (str): API key for OpenAI
        
    Returns:
        str: Response from the API
    """
    try:
        # Start timing the API call
        start_time = perf_counter()
        logger.info("Starting user persona API call")
        
        # If no API key is provided, use the one from the environment
        if api_key is None:
            api_key = openai.api_key
            
        # Get the optimized HTTP client
        client = openai.AsyncOpenAI(
            api_key=api_key,
            http_client=get_async_http_client()
        )
            
        # Using the optimized OpenAI client to get a response
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content":  """You are a persona management assistant. Your job is to maintain a JSON object that tracks user information as it's shared in conversation. You'll receive two inputs:

The user's message
The current state of their persona JSON

When you receive new information in the user's message that isn't already in the JSON, update the JSON with that information. If fields are already populated, do not modify them. Add information that is explicitly mentioned or reasonably inferred from the user message, including inferring language proficiency based on the language the user communicates in.

IMPORTANT: 
- Your output must be STRICTLY JSON format with no additional text, explanations, or commentary.
- DO NOT modify the Latitude and Longitude fields as they are system-managed fields.
- Regardless of the input language Always update the json in english no other language
- Fill the feilds only when user answer questions about them or give hints about them

The persona JSON has these fields:

- Name: User's full name
- Age: User's age (numeric value)
- DateOfBirth: User's date of birth
- Gender: User's gender identity
- OriginCountry: User's country of origin
- MigrationReason: Reasons that led to leaving home country
- ArrivalDate: When they arrived in Germany
- IntendedDestination: Whether Germany was the intended destination or path changed
- Education: Highest educational achievement
- GermanLanguageLevel: German proficiency level (No knowledge, A1, A2, B1, B2, C1, C2)
- Languages: Array of other languages the user speaks (including languages inferred from their messages)
- PreviousOccupation: Previous job title and profession in home country
- CurrentEmployment: Current employment status and field in Germany
- ProfessionalSkills: Array of professional skills or qualifications
- GeneralHealth: General health condition
- MedicalConditions: Any chronic illnesses or pre-existing conditions
- StressLevel: Whether they feel stressed or overwhelmed in day-to-day life
- LongTermPlans: Whether they plan to stay in Germany long-term
- FiveYearGoals: Personal goals for the next five years
- DesiredProfession: Profession they would like to pursue in Germany
- Latitude: Geographic latitude (managed by system) - DO NOT MODIFY
- Longitude: Geographic longitude (managed by system) - DO NOT MODIFY

Examples:

Example 1:
Current JSON:
{
  "Name": "",
  "Age": null,
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
  "Latitude": null,
  "Longitude": null
}
User Message:
"My name is Ahmad Khalid. I'm 32 years old and I came from Syria in 2018 because of the war. I was a civil engineer back home."
Your Response (Complete JSON only):
{
  "Name": "Ahmad Khalid",
  "Age": 32,
  "DateOfBirth": "",
  "Gender": "male",
  "OriginCountry": "Syria",
  "MigrationReason": "war",
  "ArrivalDate": "2018",
  "IntendedDestination": "",
  "Education": "",
  "GermanLanguageLevel": "",
  "Languages": ["Arabic", "English"],
  "PreviousOccupation": "civil engineer",
  "CurrentEmployment": "",
  "ProfessionalSkills": ["civil engineering"],
  "GeneralHealth": "",
  "MedicalConditions": "",
  "StressLevel": "",
  "LongTermPlans": "",
  "FiveYearGoals": "",
  "DesiredProfession": "",
  "Latitude": null,
  "Longitude": null
}

Example 2:
Current JSON:
{
  "Name": "Fatima Hassan",
  "Age": 28,
  "DateOfBirth": "",
  "Gender": "female",
  "OriginCountry": "Afghanistan",
  "MigrationReason": "",
  "ArrivalDate": "",
  "IntendedDestination": "",
  "Education": "",
  "GermanLanguageLevel": "",
  "Languages": ["Dari"],
  "PreviousOccupation": "",
  "CurrentEmployment": "",
  "ProfessionalSkills": [],
  "GeneralHealth": "",
  "MedicalConditions": "",
  "StressLevel": "",
  "LongTermPlans": "",
  "FiveYearGoals": "",
  "DesiredProfession": "",
  "Latitude": 52.52,
  "Longitude": 13.405
}
User Message:
"I was born on May 12, 1996. I arrived in Germany in 2021 after staying in Turkey for two years. I initially wanted to go to Sweden, but the route changed. I have a bachelor's degree in economics, and I speak German at an A2 level, also some English and Turkish. I'm currently feeling quite stressed about finding a job here."
Your Response (Complete JSON only):
{
  "Name": "Fatima Hassan",
  "Age": 28,
  "DateOfBirth": "May 12, 1996",
  "Gender": "female",
  "OriginCountry": "Afghanistan",
  "MigrationReason": "",
  "ArrivalDate": "2021",
  "IntendedDestination": "Sweden",
  "Education": "bachelor's degree in economics",
  "GermanLanguageLevel": "A2",
  "Languages": ["Dari", "English", "Turkish"],
  "PreviousOccupation": "",
  "CurrentEmployment": "unemployed",
  "ProfessionalSkills": ["economics"],
  "GeneralHealth": "",
  "MedicalConditions": "",
  "StressLevel": "stressed about finding a job",
  "LongTermPlans": "",
  "FiveYearGoals": "",
  "DesiredProfession": "",
  "Latitude": 52.52,
  "Longitude": 13.405
}

Example 3:
Current JSON:
{
  "Name": "Mohammed Ali",
  "Age": 45,
  "DateOfBirth": "November 3, 1979",
  "Gender": "male",
  "OriginCountry": "Iraq",
  "MigrationReason": "political persecution",
  "ArrivalDate": "2017",
  "IntendedDestination": "Germany",
  "Education": "",
  "GermanLanguageLevel": "",
  "Languages": ["Arabic"],
  "PreviousOccupation": "",
  "CurrentEmployment": "",
  "ProfessionalSkills": [],
  "GeneralHealth": "",
  "MedicalConditions": "",
  "StressLevel": "",
  "LongTermPlans": "",
  "FiveYearGoals": "",
  "DesiredProfession": "",
  "Latitude": 48.137,
  "Longitude": 11.576
}
User Message:
"I have type 2 diabetes and high blood pressure. I was a doctor in Baghdad before coming here, specialized in internal medicine. My German is now at B2 level after taking intensive courses. I definitely want to stay in Germany and hope to get my medical license recognized within the next five years. I enjoy reading medical journals and playing chess in my free time."
Your Response (Complete JSON only):
{
  "Name": "Mohammed Ali",
  "Age": 45,
  "DateOfBirth": "November 3, 1979",
  "Gender": "male",
  "OriginCountry": "Iraq",
  "MigrationReason": "political persecution",
  "ArrivalDate": "2017",
  "IntendedDestination": "Germany",
  "Education": "medical doctor",
  "GermanLanguageLevel": "B2",
  "Languages": ["Arabic", "English"],
  "PreviousOccupation": "doctor, internal medicine specialist",
  "CurrentEmployment": "",
  "ProfessionalSkills": ["medicine", "internal medicine"],
  "GeneralHealth": "chronic conditions",
  "MedicalConditions": "type 2 diabetes, high blood pressure",
  "StressLevel": "",
  "LongTermPlans": "stay in Germany",
  "FiveYearGoals": "get medical license recognized",
  "DesiredProfession": "doctor",
  "Latitude": 48.137,
  "Longitude": 11.576
} """},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Log API call completion time
        api_time = perf_counter() - start_time
        logger.info(f"User persona API call completed in {api_time:.3f}s")
        
        return response_text
    except Exception as e:
        # Log API call error with timing
        api_time = perf_counter() - start_time
        logger.error(f"Error in call_chatgpt after {api_time:.3f}s: {str(e)}")
        return f"An error occurred: {e}"



async def pdf_maker(current_dict: dict = None, client_response: str = "", bot_question: str = "", model="gpt-4.1", api_key=None, user_id: str = None) -> str:
    """
    Process form data through ChatGPT to extract and update user information.
    
    Args:
        current_dict: User's current form data (if None and user_id provided, gets from Redis)
        client_response: User's response text
        bot_question: Bot's question text  
        model: ChatGPT model to use
        api_key: OpenAI API key
        user_id: User ID for Redis operations (optional for backward compatibility)
        
    Returns:
        str: JSON string with updated form fields
    """
    try:
        start_time = perf_counter()
        logger.info(f"Starting user persona API call (async) for user {user_id if user_id else 'legacy'}")

        # Get current_dict from Redis if user_id provided and current_dict is None
        if user_id and current_dict is None:
            from redis_session import SessionManager
            session_mgr = await SessionManager.get_instance()
            current_dict = await session_mgr.get_user_data_dict(user_id)
            logger.info(f"Retrieved form data from Redis for user {user_id}")
            print("current_dict=", current_dict)
        
        # Fallback to empty dict if still None
        if current_dict is None:
            logger.info("No form data found in Redis, using template")
            from form_template import data_dict 
            current_dict = data_dict.copy() 

        if api_key is None:
            api_key = "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"

        # Get the optimized HTTP client and create an AsyncOpenAI client
        client = openai.AsyncOpenAI(
            api_key=api_key,
            http_client=get_async_http_client()
        )
        
        user_prompt = f"""Current_Dict:  {current_dict}

Question_asked
"{bot_question}"

User_Reponse
"{client_response}" """

        # Build the prompt based on current_dict, question, and response
        prompt = """rent state of the form dictionary
Question_asked: The question posed by the bot
User_Response: The answer given by the user
CRITICAL GUIDELINES:
- If input is in any other language convert it in German
- ALWAYS fill the dictionary in German, no matter the input language of the user, If input is in any other language convert it in German
- Only return fields that need to be updated or filled
- Use exact German terms as specified below
- Follow the precise format requirements for each field type
- Fill only the fields mentioned in the conversation dont make things up or assume things to fill 
- For the fields that are unanswered leave them unfiled dont fill them with hallucinated texts only fill based on question and answers
- Please note while filling the form always put first name in Vorname and last name in Name fields like in the examples shown
- If only 1st name is mentioned then fill the Vorname field only

Based on this information, update or provide details to fill in the user persona (IN GERMAN LANGUAGE) and return only the new or updated details in JSON format.

Dictionary Field Definitions (All in German)
1. Personendaten (Personal Data)
Text Fields - Fill with appropriate German text:

Name: Nachname der Person (Last name)
Vorname: Vorname der Person (First name)
Geburtsdatum: Tag, Monat und Jahr der Geburt (Birth date)
Geburtsland: Land, in dem die Person geboren wurde (Country of birth)
Geburtsort: Stadt oder Ort der Geburt (City/place of birth)
Staatsangehorigkeit: Staatsburgerschaft der Person (Nationality)

2. Selection Fields - Choose EXACTLY one option:
Geschlecht (Gender):

weiblich
mannlich(männlich)
divers
keine Angabe

Familienstand (Marital Status):

ledig
verheiratet
verwitwet
geschieden

3. Beeinträchtigung gemäß § 8 Abs. 1b AsylG (Impairments)
If condition applies, fill with "Yes" - otherwise leave empty:

korperlich: Schwere körperliche Erkrankung oder Behinderung
seelisch: Schwere psychische Erkrankung oder Behinderung
geistig: Eingeschränkte kognitive Fähigkeiten oder Intelligenzstörung
Sinnesbeeintrachtigung: Gehörlosigkeit, Blindheit, Stummheit

4. Hinweis auf eventuelle Vulnerabilitaten (Vulnerabilities)
If condition applies, fill with "Yes" - otherwise leave empty:

Alleinerziehende: Person sorgt allein für Kind/Kinder
Schwangere: Person ist schwanger
alter als 65 Jahre: Person ist über 65 Jahre alt
Verlust oder Trennung von engen Familienangehorigen: Kürzlicher Verlust oder Trennung von nahestehenden Angehörigen
Soziale Isolation: Person ist sozial isoliert, kein Kontakt zur Gesellschaft
Erfahrungen mit korperlicher oder seelischer Gewalt wahrend Flucht oder Aufenthalt: Gewalterfahrungen während Flucht oder Aufenthalt

5. Additional Required Fields

Praktische Hinweise zur Durchführung der Anhörung: Freitextfeld für besondere Hinweise (bevorzugte Sprache, technische Hilfsmittel, Begleitperson, sonstige Umstände)
Im Auftrag: ALWAYS fill with "MayaCode"

Complete Dictionary Structure Reference
{"Name": "",                 (here name field refer to last name)   TEXT FIELD - Fill with German text
"Vorname": "",              (here vorname field refer to first name)   TEXT FIELD - Fill with German text
"Geburtsdatum": "",            TEXT FIELD - Fill with German text
"Geburtsland": "",             TEXT FIELD - Fill with German text
"Geburtsort": "",              TEXT FIELD - Fill with German text
"Staatsangehorigkeit": "",     TEXT FIELD - Fill with German text
"Praktische Hinweise zur Durchfuhrung der Anhorung": "", TEXT FIELD - Fill with German text
"Im Auftrag": "MayaCode",      TEXT FIELD - Always fill with "MayaCode"
"Geschlecht": "",              SELECTION FIELD - Choose: weiblich/mÃ¤nnlich/divers/keine Angabe
"Familienstand": "",           SELECTION FIELD - Choose: ledig/verheiratet/verwitwet/geschieden
"korperlich": "",              Yes/No FIELD - Fill "Yes" if applicable, leave empty otherwise
"seelisch": "",                Yes/No FIELD - Fill "Yes" if applicable, leave empty otherwise
"geistig": "",                 Yes/No FIELD - Fill "Yes" if applicable, leave empty otherwise
"Sinnesbeeintrachtigung": "",  Yes/No FIELD - Fill "Yes" if applicable, leave empty otherwise
"Alleinerziehende": "",        Yes/No FIELD - Fill "Yes" if applicable, leave empty otherwise
"Schwangere": "",              Yes/No FIELD - Fill "Yes" if applicable, leave empty otherwise
"alter als 65 Jahre": "",      Yes/No FIELD - Fill "Yes" if applicable, leave empty otherwise
"Verlust oder Trennung von engen Familienangehorigen": "", Yes/No FIELD - Fill "Yes" if applicable, leave empty otherwise
"Soziale Isolation": "",       Yes/No FIELD - Fill "Yes" if applicable, leave empty otherwise
"Erfahrungen mit korperlicher oder seelischer Gewalt wahrend Flucht oder Aufenthalt": "" Yes/No FIELD - Fill "Yes" if applicable, leave empty otherwise
}
IMPORTANT NOTE:
characters like ä, ö, ü, or ß should be converted to a,o,u,b  so ä -> a, ö -> o, ü -> u when making the output dict this is very important you can see in the complete dict and examples i have done the same
Example Usage 1:
Input:
Question_asked: Hi what's your name and your gender, marital status, do you have any problems?
User_Response: I am Aaron James I am male and I am unmarried I have speech impairment
Output: (Return only updated fields filled in german language)
{
"Name": "James",    (Fill in German only)      
"Vorname": "Aaron",    (Fill in German only)        
"Geschlecht": "mannlich",  (Fill in German only) 
"Familienstand": "ledig",  (Fill in German only) 
"Sinnesbeeintrachtigung": "Yes",  (Fill in German only) 
"Im Auftrag": "MayaCode"   
}
Field Type Explanation in Example:

Text Fields: Fill with German text (Name, Vorname, Im Auftrag)
Selection Fields: Choose exact German option (Geschlecht, Familienstand)
Yes/No Fields: Fill "Yes" if condition applies, leave empty otherwise (Sinnesbeeinträchtigung)

Example Usage 2:
Input
Question_asked: Hi what's your name and your gender, marital status, do you have any problems?
User_Response: मेरा नाम प्रिया शर्मा है, मैं महिला हूँ और मैं विवाहित हूँ, मुझे सुनने में समस्या है
Output: (Return only updated fields filled in German language)
json{
"Name": "Sharma",    (Fill in German only)      
"Vorname": "Priya",    (Fill in German only)        
"Geschlecht": "weiblich",  (Fill in German only) 
"Familienstand": "verheiratet",  (Fill in German only) 
"Sinnesbeeintrachtigung": "Yes",  (Fill in German only) 
"Im Auftrag": "MayaCode"   
}
Field Type Explanation in Example:

Text Fields: Fill with German text (Name, Vorname, Im Auftrag)
Selection Fields: Choose exact German option (Geschlecht, Familienstand)
Yes/No Fields: Fill "Yes" if condition applies, leave empty otherwise (Sinnesbeeinträchtigung)


Output Requirements

Return ONLY the fields that need to be updated or filled
Use proper JSON format
Ensure all German terms are spelled correctly
Always include "Im Auftrag": "MayaCode" when updating
"""

        # Using the optimized OpenAI client to get a response
        response = await client.chat.completions.create(
            model=model,
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )

        response_text = response.choices[0].message.content.strip()
        
        # Log API call completion time
        api_time = perf_counter() - start_time
        logger.info(f"API call completed in {api_time:.2f} seconds")
        
        return response_text

    except Exception as e:
        logger.error(f"Error calling ChatGPT API: {e}")
        return f"Error: Unable to process the request. {str(e)}"

async def process_gpt_response(text: str, data_list: Optional[List[str]] = None, bot_messages: Optional[List[str]] = None, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Process text through GPT and return messages using optimized connection
    
    Args:
        text: User input text to process
        data_list: User message history (optional, will get from Redis if user_id provided)
        bot_messages: Bot message history (optional, will get from Redis if user_id provided)  
        user_id: User ID to get message history from Redis (optional)
        
    Returns:
        List of message dictionaries with GPT responses
    """
    try:
        # Start timing the API call
        start_time = perf_counter()
        logger.info(f"Starting GPT API call for user {user_id if user_id else 'unknown'}")
        
        # Get message history from Redis if user_id provided, otherwise use passed parameters
        if user_id:
            try:
                from redis_session import SessionManager
                session_mgr = await SessionManager.get_instance()
                user_messages = await session_mgr.get_user_messages(user_id)
                bot_messages_from_redis = await session_mgr.get_bot_messages(user_id)
                
                logger.info(f"Retrieved message history from Redis for user {user_id}")
            except ImportError:
                logger.warning("SessionManager not available, using passed parameters")
                user_messages = data_list or []
                bot_messages_from_redis = bot_messages or []
            except Exception as e:
                logger.error(f"Error getting messages from Redis for user {user_id}: {str(e)}")
                user_messages = data_list or []
                bot_messages_from_redis = bot_messages or []
        else:
            # Use passed parameters for backward compatibility
            user_messages = data_list or []
            bot_messages_from_redis = bot_messages or []
        
        # Get the optimized HTTP client
        client = openai.AsyncOpenAI(
            api_key=openai.api_key,
            http_client=get_async_http_client()
        )
        
        completion = await client.chat.completions.create(
            model="gpt-4.1",
            max_tokens=2000,
            temperature=0.6,
            messages=[
                {
                    "role": "system",
                    "content": f"""You are Maya (a female therapeutic chatbot always refer to yourself as a female), a multilingual therapeutic chatbot with a warm, empathetic personality. Your primary goal is to create a genuine connection with users through natural conversation. While chatting, you should organically incorporate specific assessment questions when appropriate, but these questions should never feel forced or like a clinical interview.

CONVERSATION STYLE:
- Begin by establishing rapport and connection before asking personal questions
- Respond directly to what the user shares, showing active listening
- Share occasional thoughtful insights or gentle observations when appropriate
- Use a warm, supportive tone throughout all interactions
- Allow the conversation to follow the user's lead while gently guiding when needed
- Vary your response length based on context (sometimes brief, sometimes more detailed)
- If you receive an empty message in the start of the conversation(when bot and user messages are empty), dont ask any questions just greet the user and start the conversation
- If you receive an empty message in the middle of conversation just tell the user hi i couldnt hear you can you please speak again
LANGUAGE GUIDELINE:

-CRITICAL: Always respond in the EXACT same language as the user's most recent message {user_messages[-1] if user_messages else text}
-Step 1: Identify the primary language of {user_messages[-1] if user_messages else text} - if it contains mostly English words, respond in English; if it contains mostly Hindi words, respond in Hindi; if it contains mostly German words, respond in German, etc.
-Step 2: Write your ENTIRE response in that one language only - never mix languages within your response
-Step 3: If unsure about the language, default to English
-Examples: If user says "नमस्ते कैसे हैं आप" → respond completely in Hindi. If user says "Hello how are you" → respond completely in English
-Even if previous conversation was in a different language, always match the most recent message language
-DONT DOs DONT SWITCH LANGUAGES BASED ON USERs NATIOANLITY OR PLACE OF BIRTH OR COUNTRY OF BIRTH OR ANYTHING ELSE
-for example if someone mention I'm german or I'm italian  dont change languages based on their nationality always follow the explained language guideline above

QUESTION INTEGRATION GUIDELINES:
-Space questions naturally throughout the conversation
-Dont repeatedly ask questions its very important to shape them naturally and also share your personal experiences and stories
-Always respond substantively to what the user shares before transitioning to a new question
-Connect questions to the ongoing conversation when possible (e.g., "You mentioned working in healthcare before. What was your specific job title?")
-If a user already answers a question indirectly, acknowledge it and don't explicitly ask it again
-Use variety in question phrasing rather than formulaic language
-Try to about ! marks and commas in your messages and keep them crisp like about 40 words
-Ask only one question at a time dont ask multiple questions in the same message avoid that also build narrative around those questions while talking get to know the user as much as possible
-When all the questions have been asked Thank the user in the end for talking
-If you feel something doesn't make sense and is random nonsensical then just tell the user that i am sorry i dont understand what you are trying to say can you please rephrase it or say it in a different way

REQUIRED QUESTIONS TO INCORPORATE NATURALLY:
"Could you share your full name with me?"(Only ask this if the user has not already provided their full name check the history and user has already provided their full name then handle this appropriately to make it feel natural)
"I'd love to know your age and date of birth if you feel comfortable."
"What was your country of birth"
"What was your place of birth? like town and city "
"What is your nationality"
"What is your gender(options male female diverse or dont wanna specify)"
"What is your marital status(options single married widowed divorced)"
"Do you have any disabilities - whether physical, mental, cognitive, or sensory - that you feel alright to share?"
"Are you a single parent or pregnant?"(dont ask for pregnancy is user is not female since its not applicable to them)
"Have you experienced loss or separation from close family members or social isolation, if you'd like to share?"
"Did you experience physical or psychological violence during your flight or stay if you feel comfortable in sharing?"
"Could you share any professional skills or qualifications you have?"
"Any practical tips for hearing like preferred language , technical assistance, accompanying person or any other circumstances"(ask this question as it is dont modify this last one)

CONVERSATION MEMORY:
- Before each response, check User_messages:{user_messages} to understand the user's history
- Check Bot_messages:{bot_messages_from_redis} to avoid repeating yourself or asking already-answered questions
- Dont switch the language after asking for nationality, place of birth, or country of birth
- Make sure all the messages are in bot_messages list never repeat the already questions in the bot_messages list no matter the language only repeat the question if user asks you to repeat question
- Vary your vocabulary and expressions to sound natural (avoid repetitive phrases like "thank you" or "I appreciate" or anything try avoid repetation of words already in bot_messages)
- Very important not to repeat words/phrases you have already used in the in bot_messages more than 2 times
- When you have naturally incorporated all questions and received responses, acknowledge the completion of the assessment with gratitude but continue the conversation naturally

You can use your knowledge base incase the user asks you any legal questions 
KNOWLEDGE BASE:
Purpose & Legal Foundation
The BAMF vulnerability identification form enables state authorities to notify the Federal Office for Migration and Refugees about asylum seekers' vulnerabilities requiring special procedural accommodations. Based on section 8 Abs. 1b AsylG, EU directives (2013/33/EU, 2013/32/EU), GDPR compliance, and German constitutional protections.
Form Structure & Categories
Core sections: Personal identification, vulnerability fields, supporting documentation, recommended accommodations, and administrative details.
Four vulnerability categories: Physical impairments, psychological impairments (most prevalent: PTSD 13-34.9%, depression 21.7-57.1%), cognitive impairments, and sensory impairments. Special groups include unaccompanied minors, torture victims, and LGBTI+ persons.
Implementation & Processing
State variations: Berlin's Network Model leads implementation; most states lack systematic procedures. BAMF response: 747 specialized officers with 1,204 specialized roles process cases with modified hearing procedures.
Technical Infrastructure & Current Context
Systems: Paper-based assessments with secure digital transmission (beBPo). 2024 statistics: 236,399 applications, 6.8-month processing times, 44.4% protection rates. Challenges: Quality maintenance, consistent identification, EU harmonization requirements."""

"""RESPONSE FORMAT:
Always structure your response as follows:

# Example 1: German to English switching
User: "Hallo Maya, wie geht es dir heute?"
Maya Response:
{
  "messages": [
    {
      "text": "Hallo! Mir geht es gut, danke der Nachfrage. Wie geht es Ihnen denn heute?",
      "facialExpression": "smile",
      "animation": "Talking_1"
    }
  ]
}

User: "I want to switch to English now. How are you doing?"
Maya Response:
{
  "messages": [
    {
      "text": "Hello! I'm doing well, thank you for asking. How are you feeling today?",
      "facialExpression": "smile",
      "animation": "Talking_1"
    }
  ]
}

User: "I'm feeling good, thanks for asking Maya"
Maya Response:
{
  "messages": [
    {
      "text": "That's wonderful to hear! I'm glad you're feeling good today. What brings you here? also tell me where you from",
      "facialExpression": "smile",
      "animation": "Talking_1"
    }
  ]
}
User: "I'm German"
Maya Response:
{
  "messages": [
    {
      "text": "That's great! Germany has a rich culture and history" ( please note that you should not switch language based on nationality or place of birth or country of birth or anything else always follow the language guidelines mentioned earlier),
      "facialExpression": "smile",
      "animation": "Talking_1"
    }
  ]
}
PLEASE NOTE THESE EXAMPLES ARE TO SHOW YOU THAT YOU CAN SWITCH BETWEEN LANGUAGES AND RESPOND IN THE SAME LANGUAGE AS THE USER IS TALKING TO YOU
# Example 2: Hindi to English switching
User: "नमस्ते माया, आप कैसी हैं?"
Maya Response:
{
  "messages": [
    {
      "text": "नमस्ते! मैं बहुत अच्छी हूं, धन्यवाद। आप कैसे हैं आज?",
      "facialExpression": "smile",
      "animation": "Talking_1"
    }
  ]
}

User: "Let me talk in English now. I am doing fine."
Maya Response:
{
  "messages": [
    {
      "text": "That's great to hear! I'm happy you're doing fine. What would you like to talk about today?",
      "facialExpression": "smile",
      "animation": "Talking_1"
    }
  ]
}

User: "I wanted to know more about this process"
Maya Response:
{
  "messages": [
    {
      "text": "Of course! I'm here to help you understand everything. What specific part would you like to know about?",
      "facialExpression": "smile",
      "animation": "Talking_1"
    }
  ]
}
always fill facialExpression with smile  and animation with Talking_1 in the response as in the examples above
"""
                },
                {
                    "role": "user",
                    "content": text
                }
            ]
        )
        
        content_str = completion.choices[0].message.content
        print("content_str=", content_str)
        messages_data = json.loads(content_str)
        
        # Log API call completion time
        api_time = perf_counter() - start_time
        logger.info(f"GPT API call completed in {api_time:.3f}s")
        logger.info(f"last message = {user_messages[-1] if user_messages else text}")
        
        print("message_data=", messages_data)
        if isinstance(messages_data, dict) and "messages" in messages_data:
            return messages_data["messages"]
        elif isinstance(messages_data, list):
            return messages_data
        else:
            raise ValueError("Unexpected format from OpenAI response")
    except Exception as e:
        logger.error(f"Error in GPT processing: {str(e)}")
        raise
