import os
import json
import time
from redis.asyncio.cluster import RedisCluster
from typing import Optional, Any, Dict
import json
import os
from datetime import datetime
import ssl

class SessionManager:
    _instance = None
    _redis = None
    
    @classmethod
    async def get_instance(cls):
        """Get singleton instance of SessionManager"""
        if cls._instance is None:
            cls._instance = SessionManager()
            await cls._instance._initialize()
        return cls._instance
    
    async def _initialize(self):
            redis_host = os.getenv(
                "REDIS_HOST",
                "mayarediscache-injfba.serverless.use1.cache.amazonaws.com"
            )
            redis_pw = os.getenv("REDIS_PASSWORD", None)

            # Create SSL context to validate AWS certs

            # Use RedisCluster for cluster mode
            self._redis = RedisCluster.from_url(
                f"rediss://{redis_host}:6379",
                password=redis_pw,
                decode_responses=True,
                ssl=True,
            )

            # Validate connection and routing with both read & write tests
            try:
                pong = await self._redis.ping()
                print("✅ Connected to cluster:", pong)
            except Exception as e:
                print("❌ Initialization error:", repr(e))
                raise
    
    async def set_user_data(self, user_id: str, key: str, data: Any, expiry: int = 3600) -> bool:
        """Store user-specific data in Redis
        
        Args:
            user_id: User's unique identifier
            key: Data key
            data: Data to store (will be JSON serialized if dict or list)
            expiry: Expiration time in seconds (default 1 hour)
            
        Returns:
            bool: Success or failure
        """
        session_key = f"user:{user_id}:{key}"
        
        # Convert complex data types to JSON
        if isinstance(data, (dict, list)):
            data = json.dumps(data)
            
        return await self._redis.set(session_key, data, ex=expiry)
    
    async def get_user_data(self, user_id: str, key: str) -> Any:
        """Retrieve user-specific data from Redis
        
        Args:
            user_id: User's unique identifier
            key: Data key
            
        Returns:
            The data value, or None if not found
        """
        session_key = f"user:{user_id}:{key}"
        data = await self._redis.get(session_key)
        
        if data is None:
            return None
            
        # Try to parse as JSON, return raw string if not valid JSON
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return data
    
    async def delete_user_data(self, user_id: str, key: str) -> int:
        """Delete user-specific data from Redis
        
        Args:
            user_id: User's unique identifier
            key: Data key
            
        Returns:
            int: Number of keys deleted (0 or 1)
        """
        session_key = f"user:{user_id}:{key}"
        return await self._redis.delete(session_key)
    
    async def update_activity(self, user_id: str) -> None:
        """Update user's last activity timestamp
        
        Args:
            user_id: User's unique identifier
        """
        await self._redis.set(
            f"user:{user_id}:last_active", 
            time.time(),
            ex=86400  # 24 hours expiry
        )
    
    async def get_user_messages(self, user_id: str) -> list:
        """Get user's message history
        
        Args:
            user_id: User's unique identifier
            
        Returns:
            list: List of user messages
        """
        return await self.get_user_data(user_id, "user_messages") or []
    
    async def get_bot_messages(self, user_id: str) -> list:
        """Get bot's message history for this user
        
        Args:
            user_id: User's unique identifier
            
        Returns:
            list: List of bot messages for this user
        """
        return await self.get_user_data(user_id, "bot_messages") or []
    
    async def store_user_message(self, user_id: str, message: str) -> None:
        """Add a user message to history
        
        Args:
            user_id: User's unique identifier
            message: User message to store
        """
        messages = await self.get_user_messages(user_id)
        messages.append(message)
        await self.set_user_data(user_id, "user_messages", messages)
    
    async def store_bot_message(self, user_id: str, message: str) -> None:
        """Add a bot message to history
        
        Args:
            user_id: User's unique identifier
            message: Bot message to store
        """
        messages = await self.get_bot_messages(user_id)
        messages.append(message)
        await self.set_user_data(user_id, "bot_messages", messages)
    
    async def get_user_form_data(self, user_id: str) -> Dict:
        """Get user's form data
        
        Args:
            user_id: User's unique identifier
            
        Returns:
            Dict: User's form data or template if not found
        """
        from form_template import data_dict
        
        # Get user's form data or initialize with template
        form_data = await self.get_user_data(user_id, "form_data")
        if not form_data:
            form_data = data_dict.copy()
            await self.set_user_data(user_id, "form_data", form_data)
            
        return form_data
    
    async def update_user_form_data(self, user_id: str, form_data: Dict) -> None:
        """Update user's form data
        
        Args:
            user_id: User's unique identifier
            form_data: Updated form data
        """
        await self.set_user_data(user_id, "form_data", form_data, expiry=604800)  # 7 days
        
    async def set_user_language(self, user_id: str, language: str) -> None:
        """Set user's preferred language
        
        Args:
            user_id: User's unique identifier
            language: Language code
        """
        await self.set_user_data(user_id, "language", language, expiry=604800)  # 7 days
    
    async def get_user_language(self, user_id: str) -> str:
        """Get user's preferred language
        
        Args:
            user_id: User's unique identifier
            
        Returns:
            str: Language code or default "en"
        """
        return await self.get_user_data(user_id, "language") or None
        
    async def set_user_voice_id(self, user_id: str, voice_id: str) -> None:
        """Set user's preferred voice ID
        
        Args:
            user_id: User's unique identifier
            voice_id: Voice ID for text-to-speech
        """
        await self.set_user_data(user_id, "voice_id", voice_id, expiry=604800)  # 7 days
        
    async def get_user_voice_id(self, user_id: str) -> str:
        """Get user's preferred voice ID
        
        Args:
            user_id: User's unique identifier
            
        Returns:
            str: Voice ID or default voice
        """
        return await self.get_user_data(user_id, "voice_id") or "9BWtsMINqrJLrRacOk9x"  # Default voice ID
        
    async def get_user_persona(self, user_id: str) -> Dict:
        """Get user's persona data
        
        Args:
            user_id: User's unique identifier
            
        Returns:
            Dict: User persona data
        """
        # Get existing persona or initialize from template
        user_persona = await self.get_user_data(user_id, "persona")
        if user_persona is not None:
            return user_persona
        
        # Load the default persona template
        try:

            persona_path = os.path.join(os.path.dirname(__file__), 'userpersona.json')
            with open(persona_path, 'r') as f:
                default_persona = json.load(f)
            
            # Store this template for the user
            await self.set_user_persona(user_id, default_persona)
            return default_persona
        except Exception as e:
            import logging
            logging.error(f"Error loading user persona template: {e}")
            return {}
        
    async def set_user_persona(self, user_id: str, persona_data: Dict) -> None:
        """Set user's persona data
        
        Args:
            user_id: User's unique identifier
            persona_data: User's persona data
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            logger.info(f"SessionManager: Setting persona data for user {user_id} in Redis")
            await self.set_user_data(user_id, "persona", persona_data, expiry=604800)  # 7 days
            logger.info(f"SessionManager: Successfully set persona data for user {user_id}")
        except Exception as e:
            logger.error(f"SessionManager: Error setting persona for user {user_id}: {str(e)}", exc_info=True)
            raise
        
    async def update_user_persona(self, user_id: str, persona_data: Dict) -> None:
        """Update user's persona data (alias for set_user_persona for backward compatibility)
        
        Args:
            user_id: User's unique identifier
            persona_data: User's persona data
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            logger.info(f"SessionManager: Updating persona for user {user_id}")
            logger.info(f"SessionManager: Persona data type: {type(persona_data)}, keys: {list(persona_data.keys()) if isinstance(persona_data, dict) else 'Not a dict'}")
            
            await self.set_user_persona(user_id, persona_data)
            
            logger.info(f"SessionManager: Successfully updated persona for user {user_id}")
        except Exception as e:
            logger.error(f"SessionManager: Error updating persona for user {user_id}: {str(e)}", exc_info=True)
            raise
        
    async def update_user_data_dict(self, user_id: str, data_dict: Dict) -> None:
        """Update the entire user data dictionary
        
        Args:
            user_id: User's unique identifier
            data_dict: Complete data dictionary to store
        """
        await self.set_user_data(user_id, "data_dict", data_dict, expiry=604800)  # 7 days
        
    async def get_user_data_dict(self, user_id: str) -> Dict:
        """Get the entire user data dictionary
        
        Args:
            user_id: User's unique identifier
            
        Returns:
            Dict: User data dictionary or empty dict if not found
        """
        # Import here to avoid circular imports
        from form_template import data_dict as template_dict
        
        # Get current data or use template if not found
        return await self.get_user_data(user_id, "data_dict") or template_dict.copy()

    async def save_user_pdf(self, user_id: str, pdf_data: bytes, pdf_filename: str = None) -> bool:
        """Save user's PDF to Redis for persistent storage"""
        try:
            import base64
            
            # Convert PDF bytes to base64 for Redis storage
            pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
            
            # Create PDF metadata
            pdf_info = {
                "filename": pdf_filename or f"filled_form_{user_id}.pdf",
                "data": pdf_base64,
                "timestamp": str(datetime.now()),
                "user_id": user_id
            }
            
            # Save to Redis with consistent key pattern - overwrites previous PDF
            await self._redis.set(
                f"user:{user_id}:saved_pdf",  # Consistent pattern, overwrites each time
                json.dumps(pdf_info),
                ex=86400 * 30  # Expire in 30 days
            )
            
            print(f"PDF saved to Redis for user {user_id}")
            return True
            
        except Exception as e:
            print(f"Error saving PDF to Redis for user {user_id}: {e}")
            return False
