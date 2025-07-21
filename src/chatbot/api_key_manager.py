"""
API Key Manager for Punch IQ AI
Handles secure loading and management of API keys
"""
import os
import streamlit as st
from pathlib import Path
from src.chatbot.path_manager import get_path_manager


class APIKeyManager:
    """Manages API keys securely"""
    
    def __init__(self):
        self.path_mgr = get_path_manager()
    
    def load_api_key(self, key_name: str) -> str:
        """Load API key from file or environment"""
        # Try to load from file first
        key_file = self.path_mgr.get_path(f"{key_name.lower()}_key")
        if key_file and key_file.exists():
            try:
                return key_file.read_text().strip()
            except Exception as e:
                st.warning(f"⚠️ Could not read {key_name} key from file: {e}")
        
        # Try environment variable
        env_key = f"{key_name.upper()}_API_KEY"
        env_value = os.getenv(env_key)
        if env_value:
            return env_value
        
        return None
    
    def get_mistral_key(self) -> str:
        """Get Mistral API key"""
        return self.load_api_key("mistral")
    
    def get_openai_key(self) -> str:
        """Get OpenAI API key"""
        return self.load_api_key("openai")
    
    def save_api_key(self, key_name: str, key_value: str) -> bool:
        """Save API key to file securely"""
        try:
            key_file = self.path_mgr.get_path(f"{key_name.lower()}_key")
            if key_file:
                # Ensure secret directory exists
                key_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Write key with restricted permissions
                key_file.write_text(key_value)
                
                # Set file permissions (read only for owner)
                if os.name != 'nt':  # Unix-like systems
                    key_file.chmod(0o600)
                
                return True
        except Exception as e:
            st.error(f"❌ Could not save {key_name} key: {e}")
        
        return False
    
    def display_key_status(self):
        """Display the status of API keys"""
        mistral_key = self.get_mistral_key()
        openai_key = self.get_openai_key()
        
        with st.expander("🔑 API Key Status"):
            if mistral_key:
                st.success(f"✅ Mistral API Key: ***{mistral_key[-4:]}")
            else:
                st.warning("⚠️ Mistral API Key: Not found")
            
            if openai_key:
                st.success(f"✅ OpenAI API Key: ***{openai_key[-4:]}")
            else:
                st.info("💡 OpenAI API Key: Not configured (optional)")


# Global instance
api_key_manager = APIKeyManager()


def get_api_key_manager() -> APIKeyManager:
    """Get the global API key manager instance"""
    return api_key_manager
