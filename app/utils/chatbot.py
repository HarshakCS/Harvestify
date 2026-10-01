import requests
import json
from typing import Optional, Dict, Any

class OllamaChatbot:
    def __init__(self, model_name: str = "phi3:latest", base_url: str = "http://localhost:11434"):
        """
        Initialize the Ollama chatbot with the specified model.
        
        Args:
            model_name: Name of the Ollama model to use (default: phi3:latest)
            base_url: Base URL of the Ollama API server (default: http://localhost:11434)
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.chat_history = []
    
    def generate_response(self, message: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a response from the chatbot.
        
        Args:
            message: User's message
            system_prompt: Optional system prompt to guide the model's behavior
            
        Returns:
            str: Generated response from the model
        """
        try:
            # Prepare the messages
            messages = []
            
            # Add system prompt if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
                
            # Add chat history
            messages.extend(self.chat_history)
            
            # Add the new user message
            messages.append({"role": "user", "content": message})
            
            # Prepare the request payload
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False
            }
            
            # Make the API request
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            
            # Check for errors
            response.raise_for_status()
            
            # Parse the response
            response_data = response.json()
            
            # Extract the assistant's response
            assistant_message = response_data.get('message', {}).get('content', '').strip()
            
            # Update chat history (keep last 10 messages)
            self.chat_history.append({"role": "user", "content": message})
            self.chat_history.append({"role": "assistant", "content": assistant_message})
            self.chat_history = self.chat_history[-10:]  # Keep last 5 exchanges (10 messages)
            
            return assistant_message
            
        except requests.exceptions.RequestException as e:
            return f"Error communicating with Ollama API: {str(e)}"
        except Exception as e:
            return f"An error occurred: {str(e)}"
    
    def clear_history(self) -> None:
        """Clear the chat history."""
        self.chat_history = []

# Create a global instance of the chatbot
chatbot = OllamaChatbot()
