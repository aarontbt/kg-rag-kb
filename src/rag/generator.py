from typing import List, Dict, Any, Optional
import logging
import subprocess
import json

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """Generate responses using local LLM (Ollama)"""
    
    def __init__(self, 
                 model_name: str = "llama2",
                 temperature: float = 0.7,
                 max_tokens: int = 1000):
        """
        Initialize response generator
        
        Args:
            model_name: Name of the Ollama model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._check_ollama_availability()
    
    def _check_ollama_availability(self):
        """Check if Ollama is available and the model is installed"""
        try:
            # Check if Ollama is running
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                # Silently handle Ollama not running - will be caught when querying
                return
            
            # Check if model is installed
            models_output = result.stdout
            if self.model_name not in models_output:
                # Silently handle missing model - will be caught when querying
                return
            
            # Only log success when everything is working
            pass
            
        except FileNotFoundError:
            # Silently handle Ollama not installed
            pass
        except subprocess.TimeoutExpired:
            # Silently handle timeout
            pass
        except Exception as e:
            # Silently handle other errors
            pass
    
    def generate_response(self, 
                         query: str,
                         context: str,
                         system_prompt: Optional[str] = None) -> str:
        """
        Generate a response based on query and context
        
        Args:
            query: User query
            context: Retrieved document context
            system_prompt: Optional system prompt for the LLM
            
        Returns:
            Generated response
        """
        try:
            # Build the prompt
            full_prompt = self._build_prompt(query, context, system_prompt)
            
            # Call Ollama API
            response = self._call_ollama(full_prompt)
            
            logger.info("Successfully generated response")
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def _build_prompt(self, 
                     query: str,
                     context: str,
                     system_prompt: Optional[str] = None) -> str:
        """
        Build the complete prompt for the LLM
        
        Args:
            query: User query
            context: Retrieved document context
            system_prompt: Optional system prompt
            
        Returns:
            Complete prompt string
        """
        if not system_prompt:
            system_prompt = (
                "You are a helpful AI assistant that answers questions based on the provided context. "
                "Use only the information from the context to answer the question. "
                "If the context doesn't contain enough information to answer the question, "
                "say so clearly. Provide concise and accurate answers."
            )
        
        prompt = f"""{system_prompt}

Context:
{context}

Question: {query}

Answer:"""
        
        return prompt
    
    def _call_ollama(self, prompt: str) -> str:
        """
        Call Ollama to generate response
        
        Args:
            prompt: The prompt to send to Ollama
            
        Returns:
            Generated response text
        """
        try:
            # Use ollama run command to generate response
            result = subprocess.run(
                ["ollama", "run", self.model_name],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=120  # 2 minutes timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Ollama error: {result.stderr}")
                return f"Ollama error: {result.stderr}"
            
            response = result.stdout.strip()
            
            return response
            
        except subprocess.TimeoutExpired:
            logger.error("Ollama call timed out")
            return "Response generation timed out"
        except Exception as e:
            logger.error(f"Error calling Ollama: {str(e)}")
            return f"Error calling Ollama: {str(e)}"
    
    def generate_simple_response(self, query: str) -> str:
        """
        Generate a simple response without context (for general conversation)
        
        Args:
            query: User query
            
        Returns:
            Generated response
        """
        try:
            system_prompt = (
                "You are a helpful AI assistant. Provide helpful and accurate responses to user questions."
            )
            
            return self.generate_response(query, "", system_prompt)
            
        except Exception as e:
            logger.error(f"Error generating simple response: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def is_available(self) -> bool:
        """
        Check if the generator is available
        
        Returns:
            True if Ollama is available and model is installed
        """
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                return False
            
            return self.model_name in result.stdout
            
        except:
            return False
