"""
Vayu LLM Service
Wrapper for OpenRouter API using the OpenAI client.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import Optional, List, Dict, AsyncGenerator
from openai import AsyncOpenAI
from colorama import Fore, Style

import config
from src.interfaces import ILLMService


class LLMService(ILLMService):
    """
    LLM service using OpenRouter API.
    Supports streaming responses for low-latency voice applications.
    """
    
    def __init__(self, model: str = None, system_prompt: str = None):
        """
        Initialize the LLM service with OpenRouter.
        
        Args:
            model: Model identifier (default from config)
            system_prompt: System prompt for the assistant (default from config)
        """
        if not config.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        
        self.client = AsyncOpenAI(
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL
        )
        
        self.model = model or config.LLM_MODEL
        self.system_prompt = system_prompt or config.SYSTEM_PROMPT
        self.max_tokens = config.LLM_MAX_TOKENS
        self.temperature = config.LLM_TEMPERATURE
        
        # Conversation history for context
        self.conversation_history: List[Dict[str, str]] = []
        
        print(f"{Fore.GREEN}[LLM] Initialized with model: {self.model}{Style.RESET_ALL}")
    
    def _build_messages(self, prompt: str, context: Optional[List[Dict]] = None) -> List[Dict[str, str]]:
        """
        Build the messages array for the API call.
        
        Args:
            prompt: User's current prompt
            context: Optional conversation context
            
        Returns:
            List of message dictionaries
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add context if provided
        if context:
            messages.extend(context)
        else:
            # Use internal conversation history
            messages.extend(self.conversation_history)
        
        # Add current user prompt
        messages.append({"role": "user", "content": prompt})
        
        return messages
    
    async def generate(self, prompt: str, context: Optional[List[Dict]] = None) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User's input prompt
            context: Optional conversation context
            
        Returns:
            Generated response string
        """
        messages = self._build_messages(prompt, context)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            
            assistant_message = response.choices[0].message.content
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            
            # Trim history if too long
            self._trim_history()
            
            return assistant_message
            
        except Exception as e:
            print(f"{Fore.RED}[LLM] Error generating response: {e}{Style.RESET_ALL}")
            raise
    
    async def generate_stream(self, prompt: str, context: Optional[List[Dict]] = None) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response from the LLM.
        
        Yields partial responses as they're generated for low-latency TTS.
        
        Args:
            prompt: User's input prompt
            context: Optional conversation context
            
        Yields:
            Partial response strings as they're generated
        """
        messages = self._build_messages(prompt, context)
        
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True,
            )
            
            full_response = ""
            sentence_buffer = ""
            sentence_delimiters = {'.', '!', '?', '\n', ':', ';', ',', '—'}
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    sentence_buffer += content
                    
                    # Check if we have a complete sentence to yield
                    # This allows TTS to start processing earlier
                    for delimiter in sentence_delimiters:
                        if delimiter in sentence_buffer:
                            parts = sentence_buffer.split(delimiter, 1)
                            if len(parts) > 1:
                                complete_part = parts[0] + delimiter
                                sentence_buffer = parts[1]
                                yield complete_part.strip()
            
            # Yield any remaining content
            if sentence_buffer.strip():
                yield sentence_buffer.strip()
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": full_response})
            self._trim_history()
            
        except Exception as e:
            print(f"{Fore.RED}[LLM] Error in streaming response: {e}{Style.RESET_ALL}")
            raise
    
    def _trim_history(self, max_turns: int = 10):
        """
        Trim conversation history to prevent context overflow.
        
        Args:
            max_turns: Maximum number of turns to keep (1 turn = user + assistant)
        """
        max_messages = max_turns * 2
        if len(self.conversation_history) > max_messages:
            self.conversation_history = self.conversation_history[-max_messages:]
    
    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
        print(f"{Fore.CYAN}[LLM] Conversation history cleared{Style.RESET_ALL}")
    
    def set_model(self, model: str):
        """Change the LLM model."""
        self.model = model
        print(f"{Fore.CYAN}[LLM] Model changed to: {self.model}{Style.RESET_ALL}")
    
    def set_system_prompt(self, prompt: str):
        """Change the system prompt."""
        self.system_prompt = prompt
        print(f"{Fore.CYAN}[LLM] System prompt updated{Style.RESET_ALL}")


if __name__ == "__main__":
    import asyncio
    
    async def test_llm():
        print("Testing LLM Service...")
        llm = LLMService()
        
        # Test non-streaming
        print("\n--- Non-streaming test ---")
        response = await llm.generate("Hello, who are you?")
        print(f"Response: {response}")
        
        # Test streaming
        print("\n--- Streaming test ---")
        async for chunk in llm.generate_stream("Tell me a very short joke."):
            print(f"Chunk: {chunk}")
    
    asyncio.run(test_llm())
