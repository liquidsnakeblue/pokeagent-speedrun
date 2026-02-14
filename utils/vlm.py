from io import BytesIO
from PIL import Image
import os
import base64
import random
import time
import logging
from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any, Optional
import numpy as np

# Set up module logging
logger = logging.getLogger(__name__)

# Import LLM logger
from utils.llm_logger import log_llm_interaction, log_llm_error

# Define the retry decorator with exponential backoff
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (Exception,),
):
    """Retry a function with exponential backoff."""
    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay
        while True:
            try:
                return func(*args, **kwargs)
            except errors as e:
                num_retries += 1
                if num_retries > max_retries:
                    raise Exception(f"Maximum number of retries ({max_retries}) exceeded.")
                # Increase the delay with exponential factor and random jitter
                delay *= exponential_base * (1 + jitter * random.random())
                time.sleep(delay)
            except Exception as e:
                raise e
    return wrapper

class VLMBackend(ABC):
    """Abstract base class for VLM backends"""
    
    @abstractmethod
    def get_query(self, img: Union[Image.Image, np.ndarray], text: str, module_name: str = "Unknown") -> str:
        """Process an image and text prompt"""
        pass
    
    @abstractmethod
    def get_text_query(self, text: str, module_name: str = "Unknown") -> str:
        """Process a text-only prompt"""
        pass

class OpenAIBackend(VLMBackend):
    """OpenAI API backend"""
    
    def __init__(self, model_name: str, **kwargs):
        try:
            import openai
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package not found. Install with: pip install openai")
        
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("Error: OpenAI API key is missing! Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.errors = (openai.RateLimitError,)
    
    @retry_with_exponential_backoff
    def _call_completion(self, messages):
        """Calls the completions.create method with exponential backoff."""
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )
    
    def get_query(self, img: Union[Image.Image, np.ndarray], text: str, module_name: str = "Unknown") -> str:
        """Process an image and text prompt using OpenAI API"""
        start_time = time.time()
        
        # Handle both PIL Images and numpy arrays
        if hasattr(img, 'convert'):  # It's a PIL Image
            image = img
        elif hasattr(img, 'shape'):  # It's a numpy array
            image = Image.fromarray(img)
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
        
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]
        }]
        
        try:
            response = self._call_completion(messages)
            result = response.choices[0].message.content
            duration = time.time() - start_time
            
            # Extract token usage if available
            token_usage = {}
            if hasattr(response, 'usage'):
                token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            
            # Log the interaction
            log_llm_interaction(
                interaction_type=f"openai_{module_name}",
                prompt=text,
                response=result,
                duration=duration,
                metadata={"model": self.model_name, "backend": "openai", "has_image": True, "token_usage": token_usage},
                model_info={"model": self.model_name, "backend": "openai"}
            )
            
            return result
        except Exception as e:
            duration = time.time() - start_time
            log_llm_error(
                interaction_type=f"openai_{module_name}",
                prompt=text,
                error=str(e),
                metadata={"model": self.model_name, "backend": "openai", "duration": duration, "has_image": True}
            )
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def get_text_query(self, text: str, module_name: str = "Unknown") -> str:
        """Process a text-only prompt using OpenAI API"""
        start_time = time.time()
        
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": text}]
        }]
        
        try:
            response = self._call_completion(messages)
            result = response.choices[0].message.content
            duration = time.time() - start_time
            
            # Extract token usage if available
            token_usage = {}
            if hasattr(response, 'usage'):
                token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            
            # Log the interaction
            log_llm_interaction(
                interaction_type=f"openai_{module_name}",
                prompt=text,
                response=result,
                duration=duration,
                metadata={"model": self.model_name, "backend": "openai", "has_image": False, "token_usage": token_usage},
                model_info={"model": self.model_name, "backend": "openai"}
            )
            
            return result
        except Exception as e:
            duration = time.time() - start_time
            log_llm_error(
                interaction_type=f"openai_{module_name}",
                prompt=text,
                error=str(e),
                metadata={"model": self.model_name, "backend": "openai", "duration": duration, "has_image": False}
            )
            logger.error(f"OpenAI API error: {e}")
            raise

class OpenRouterBackend(VLMBackend):
    """OpenRouter API backend"""
    
    def __init__(self, model_name: str, **kwargs):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package not found. Install with: pip install openai")
        
        self.model_name = model_name
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        
        if not self.api_key:
            raise ValueError("Error: OpenRouter API key is missing! Set OPENROUTER_API_KEY environment variable.")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
    
    @retry_with_exponential_backoff
    def _call_completion(self, messages):
        """Calls the completions.create method with exponential backoff."""
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )
    
    def get_query(self, img: Union[Image.Image, np.ndarray], text: str, module_name: str = "Unknown") -> str:
        """Process an image and text prompt using OpenRouter API"""
        # Handle both PIL Images and numpy arrays
        if hasattr(img, 'convert'):  # It's a PIL Image
            image = img
        elif hasattr(img, 'shape'):  # It's a numpy array
            image = Image.fromarray(img)
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
        
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]
        }]
        
        # Log the prompt
        prompt_preview = text[:2000] + "..." if len(text) > 2000 else text
        logger.info(f"[{module_name}] OPENROUTER VLM IMAGE QUERY:")
        logger.info(f"[{module_name}] PROMPT: {prompt_preview}")
        
        response = self._call_completion(messages)
        result = response.choices[0].message.content
        
        # Log the response
        result_preview = result[:1000] + "..." if len(result) > 1000 else result
        logger.info(f"[{module_name}] RESPONSE: {result_preview}")
        logger.info(f"[{module_name}] ---")
        
        return result
    
    def get_text_query(self, text: str, module_name: str = "Unknown") -> str:
        """Process a text-only prompt using OpenRouter API"""
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": text}]
        }]
        
        # Log the prompt
        prompt_preview = text[:2000] + "..." if len(text) > 2000 else text
        logger.info(f"[{module_name}] OPENROUTER VLM TEXT QUERY:")
        logger.info(f"[{module_name}] PROMPT: {prompt_preview}")
        
        response = self._call_completion(messages)
        result = response.choices[0].message.content
        
        # Log the response
        result_preview = result[:1000] + "..." if len(result) > 1000 else result
        logger.info(f"[{module_name}] RESPONSE: {result_preview}")
        logger.info(f"[{module_name}] ---")
        
        return result

class LocalHuggingFaceBackend(VLMBackend):
    """Local HuggingFace transformers backend with bitsandbytes optimization"""
    
    def __init__(self, model_name: str, device: str = "auto", load_in_4bit: bool = False, **kwargs):
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
            from PIL import Image
        except ImportError as e:
            raise ImportError(f"Required packages not found. Install with: pip install torch transformers bitsandbytes accelerate. Error: {e}")
        
        self.model_name = model_name
        self.device = device
        self.torch = torch
        
        logger.info(f"Loading local VLM model: {model_name}")
        
        # Configure quantization if requested
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            logger.info("Using 4-bit quantization with bitsandbytes")
        
        # Load processor and model
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map=device if device != "auto" else "auto",
                torch_dtype=torch.float16 if not load_in_4bit else None,
                trust_remote_code=True
            )
            
            if device != "auto" and not load_in_4bit:
                self.model = self.model.to(device)
                
            logger.info(f"Model loaded successfully on {device}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def _generate_response(self, inputs: Dict[str, Any], text: str, module_name: str) -> str:
        """Generate response using the local model"""
        try:
            start_time = time.time()
            
            # Log the prompt
            prompt_preview = text[:2000] + "..." if len(text) > 2000 else text
            logger.info(f"[{module_name}] LOCAL HF VLM QUERY:")
            logger.info(f"[{module_name}] PROMPT: {prompt_preview}")
            
            with self.torch.no_grad():
                # Ensure all inputs are on the correct device
                if hasattr(self.model, 'device'):
                    device = self.model.device
                elif hasattr(self.model, 'module') and hasattr(self.model.module, 'device'):
                    device = self.model.module.device
                else:
                    device = next(self.model.parameters()).device
                
                # Move inputs to device if needed
                inputs_on_device = {}
                for k, v in inputs.items():
                    if hasattr(v, 'to'):
                        inputs_on_device[k] = v.to(device)
                    else:
                        inputs_on_device[k] = v
                
                generated_ids = self.model.generate(
                    **inputs_on_device,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
                
                # Decode the response
                generated_text = self.processor.decode(generated_ids[0], skip_special_tokens=True)
                
                # Extract only the generated part (remove the prompt)
                if text in generated_text:
                    result = generated_text.split(text)[-1].strip()
                else:
                    result = generated_text.strip()
            
            # Log the interaction
            duration = time.time() - start_time
            log_llm_interaction(
                interaction_type=f"local_{module_name}",
                prompt=text,
                response=result,
                duration=duration,
                metadata={"model": self.model_name, "backend": "local", "has_image": "images" in inputs},
                model_info={"model": self.model_name, "backend": "local"}
            )
            
            # Log the response
            result_preview = result[:1000] + "..." if len(result) > 1000 else result
            logger.info(f"[{module_name}] RESPONSE: {result_preview}")
            logger.info(f"[{module_name}] ---")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def get_query(self, img: Union[Image.Image, np.ndarray], text: str, module_name: str = "Unknown") -> str:
        """Process an image and text prompt using local HuggingFace model"""
        # Handle both PIL Images and numpy arrays
        if hasattr(img, 'convert'):  # It's a PIL Image
            image = img
        elif hasattr(img, 'shape'):  # It's a numpy array
            image = Image.fromarray(img)
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
        
        # Prepare messages with proper chat template format
        messages = [
            {"role": "user",
             "content": [
                 {"type": "image", "image": image},
                 {"type": "text", "text": text}
             ]}
        ]
        formatted_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=formatted_text, images=image, return_tensors="pt")
        
        return self._generate_response(inputs, text, module_name)
    
    def get_text_query(self, text: str, module_name: str = "Unknown") -> str:
        """Process a text-only prompt using local HuggingFace model"""
        # For text-only queries, use simple text format without image
        messages = [
            {"role": "user", "content": text}
        ]
        formatted_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=formatted_text, return_tensors="pt")
        
        return self._generate_response(inputs, text, module_name)

class LegacyOllamaBackend(VLMBackend):
    """Legacy Ollama backend for backward compatibility"""
    
    def __init__(self, model_name: str, port: int = 8010, **kwargs):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package not found. Install with: pip install openai")
        
        self.model_name = model_name
        self.port = port
        self.client = OpenAI(api_key='', base_url=f'http://localhost:{port}/v1')
    
    @retry_with_exponential_backoff
    def _call_completion(self, messages):
        """Calls the completions.create method with exponential backoff."""
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )
    
    def get_query(self, img: Union[Image.Image, np.ndarray], text: str, module_name: str = "Unknown") -> str:
        """Process an image and text prompt using legacy Ollama backend"""
        # Handle both PIL Images and numpy arrays
        if hasattr(img, 'convert'):  # It's a PIL Image
            image = img
        elif hasattr(img, 'shape'):  # It's a numpy array
            image = Image.fromarray(img)
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
        
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]
        }]
        
        # Log the prompt
        prompt_preview = text[:2000] + "..." if len(text) > 2000 else text
        logger.info(f"[{module_name}] OLLAMA VLM IMAGE QUERY:")
        logger.info(f"[{module_name}] PROMPT: {prompt_preview}")
        
        response = self._call_completion(messages)
        result = response.choices[0].message.content
        
        # Log the response
        result_preview = result[:1000] + "..." if len(result) > 1000 else result
        logger.info(f"[{module_name}] RESPONSE: {result_preview}")
        logger.info(f"[{module_name}] ---")
        
        return result
    
    def get_text_query(self, text: str, module_name: str = "Unknown") -> str:
        """Process a text-only prompt using legacy Ollama backend"""
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": text}]
        }]
        
        # Log the prompt
        prompt_preview = text[:2000] + "..." if len(text) > 2000 else text
        logger.info(f"[{module_name}] OLLAMA VLM TEXT QUERY:")
        logger.info(f"[{module_name}] PROMPT: {prompt_preview}")
        
        response = self._call_completion(messages)
        result = response.choices[0].message.content
        
        # Log the response
        result_preview = result[:1000] + "..." if len(result) > 1000 else result
        logger.info(f"[{module_name}] RESPONSE: {result_preview}")
        logger.info(f"[{module_name}] ---")
        
        return result

class VertexBackend(VLMBackend):
    """Google Gemini API with Vertex backend"""

    def __init__(self, model_name: str, **kwargs):
        try:
            from google import genai
        except ImportError:
            raise ImportError("Google Generative AI package not found. Install with: pip install google-generativeai")

        self.model_name = model_name

        # Get vertex_id from kwargs, raise error if not provided
        vertex_id = kwargs.get('vertex_id')
        if not vertex_id:
            raise ValueError("vertex_id is required for VertexBackend. Pass it via --vertex-id parameter.")

        # Initialize the model
        self.client = genai.Client(
            vertexai=True,
            project=vertex_id,
            location='us-central1',
        )
        self.genai = genai

        logger.info(f"Vertex backend initialized with model: {model_name}, project: {vertex_id}")
    
    def _prepare_image(self, img: Union[Image.Image, np.ndarray]) -> Image.Image:
        """Prepare image for Gemini API"""
        # Handle both PIL Images and numpy arrays
        if hasattr(img, 'convert'):  # It's a PIL Image
            return img
        elif hasattr(img, 'shape'):  # It's a numpy array
            return Image.fromarray(img)
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
    
    @retry_with_exponential_backoff
    def _call_generate_content(self, content_parts):
        """Calls the generate_content method with exponential backoff."""
        response = self.client.models.generate_content(
            model='gemini-2.5-flash',
            contents=content_parts
        )
        return response
    
    def get_query(self, img: Union[Image.Image, np.ndarray], text: str, module_name: str = "Unknown") -> str:
        """Process an image and text prompt using Gemini API"""
        try:
            start_time = time.time()
            image = self._prepare_image(img)
            
            # Prepare content for Gemini
            content_parts = [text, image]
            
            # Log the prompt
            prompt_preview = text[:2000] + "..." if len(text) > 2000 else text
            logger.info(f"[{module_name}] GEMINI VLM IMAGE QUERY:")
            logger.info(f"[{module_name}] PROMPT: {prompt_preview}")
            
            # Generate response
            response = self._call_generate_content(content_parts)
            
            # Check for safety filter or content policy issues
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason == 12:
                    logger.warning(f"[{module_name}] Gemini safety filter triggered (finish_reason=12). Trying text-only fallback.")
                    # Fallback to text-only query
                    return self.get_text_query(text, module_name)
            
            result = response.text
            # Log the interaction
            duration = time.time() - start_time
            log_llm_interaction(
                interaction_type=f"local_{module_name}",
                prompt=text,
                response=result,
                duration=duration,
                metadata={"model": self.model_name, "backend": "local", "has_image": True},
                model_info={"model": self.model_name, "backend": "local"}
            )
            
            # Log the response
            result_preview = result[:1000] + "..." if len(result) > 1000 else result
            logger.info(f"[{module_name}] RESPONSE: {result_preview}")
            logger.info(f"[{module_name}] ---")
            print(f'RESPONSE: {result}')
            
            return result
            
        except Exception as e:
            print(f"Error in Gemini image query: {e}")
            logger.error(f"Error in Gemini image query: {e}")
            # Try text-only fallback for any Gemini error
            try:
                logger.info(f"[{module_name}] Attempting text-only fallback due to error: {e}")
                return self.get_text_query(text, module_name)
            except Exception as fallback_error:
                logger.error(f"[{module_name}] Text-only fallback also failed: {fallback_error}")
                raise e
    
    def get_text_query(self, text: str, module_name: str = "Unknown") -> str:
        """Process a text-only prompt using Gemini API"""
        try:
            start_time = time.time()
            # Log the prompt
            prompt_preview = text[:2000] + "..." if len(text) > 2000 else text
            logger.info(f"[{module_name}] GEMINI VLM TEXT QUERY:")
            logger.info(f"[{module_name}] PROMPT: {prompt_preview}")
            
            # Generate response
            response = self._call_generate_content([text])
            
            # Check for safety filter or content policy issues
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason == 12:
                    logger.warning(f"[{module_name}] Gemini safety filter triggered (finish_reason=12). Returning default response.")
                    return "I cannot analyze this content due to safety restrictions. I'll proceed with a basic action: press 'A' to continue."
            
            result = response.text
            
            # Log the interaction
            duration = time.time() - start_time
            log_llm_interaction(
                interaction_type=f"local_{module_name}",
                prompt=text,
                response=result,
                duration=duration,
                metadata={"model": self.model_name, "backend": "local", "has_image": False},
                model_info={"model": self.model_name, "backend": "local"}
            )
            
            # Log the response
            result_preview = result[:1000] + "..." if len(result) > 1000 else result
            logger.info(f"[{module_name}] RESPONSE: {result_preview}")
            logger.info(f"[{module_name}] ---")
            
            return result
            
        except Exception as e:
            print(f"Error in Gemini text query: {e}")
            logger.error(f"Error in Gemini text query: {e}")
            # Return a safe default response
            logger.warning(f"[{module_name}] Returning default response due to error: {e}")
            return "I encountered an error processing the request. I'll proceed with a basic action: press 'A' to continue."


class GeminiBackend(VLMBackend):
    """Google Gemini API backend"""
    
    def __init__(self, model_name: str, **kwargs):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Google Generative AI package not found. Install with: pip install google-generativeai")
        
        self.model_name = model_name
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError("Error: Gemini API key is missing! Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")
        
        # Configure the API
        genai.configure(api_key=self.api_key)
        
        # Initialize the model
        self.model = genai.GenerativeModel(model_name)
        self.genai = genai
        
        logger.info(f"Gemini backend initialized with model: {model_name}")
    
    def _prepare_image(self, img: Union[Image.Image, np.ndarray]) -> Image.Image:
        """Prepare image for Gemini API"""
        # Handle both PIL Images and numpy arrays
        if hasattr(img, 'convert'):  # It's a PIL Image
            return img
        elif hasattr(img, 'shape'):  # It's a numpy array
            return Image.fromarray(img)
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
    
    @retry_with_exponential_backoff
    def _call_generate_content(self, content_parts):
        """Calls the generate_content method with exponential backoff."""
        response = self.model.generate_content(content_parts)
        response.resolve()
        return response
    
    def get_query(self, img: Union[Image.Image, np.ndarray], text: str, module_name: str = "Unknown") -> str:
        """Process an image and text prompt using Gemini API"""
        start_time = time.time()
        try:
            image = self._prepare_image(img)
            
            # Prepare content for Gemini
            content_parts = [text, image]
            
            # Log the prompt
            prompt_preview = text[:2000] + "..." if len(text) > 2000 else text
            logger.info(f"[{module_name}] GEMINI VLM IMAGE QUERY:")
            logger.info(f"[{module_name}] PROMPT: {prompt_preview}")
            
            # Generate response
            response = self._call_generate_content(content_parts)
            
            # Check for safety filter or content policy issues
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason == 12:
                    logger.warning(f"[{module_name}] Gemini safety filter triggered (finish_reason=12). Trying text-only fallback.")
                    # Fallback to text-only query
                    return self.get_text_query(text, module_name)
            
            result = response.text
            duration = time.time() - start_time
            
            # Extract token usage if available
            token_usage = {}
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                token_usage = {
                    "prompt_tokens": getattr(usage, 'prompt_token_count', 0),
                    "completion_tokens": getattr(usage, 'candidates_token_count', 0),
                    "total_tokens": getattr(usage, 'total_token_count', 0)
                }
            
            # Log the interaction
            log_llm_interaction(
                interaction_type=f"gemini_{module_name}",
                prompt=text,
                response=result,
                duration=duration,
                metadata={"model": self.model_name, "backend": "gemini", "has_image": True, "token_usage": token_usage},
                model_info={"model": self.model_name, "backend": "gemini"}
            )
            
            # Log the response
            result_preview = result[:1000] + "..." if len(result) > 1000 else result
            logger.info(f"[{module_name}] RESPONSE: {result_preview}")
            logger.info(f"[{module_name}] ---")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Gemini image query: {e}")
            # Try text-only fallback for any Gemini error
            try:
                logger.info(f"[{module_name}] Attempting text-only fallback due to error: {e}")
                return self.get_text_query(text, module_name)
            except Exception as fallback_error:
                logger.error(f"[{module_name}] Text-only fallback also failed: {fallback_error}")
                raise e
    
    def get_text_query(self, text: str, module_name: str = "Unknown") -> str:
        """Process a text-only prompt using Gemini API"""
        start_time = time.time()
        try:
            # Log the prompt
            prompt_preview = text[:2000] + "..." if len(text) > 2000 else text
            logger.info(f"[{module_name}] GEMINI VLM TEXT QUERY:")
            logger.info(f"[{module_name}] PROMPT: {prompt_preview}")
            
            # Generate response
            response = self._call_generate_content([text])
            
            # Check for safety filter or content policy issues
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason == 12:
                    logger.warning(f"[{module_name}] Gemini safety filter triggered (finish_reason=12). Returning default response.")
                    return "I cannot analyze this content due to safety restrictions. I'll proceed with a basic action: press 'A' to continue."
            
            result = response.text
            duration = time.time() - start_time
            
            # Extract token usage if available
            token_usage = {}
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                token_usage = {
                    "prompt_tokens": getattr(usage, 'prompt_token_count', 0),
                    "completion_tokens": getattr(usage, 'candidates_token_count', 0),
                    "total_tokens": getattr(usage, 'total_token_count', 0)
                }
            
            # Log the interaction
            log_llm_interaction(
                interaction_type=f"gemini_{module_name}",
                prompt=text,
                response=result,
                duration=duration,
                metadata={"model": self.model_name, "backend": "gemini", "has_image": False, "token_usage": token_usage},
                model_info={"model": self.model_name, "backend": "gemini"}
            )
            
            # Log the response
            result_preview = result[:1000] + "..." if len(result) > 1000 else result
            logger.info(f"[{module_name}] RESPONSE: {result_preview}")
            logger.info(f"[{module_name}] ---")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Gemini text query: {e}")
            # Return a safe default response
            logger.warning(f"[{module_name}] Returning default response due to error: {e}")
            return "I encountered an error processing the request. I'll proceed with a basic action: press 'A' to continue."

class SplitBackend(VLMBackend):
    """Split backend: vision model describes the frame, reasoning model decides actions.

    Uses two separate OpenAI-compatible endpoints:
    - Vision model: accepts images, produces scene descriptions
    - Reasoning model: text-only, produces game actions
    """

    # ── Context-aware vision prompts ──────────────────────────────────

    VISION_PROMPT_TITLE = (
        "Analyze this Pokemon Emerald title/menu screen in detail.\n\n"
        "SCREEN TYPE: Identify exactly what screen this is (title screen, "
        "name input keyboard, gender select, intro cutscene, professor speech, "
        "new game/continue menu, etc.)\n\n"
        "MENU STATE: List ALL visible menu options or buttons. "
        "Indicate which option is currently HIGHLIGHTED or SELECTED by the cursor. "
        "Report the cursor position precisely.\n\n"
        "TEXT INPUT: If this is a name input screen, report:\n"
        "  - The current name entered so far (text in the name field)\n"
        "  - The character grid layout visible\n"
        "  - Which character the cursor is currently highlighting\n"
        "  - The position of special buttons (BACK, OK, END, etc.)\n\n"
        "DIALOGUE: If any dialogue or instruction text is visible, transcribe it EXACTLY "
        "character by character. Include the speaker name if shown.\n\n"
        "VISUAL CONTEXT: Describe any character sprites, professor sprites, Pokemon sprites, "
        "backgrounds, or animations visible."
    )

    VISION_PROMPT_BATTLE = (
        "Analyze this Pokemon Emerald battle screen in detail.\n\n"
        "BATTLE PHASE: Identify the current phase:\n"
        "  - Action selection (FIGHT/BAG/POKEMON/RUN menu visible)\n"
        "  - Move selection (4 moves listed)\n"
        "  - Attack animation playing\n"
        "  - Damage/effect text displaying\n"
        "  - Pokemon fainting\n"
        "  - Experience/level up screen\n"
        "  - Pokemon switch prompt\n"
        "  - Wild pokemon catch sequence\n"
        "  - New move learning prompt\n\n"
        "MENU STATE: If a menu is visible:\n"
        "  - List ALL visible options in order\n"
        "  - Which option has the selection cursor/highlight — be PRECISE\n"
        "  - For move selection: name each move and mark which is highlighted\n\n"
        "HP BARS: Describe the visual HP bar states:\n"
        "  - Player's Pokemon: approximate fill (full/high/half/low/critical/empty) and bar color (green/yellow/red)\n"
        "  - Opponent's Pokemon: approximate fill and bar color\n\n"
        "DIALOGUE: If any text box is showing, transcribe it EXACTLY. "
        "This includes effectiveness messages, damage text, status changes, \"What will X do?\" prompts.\n\n"
        "SPRITES: Note visual details about Pokemon sprites "
        "(status indicators like PSN/BRN/SLP icons, fainted, shiny sparkle)."
    )

    VISION_PROMPT_OVERWORLD = (
        "Analyze this Pokemon Emerald overworld screen in detail.\n\n"
        "ENVIRONMENT: Describe the setting — indoor/outdoor, terrain type "
        "(town, route, cave, building interior, Pokemon Center, Mart, gym, etc.)\n\n"
        "PLAYER: Describe the player character:\n"
        "  - Facing direction (up, down, left, right)\n"
        "  - Any interaction indicators (! or ? bubbles)\n"
        "  - Is the player on special terrain (grass, water, sand, etc.)?\n\n"
        "NPCS & OBJECTS: List ALL visible characters and interactive objects:\n"
        "  - For each NPC: position relative to player "
        "(e.g., '2 tiles north', 'directly east'), appearance, "
        "whether they block a path\n"
        "  - Signs, items on ground, Pokeballs, cut trees, strength boulders\n"
        "  - PC, healing machine, shop counter, etc.\n\n"
        "NAVIGATION: Identify visible pathways, doors, stairs, and exits:\n"
        "  - Which directions have open paths vs walls/obstacles\n"
        "  - Door/building entrances and their positions relative to player\n"
        "  - Ledges and which direction they can be jumped\n"
        "  - Tall grass patches\n"
        "  - Staircase locations\n\n"
        "DIALOGUE: If a dialogue box is visible, transcribe it EXACTLY.\n\n"
        "OVERLAYS: Note any screen overlays — route name banners, "
        "weather effects, map transition text."
    )

    VISION_PROMPT_DIALOGUE = (
        "Analyze this Pokemon Emerald dialogue screen in detail.\n\n"
        "DIALOGUE TEXT: Transcribe the EXACT text in the dialogue box, "
        "character by character. Include line breaks where they appear. "
        "Spell character names exactly as shown.\n\n"
        "SPEAKER: Who is speaking? (NPC name, sign, system message, narrator)\n\n"
        "DIALOGUE TYPE: Identify the type:\n"
        "  - NPC conversation\n"
        "  - Sign/notice text\n"
        "  - System message\n"
        "  - Yes/No choice prompt\n"
        "  - Multiple choice prompt\n"
        "  - Item received notification\n\n"
        "CHOICES: If this is a choice dialogue:\n"
        "  - List ALL choices visible\n"
        "  - Which choice is currently HIGHLIGHTED by the cursor\n\n"
        "CONTINUATION: Is there a flashing arrow/triangle at bottom-right "
        "indicating more text to come?\n\n"
        "BACKGROUND: Briefly describe what is visible behind the dialogue box."
    )

    VISION_PROMPT_MENU = (
        "Analyze this Pokemon Emerald menu screen in detail.\n\n"
        "MENU TYPE: Identify the menu (start menu, bag, party summary, pokedex, "
        "Pokemon summary, shop buy/sell, PC storage, options, save, etc.)\n\n"
        "MENU ITEMS: List ALL visible menu options/items in order.\n"
        "  - Mark which item has the selection cursor/highlight with [>]\n"
        "  - For the bag: list visible items with quantities\n"
        "  - For party: list Pokemon with visible HP bars\n"
        "  - For shops: list items with prices\n\n"
        "SUBMENU: If a submenu or detail panel is open:\n"
        "  - What information is displayed\n"
        "  - Stats, descriptions, or move details shown\n"
        "  - Available sub-actions (USE, GIVE, TOSS, CANCEL, etc.) with cursor position\n\n"
        "TEXT: Transcribe any descriptive text or instructions shown.\n\n"
        "SCROLL: Note any scroll indicators (arrows above/below list, page numbers)."
    )

    # Fallback for unknown context
    VISION_PROMPT_DEFAULT = VISION_PROMPT_OVERWORLD

    VISION_PROMPTS = {
        "title": VISION_PROMPT_TITLE,
        "battle": VISION_PROMPT_BATTLE,
        "overworld": VISION_PROMPT_OVERWORLD,
        "dialogue": VISION_PROMPT_DIALOGUE,
        "menu": VISION_PROMPT_MENU,
    }

    def __init__(self, model_name: str, **kwargs):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package not found. Install with: pip install openai")

        # Reasoning model config (model_name is the primary/reasoning model)
        self.reasoning_model = kwargs.get('reasoning_model') or model_name
        reasoning_url = kwargs.get('reasoning_url', 'https://api.schuyler.ai')
        reasoning_api_key = os.getenv('REASONING_API_KEY') or os.getenv('OPENAI_API_KEY') or 'none'

        # Vision model config
        self.vision_model = kwargs.get('vision_model', 'qwen3-vl-32b-thinking')
        vision_url = kwargs.get('vision_url', 'http://192.168.4.245:30002')
        vision_api_key = os.getenv('VISION_API_KEY') or os.getenv('OPENAI_API_KEY') or 'none'

        # Normalize URLs: ensure they end with /v1
        reasoning_base = self._normalize_url(reasoning_url)
        vision_base = self._normalize_url(vision_url)

        # Create two OpenAI clients
        self.vision_client = OpenAI(api_key=vision_api_key, base_url=vision_base)
        self.reasoning_client = OpenAI(api_key=reasoning_api_key, base_url=reasoning_base)

        # Token limits — must be high enough to cover thinking + content
        self.vision_max_tokens = int(kwargs.get('vision_max_tokens', 16384))
        self.reasoning_max_tokens = int(kwargs.get('reasoning_max_tokens', 32768))

        logger.info(f"SplitBackend initialized:")
        logger.info(f"  Vision: {self.vision_model} @ {vision_url}")
        logger.info(f"  Reasoning: {self.reasoning_model} @ {reasoning_url}")

    @staticmethod
    def _detect_context_from_prompt(text: str) -> str:
        """Extract game context from the text prompt to select appropriate vision prompt."""
        text_lower = text.lower()
        if "context: battle" in text_lower:
            return "battle"
        if "context: title" in text_lower:
            return "title"
        if "context: dialogue" in text_lower:
            return "dialogue"
        if "context: menu" in text_lower:
            return "menu"
        if "=== battle mode ===" in text_lower:
            return "battle"
        if "title_sequence" in text_lower:
            return "title"
        return "overworld"

    @staticmethod
    def _normalize_url(url: str) -> str:
        """Ensure URL ends with /v1 for the OpenAI client."""
        url = url.rstrip('/')
        if url.endswith('/v1'):
            return url
        return f"{url}/v1"

    @staticmethod
    def _extract_content(response) -> str:
        """Extract content from response, handling reasoning_content vs content fields."""
        message = response.choices[0].message
        content = message.content

        # If content is None or empty, check for reasoning_content
        if not content:
            reasoning_content = getattr(message, 'reasoning_content', None)
            if reasoning_content:
                logger.warning("SplitBackend: content was empty, falling back to reasoning_content")
                content = reasoning_content

        if not content:
            content = ""
            logger.error("SplitBackend: both content and reasoning_content were empty")

        return content

    @retry_with_exponential_backoff
    def _call_vision_completion(self, messages):
        """Call vision model with retry."""
        return self.vision_client.chat.completions.create(
            model=self.vision_model,
            messages=messages,
            max_tokens=self.vision_max_tokens
        )

    @retry_with_exponential_backoff
    def _call_reasoning_completion(self, messages):
        """Call reasoning model with retry."""
        return self.reasoning_client.chat.completions.create(
            model=self.reasoning_model,
            messages=messages,
            max_tokens=self.reasoning_max_tokens
        )

    def _call_vision(self, img, module_name: str = "Unknown", context: str = "overworld") -> str:
        """Send image to vision model and get scene description."""
        start_time = time.time()

        # Select context-appropriate vision prompt
        vision_prompt = self.VISION_PROMPTS.get(context, self.VISION_PROMPT_DEFAULT)
        logger.info(f"[{module_name}] Vision context: {context}")

        # Convert image to base64 PNG
        if hasattr(img, 'convert'):
            image = img
        elif hasattr(img, 'shape'):
            image = Image.fromarray(img)
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": vision_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]
        }]

        try:
            response = self._call_vision_completion(messages)
            description = self._extract_content(response)
            duration = time.time() - start_time

            token_usage = {}
            if hasattr(response, 'usage') and response.usage:
                token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }

            log_llm_interaction(
                interaction_type=f"split_vision_{module_name}",
                prompt=vision_prompt,
                response=description,
                duration=duration,
                metadata={"model": self.vision_model, "backend": "split_vision", "has_image": True, "token_usage": token_usage, "context": context},
                model_info={"model": self.vision_model, "backend": "split_vision"}
            )
            logger.info(f"[{module_name}] SPLIT VISION [{context}] ({duration:.2f}s): {description[:200]}")
            return description

        except Exception as e:
            duration = time.time() - start_time
            log_llm_error(
                interaction_type=f"split_vision_{module_name}",
                prompt=vision_prompt,
                error=str(e),
                metadata={"model": self.vision_model, "backend": "split_vision", "duration": duration, "context": context}
            )
            logger.error(f"SplitBackend vision call failed: {e}")
            raise

    def _call_reasoning(self, text: str, module_name: str = "Unknown") -> str:
        """Send text prompt to reasoning model and get response."""
        start_time = time.time()

        messages = [{"role": "user", "content": text}]

        try:
            response = self._call_reasoning_completion(messages)
            result = self._extract_content(response)
            duration = time.time() - start_time

            token_usage = {}
            if hasattr(response, 'usage') and response.usage:
                token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }

            log_llm_interaction(
                interaction_type=f"split_reasoning_{module_name}",
                prompt=text,
                response=result,
                duration=duration,
                metadata={"model": self.reasoning_model, "backend": "split_reasoning", "has_image": False, "token_usage": token_usage},
                model_info={"model": self.reasoning_model, "backend": "split_reasoning"}
            )
            logger.info(f"[{module_name}] SPLIT REASONING ({duration:.2f}s): {result[:200]}")
            return result

        except Exception as e:
            duration = time.time() - start_time
            log_llm_error(
                interaction_type=f"split_reasoning_{module_name}",
                prompt=text,
                error=str(e),
                metadata={"model": self.reasoning_model, "backend": "split_reasoning", "duration": duration}
            )
            logger.error(f"SplitBackend reasoning call failed: {e}")
            raise

    def get_query(self, img: Union[Image.Image, np.ndarray], text: str, module_name: str = "Unknown") -> str:
        """Process image+text by splitting into vision description then reasoning.

        1. Detect game context from the text prompt
        2. Send frame to vision model with context-appropriate prompt -> get scene description
        3. Prepend description to the text prompt
        4. Send augmented text to reasoning model -> get action response
        """
        # Stage 1: Detect context for vision prompt selection
        context = self._detect_context_from_prompt(text)

        # Stage 2: Vision with context-aware prompt
        description = self._call_vision(img, module_name, context=context)

        # Stage 3: Augment prompt with vision description
        augmented_prompt = (
            f"VISUAL FRAME ANALYSIS ({context.upper()} screen):\n"
            f"{description}\n\n"
            f"---\n\n"
            f"{text}"
        )

        # Stage 4: Reasoning (text-only)
        return self._call_reasoning(augmented_prompt, module_name)

    def get_text_query(self, text: str, module_name: str = "Unknown") -> str:
        """Process text-only prompt using reasoning model directly."""
        return self._call_reasoning(text, module_name)


class VLM:
    """Main VLM class that supports multiple backends"""

    BACKENDS = {
        'openai': OpenAIBackend,
        'openrouter': OpenRouterBackend,
        'local': LocalHuggingFaceBackend,
        'gemini': GeminiBackend,
        'ollama': LegacyOllamaBackend,  # Legacy support
        'vertex': VertexBackend,  # Added Vertex backend
        'split': SplitBackend,
    }
    
    def __init__(self, model_name: str, backend: str = 'openai', port: int = 8010, **kwargs):
        """
        Initialize VLM with specified backend
        
        Args:
            model_name: Name of the model to use
            backend: Backend type ('openai', 'openrouter', 'local', 'gemini', 'ollama')
            port: Port for Ollama backend (legacy)
            **kwargs: Additional arguments passed to backend
        """
        self.model_name = model_name
        self.backend_type = backend.lower()
        
        # Auto-detect backend based on model name if not explicitly specified
        if backend == 'auto':
            self.backend_type = self._auto_detect_backend(model_name)
        
        if self.backend_type not in self.BACKENDS:
            raise ValueError(f"Unsupported backend: {self.backend_type}. Available: {list(self.BACKENDS.keys())}")
        
        # Initialize the appropriate backend
        backend_class = self.BACKENDS[self.backend_type]
        
        # Pass port parameter for legacy Ollama backend
        if self.backend_type == 'ollama':
            self.backend = backend_class(model_name, port=port, **kwargs)
        else:
            self.backend = backend_class(model_name, **kwargs)
        
        logger.info(f"VLM initialized with {self.backend_type} backend using model: {model_name}")
    
    def _auto_detect_backend(self, model_name: str) -> str:
        """Auto-detect backend based on model name"""
        model_lower = model_name.lower()
        
        if any(x in model_lower for x in ['gpt', 'o4-mini', 'o3', 'claude']):
            return 'openai'
        elif any(x in model_lower for x in ['gemini', 'palm']):
            return 'gemini'
        elif any(x in model_lower for x in ['llama', 'mistral', 'qwen', 'phi']):
            return 'local'
        else:
            # Default to OpenAI for unknown models
            return 'openai'
    
    def get_query(self, img: Union[Image.Image, np.ndarray], text: str, module_name: str = "Unknown") -> str:
        """Process an image and text prompt"""
        try:
            # Backend handles its own logging, so we don't duplicate it here
            result = self.backend.get_query(img, text, module_name)
            return result
        except Exception as e:
            # Only log errors that aren't already logged by the backend
            duration = 0  # Backend tracks actual duration
            log_llm_error(
                interaction_type=f"{self.backend.__class__.__name__.lower()}_{module_name}",
                prompt=text,
                error=str(e),
                metadata={"model": self.model_name, "backend": self.backend.__class__.__name__, "duration": duration, "has_image": True}
            )
            raise
    
    def get_text_query(self, text: str, module_name: str = "Unknown") -> str:
        """Process a text-only prompt"""
        try:
            # Backend handles its own logging, so we don't duplicate it here
            result = self.backend.get_text_query(text, module_name)
            return result
        except Exception as e:
            # Only log errors that aren't already logged by the backend
            duration = 0  # Backend tracks actual duration
            log_llm_error(
                interaction_type=f"{self.backend.__class__.__name__.lower()}_{module_name}",
                prompt=text,
                error=str(e),
                metadata={"model": self.model_name, "backend": self.backend.__class__.__name__, "duration": duration, "has_image": False}
            )
            raise