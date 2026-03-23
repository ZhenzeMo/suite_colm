"""
LLM Client Utilities
Universal interface for calling different LLM models with function calling support.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class BaseLLMClient:
    """Base interface for LLM clients"""
    
    def call(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None,
             tool_choice: Optional[Any] = None, max_tokens: Optional[int] = None,
             **kwargs) -> Dict[str, Any]:
        """Make LLM API call - to be implemented by subclasses"""
        raise NotImplementedError
    
    def call_with_function(self, messages: List[Dict[str, str]], 
                          function_name: str, function_schema: Dict,
                          **kwargs) -> Dict[str, Any]:
        """Simplified function calling - to be implemented by subclasses"""
        raise NotImplementedError


class LLMClient(BaseLLMClient):
    """Universal LLM client supporting multiple models and function calling"""
    
    # Model configurations
    MODEL_CONFIGS = {
        'qwen': {
            'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
            'api_key_env': 'QWEN_API_KEY',
            'tool_choice_format': 'object',  # Qwen: supports object format in compatible mode
        },
        'deepseek': {
            'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
            'api_key_env': 'QWEN_API_KEY',
            'tool_choice_format': 'object',  # DeepSeek-V3/V3.2: same as Qwen (DeepSeek-R1 has separate client)
        },
        'llama': {
            'base_url': 'https://api.novita.ai/openai',
            'api_key_env': 'LLAMA_API_KEY',
            'tool_choice_format': 'required',  # Novita: use "required"
        },
        'gpt': {
            'base_url': 'https://api.openai.com/v1',
            'api_key_env': 'OPENAI_API_KEY',
            'tool_choice_format': 'object',  # OpenAI: use object format
        },
    }
    
    def __init__(self, model_name: str, temperature: float = 0.1, seed: int = 42, 
                 debug: bool = False, max_retries: int = 10, timeout: float = 120.0):
        """
        Initialize LLM client
        
        Args:
            model_name: Model identifier
                - Qwen: 'qwen-flash', 'qwen-plus', 'qwen-max' (via aliyun)
                - DeepSeek: 'deepseek-r1', 'deepseek-v3' (via aliyun, same as Qwen)
                - Llama: 'meta-llama/llama-3.3-70b-instruct' (via novita)
                - GPT: 'gpt-4o', 'gpt-4-turbo', etc. (via openai)
            temperature: Sampling temperature
            seed: Random seed for reproducibility
            debug: Enable debug output
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
        """
        self.model_name = model_name
        self.temperature = temperature
        self.seed = seed
        self.debug = debug
        
        # Detect model type and get configuration
        self.model_type = self._detect_model_type()
        config = self.MODEL_CONFIGS[self.model_type]
        
        # Get API key
        api_key = os.getenv(config['api_key_env'])
        if not api_key:
            raise ValueError(f"API key not found: {config['api_key_env']}")
        
        # Initialize OpenAI client
        self.client = OpenAI(
            base_url=config['base_url'],
            api_key=api_key,
            max_retries=max_retries,
            timeout=timeout
        )
        
        self.tool_choice_format = config['tool_choice_format']
        
        if debug:
            masked_key = f"{api_key[:8]}...{api_key[-8:]}" if len(api_key) > 16 else f"{api_key[:4]}...{api_key[-4:]}"
            logger.info(f"LLMClient initialized: {model_name} ({self.model_type})")
            logger.info(f"API Key ({config['api_key_env']}): {masked_key}")
    
    def _detect_model_type(self) -> str:
        """Detect model type from model name"""
        model_lower = self.model_name.lower()
        
        if 'llama' in model_lower:
            return 'llama'
        elif 'deepseek' in model_lower:
            return 'deepseek'
        elif model_lower.startswith('gpt') or model_lower.startswith('o1'):
            return 'gpt'
        elif 'qwen' in model_lower:
            return 'qwen'
        else:
            # Default to qwen for aliyun models
            return 'qwen'
    
    def call(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None,
             tool_choice: Optional[Any] = None, max_tokens: Optional[int] = None,
             **kwargs) -> Dict[str, Any]:
        """
        Make LLM API call
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of function calling tools
            tool_choice: Optional tool choice specification
            max_tokens: Optional max tokens limit
            **kwargs: Additional API parameters
        
        Returns:
            Dict with 'content' (text response) or 'tool_calls' (function calls)
        """
        # Build API parameters
        api_params = {
            'model': self.model_name,
            'messages': messages,
            'temperature': self.temperature,
            'seed': self.seed,
            **kwargs
        }
        
        # Add max_tokens if specified
        if max_tokens:
            api_params['max_tokens'] = max_tokens
        
        # Add tools and tool_choice if provided
        if tools:
            api_params['tools'] = tools
            if tool_choice:
                # Use explicit tool_choice if provided
                api_params['tool_choice'] = tool_choice
            elif self.tool_choice_format == 'object':
                # OpenAI/Qwen: use object format to force function calling
                func_name = tools[0]['function']['name']
                api_params['tool_choice'] = {
                    'type': 'function',
                    'function': {'name': func_name}
                }
            elif self.tool_choice_format == 'required':
                # Novita/Llama: use "required" string
                api_params['tool_choice'] = 'required'
            elif self.tool_choice_format == 'auto':
                # Fallback: use "auto" (model decides)
                api_params['tool_choice'] = 'auto'
        
        if self.debug:
            logger.info(f"LLM Call: {self.model_name}")
            logger.info(f"Tool choice format: {self.tool_choice_format}")
            logger.info(f"Messages: {json.dumps(messages, ensure_ascii=False)[:500]}...")
            if tools:
                logger.info(f"Tools: {[t['function']['name'] for t in tools]}")
                if 'tool_choice' in api_params:
                    logger.info(f"Tool choice: {api_params['tool_choice']}")
        
        # Make API call
        response = self.client.chat.completions.create(**api_params)
        
        # Parse response
        message = response.choices[0].message
        
        result = {}
        if hasattr(message, 'tool_calls') and message.tool_calls:
            # Function calling response
            tool_call = message.tool_calls[0]
            result['tool_calls'] = [{
                'function_name': tool_call.function.name,
                'arguments': json.loads(tool_call.function.arguments)
            }]
            if self.debug:
                logger.info(f"Function call: {tool_call.function.name}")
                logger.info(f"Arguments: {result['tool_calls'][0]['arguments']}")
        else:
            # Text response
            result['content'] = message.content if message.content else ""
            if self.debug:
                logger.info(f"Response: {result['content'][:200]}...")
        
        return result
    
    def call_with_function(self, messages: List[Dict[str, str]], 
                          function_name: str, function_schema: Dict,
                          **kwargs) -> Dict[str, Any]:
        """
        Simplified function calling - returns parsed function arguments directly
        
        Args:
            messages: List of message dicts
            function_name: Name of the function
            function_schema: Function parameters schema
            **kwargs: Additional API parameters
        
        Returns:
            Dict with parsed function arguments, or empty dict on error
        """
        tools = [{
            'type': 'function',
            'function': {
                'name': function_name,
                'description': function_schema.get('description', ''),
                'parameters': function_schema.get('parameters', {})
            }
        }]
        
        try:
            result = self.call(messages, tools=tools, **kwargs)
            
            if 'tool_calls' in result:
                return result['tool_calls'][0]['arguments']
            else:
                # Fallback: Extract answer from text (for models that don't force function calling)
                text_content = result.get('content', '')
                logger.warning(f"No tool calls in response, attempting to extract from text: {text_content[:100]}")
                
                # Get expected answer format from schema
                params = function_schema.get('parameters', {})
                properties = params.get('properties', {})
                answer_prop = properties.get('answer', {})
                
                if answer_prop.get('type') == 'string':
                    # Extract letter options (A, B, C, D)
                    import re
                    enum_values = answer_prop.get('enum', [])
                    if enum_values:
                        # Look for valid options in text
                        pattern = r'\b(' + '|'.join(enum_values) + r')\b'
                        matches = re.findall(pattern, text_content, re.IGNORECASE)
                        if matches:
                            extracted = matches[-1].upper()
                            logger.info(f"Extracted answer from text: {extracted}")
                            return {'answer': extracted}
                
                logger.error(f"Could not extract answer from text response")
                return {}
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse function arguments: {e}")
            return {}
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return {}


class DeepSeekR1Client(BaseLLMClient):
    """DeepSeek-R1 client with reasoning capabilities"""
    
    def __init__(self, model_name: str = "deepseek-r1", temperature: float = 0.1, 
                 seed: int = 42, debug: bool = False, max_retries: int = 10, 
                 timeout: float = 120.0):
        """
        Initialize DeepSeek-R1 client
        
        Args:
            model_name: Model identifier (e.g., 'deepseek-r1')
            temperature: Sampling temperature
            seed: Random seed for reproducibility
            debug: Enable debug output
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
        """
        self.model_name = model_name
        self.temperature = temperature
        self.seed = seed
        self.debug = debug
        
        # Get API key (using QWEN_API_KEY for dashscope)
        api_key = os.getenv('QWEN_API_KEY')
        if not api_key:
            raise ValueError("API key not found: QWEN_API_KEY (required for DeepSeek R1 via dashscope)")
        
        # Initialize OpenAI client with dashscope base URL
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            max_retries=max_retries,
            timeout=timeout
        )
        
        if debug:
            masked_key = f"{api_key[:8]}...{api_key[-8:]}" if len(api_key) > 16 else f"{api_key[:4]}...{api_key[-4:]}"
            logger.info(f"DeepSeekR1Client initialized: {model_name}")
            logger.info(f"API Key (QWEN_API_KEY): {masked_key}")
    
    def call(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None,
             tool_choice: Optional[Any] = None, max_tokens: Optional[int] = None,
             **kwargs) -> Dict[str, Any]:
        """
        Make DeepSeek-R1 API call
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of function calling tools
            tool_choice: Optional tool choice specification (default: 'auto')
            max_tokens: Optional max tokens limit
            **kwargs: Additional API parameters
        
        Returns:
            Dict with 'content' (text response), 'reasoning_content', or 'tool_calls' (function calls)
        """
        # Build API parameters
        api_params = {
            'model': self.model_name,
            'messages': messages,
            'temperature': self.temperature,
            'seed': self.seed,
            'extra_body': {
                'enable_thinking': True
            },
            **kwargs
        }
        
        # Add max_tokens if specified
        if max_tokens:
            api_params['max_tokens'] = max_tokens
        
        # Add tools and tool_choice if provided
        if tools:
            api_params['tools'] = tools
            
            if tool_choice is not None:
                # Use explicit tool_choice if provided
                api_params['tool_choice'] = tool_choice
            else:
                # Default: use 'auto' - let model decide (but prompt will guide it)
                api_params['tool_choice'] = 'auto'
        
        if self.debug:
            logger.info(f"DeepSeek-R1 Call: {self.model_name}")
            logger.info(f"Messages: {json.dumps(messages, ensure_ascii=False)[:500]}...")
            if tools:
                logger.info(f"Tools: {[t['function']['name'] for t in tools]}")
                logger.info(f"Tool choice: {api_params.get('tool_choice', 'not set')}")
        
        # Make API call
        response = self.client.chat.completions.create(**api_params)
        
        # Parse response
        message = response.choices[0].message
        
        result = {}
        
        # Extract reasoning content if available
        reasoning_content = getattr(message, 'reasoning_content', None)
        if reasoning_content:
            result['reasoning_content'] = reasoning_content
            if self.debug:
                logger.info(f"Reasoning: {reasoning_content[:200]}...")
        
        # Check for tool calls
        if hasattr(message, 'tool_calls') and message.tool_calls:
            # Function calling response
            tool_call = message.tool_calls[0]
            result['tool_calls'] = [{
                'function_name': tool_call.function.name,
                'arguments': json.loads(tool_call.function.arguments)
            }]
            if self.debug:
                logger.info(f"Function call: {tool_call.function.name}")
                logger.info(f"Arguments: {result['tool_calls'][0]['arguments']}")
        else:
            # Text response
            result['content'] = message.content if message.content else ""
            if self.debug:
                logger.info(f"Response: {result['content'][:200]}...")
        
        return result
    
    def call_with_function(self, messages: List[Dict[str, str]], 
                          function_name: str, function_schema: Dict,
                          **kwargs) -> Dict[str, Any]:
        """
        Simplified function calling - returns parsed function arguments directly
        
        Args:
            messages: List of message dicts
            function_name: Name of the function
            function_schema: Function parameters schema
            **kwargs: Additional API parameters
        
        Returns:
            Dict with parsed function arguments, or empty dict on error
        """
        tools = [{
            'type': 'function',
            'function': {
                'name': function_name,
                'description': function_schema.get('description', ''),
                'parameters': function_schema.get('parameters', {})
            }
        }]
        
        # For DeepSeek R1: Add prompt instruction to force function calling
        # Inject system message to ensure function is called
        enhanced_messages = []
        system_found = False
        
        for msg in messages:
            if msg['role'] == 'system':
                # Enhance existing system message
                enhanced_messages.append({
                    'role': 'system',
                    'content': f"{msg['content']}\n\nIMPORTANT: You MUST call the tool '{function_name}' and ONLY provide the tool call. Do not output normal text."
                })
                system_found = True
            else:
                enhanced_messages.append(msg)
        
        # If no system message found, add one at the beginning
        if not system_found:
            enhanced_messages = [
                {
                    'role': 'system',
                    'content': f"You MUST call the tool '{function_name}' and ONLY provide the tool call. Do not output normal text."
                }
            ] + messages
        
        try:
            result = self.call(enhanced_messages, tools=tools, **kwargs)
            
            if 'tool_calls' in result:
                return result['tool_calls'][0]['arguments']
            else:
                # Fallback: Extract answer from text (if model chose to respond with text instead)
                text_content = result.get('content', '')
                logger.warning(f"No tool calls in response, attempting to extract from text: {text_content[:100]}")
                
                # Get expected answer format from schema
                params = function_schema.get('parameters', {})
                properties = params.get('properties', {})
                answer_prop = properties.get('answer', {})
                
                if answer_prop.get('type') == 'string':
                    # Extract letter options (A, B, C, D)
                    import re
                    enum_values = answer_prop.get('enum', [])
                    if enum_values:
                        # Look for valid options in text
                        pattern = r'\b(' + '|'.join(enum_values) + r')\b'
                        matches = re.findall(pattern, text_content, re.IGNORECASE)
                        if matches:
                            extracted = matches[-1].upper()
                            logger.info(f"Extracted answer from text: {extracted}")
                            return {'answer': extracted}
                
                logger.error(f"Could not extract answer from text response")
                return {}
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse function arguments: {e}")
            return {}
        except Exception as e:
            logger.error(f"DeepSeek-R1 call failed: {e}")
            return {}


class GPT5Client(BaseLLMClient):
    """GPT-5 series client with reasoning capabilities (supports gpt-5, gpt-5.1, gpt-5.2, etc.)"""
    
    def __init__(self, model_name: str = "gpt-5", debug: bool = False, 
                 max_retries: int = 10, timeout: float = 120.0):
        """
        Initialize GPT-5 client
        
        Args:
            model_name: Model identifier (e.g., 'gpt-5', 'gpt-5.1', 'gpt-5.2', 'gpt-5-nano')
            debug: Enable debug output
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
        """
        self.model_name = model_name
        self.debug = debug
        
        # Get API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("API key not found: OPENAI_API_KEY")
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=api_key,
            max_retries=max_retries,
            timeout=timeout
        )
        
        if debug:
            masked_key = f"{api_key[:8]}...{api_key[-8:]}" if len(api_key) > 16 else f"{api_key[:4]}...{api_key[-4:]}"
            logger.info(f"GPT5Client initialized: {model_name}")
            logger.info(f"API Key (OPENAI_API_KEY): {masked_key}")
    
    def _convert_messages_to_gpt5_format(self, messages: List[Dict[str, str]]) -> List[Dict]:
        """Convert standard messages to GPT-5 input format"""
        gpt5_messages = []
        for msg in messages:
            role = msg['role']
            # Map roles: system/user/assistant -> developer/user/assistant
            if role == 'system':
                role = 'developer'
            gpt5_messages.append({
                'role': role,
                'content': msg['content']
            })
        return gpt5_messages
    
    def _convert_tools_to_gpt5_format(self, tools: List[Dict]) -> List[Dict]:
        """Convert standard tools to GPT-5 function format"""
        gpt5_tools = []
        for tool in tools:
            if tool.get('type') == 'function':
                func = tool['function']
                params = func.get('parameters', {})
                
                # Ensure additionalProperties is set
                if params and 'additionalProperties' not in params:
                    params['additionalProperties'] = False
                
                # GPT-5 uses standard function format
                gpt5_tools.append({
                    'type': 'function',
                    'name': func['name'],
                    'description': func.get('description', ''),
                    'parameters': params,
                    'strict': True
                })
        return gpt5_tools
    
    def call(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None,
             tool_choice: Optional[Any] = None, max_tokens: Optional[int] = None,
             **kwargs) -> Dict[str, Any]:
        """
        Make GPT-5 API call
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of function calling tools
            tool_choice: Optional tool choice specification
            max_tokens: Optional max tokens limit
            **kwargs: Additional API parameters
        
        Returns:
            Dict with 'content' (text response) or 'tool_calls' (function calls)
        """
        # Convert messages to GPT-5 format
        gpt5_input = self._convert_messages_to_gpt5_format(messages)
        
        # Build API parameters
        api_params = {
            'model': self.model_name,
            'input': gpt5_input,
            'text': {'format': {'type': 'text'}},
            'reasoning': {'effort': 'high'}
        }
        
        # Add tools and tool_choice if provided
        if tools:
            api_params['tools'] = self._convert_tools_to_gpt5_format(tools)
            
            # Set tool_choice to force function calling (GPT-5.1/5.2 style)
            if len(tools) == 1:
                tool_name = tools[0]['function']['name']
                api_params['tool_choice'] = {
                    'type': 'allowed_tools',
                    'mode': 'required',
                    'tools': [{'type': 'function', 'name': tool_name}]
                }
        
        if self.debug:
            logger.info(f"GPT-5 Call: {self.model_name}")
            logger.info(f"Messages: {json.dumps(messages, ensure_ascii=False)[:500]}...")
            if tools:
                logger.info(f"Tools: {[t['function']['name'] for t in tools]}")
                logger.info(f"Tool choice: {api_params.get('tool_choice', 'not set')}")
        
        # Make API call
        response = self.client.responses.create(**api_params)
        
        # Parse response
        result = {}
        text_content = []
        
        if self.debug:
            logger.info(f"GPT-5 response.output length: {len(response.output)}")
        
        # Parse output items - look for function calls and content
        for item in response.output:
            # Skip reasoning items
            if hasattr(item, 'type') and 'reasoning' in str(item.type).lower():
                continue
            
            # Check for function calls (GPT-5 function type, not custom)
            if hasattr(item, 'type') and item.type == 'function_call':
                # Parse function arguments
                arguments_str = item.arguments if hasattr(item, 'arguments') else '{}'
                try:
                    arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
                except json.JSONDecodeError:
                    arguments = {'answer': arguments_str.strip()}
                
                result['tool_calls'] = [{
                    'function_name': item.name,
                    'arguments': arguments
                }]
                
                if self.debug:
                    logger.info(f"Function call: {item.name}")
                    logger.info(f"Arguments: {arguments}")
                return result  # Return immediately when function call found
            
            # Extract text content
            if hasattr(item, 'content'):
                content = item.content
                if isinstance(content, list):
                    for part in content:
                        if hasattr(part, 'text'):
                            text_content.append(part.text)
                elif isinstance(content, str):
                    text_content.append(content)
        
        # Return text content
        result['content'] = ''.join(text_content)
        if self.debug:
            logger.info(f"Response (text): {result.get('content', '')[:200]}...")
        
        return result
    
    def call_with_function(self, messages: List[Dict[str, str]], 
                          function_name: str, function_schema: Dict,
                          **kwargs) -> Dict[str, Any]:
        """
        Simplified function calling - returns parsed function arguments directly
        
        Args:
            messages: List of message dicts
            function_name: Name of the function
            function_schema: Function parameters schema
            **kwargs: Additional API parameters
        
        Returns:
            Dict with parsed function arguments, or empty dict on error
        """
        tools = [{
            'type': 'function',
            'function': {
                'name': function_name,
                'description': function_schema.get('description', ''),
                'parameters': function_schema.get('parameters', {})
            }
        }]
        
        try:
            result = self.call(messages, tools=tools, **kwargs)
            
            if 'tool_calls' in result:
                return result['tool_calls'][0]['arguments']
            else:
                logger.warning(f"No tool calls in response, got text: {result.get('content', '')[:100]}")
                return {}
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse function arguments: {e}")
            return {}
        except Exception as e:
            logger.error(f"GPT-5 call failed: {e}")
            return {}


def create_client(model_name: str, **kwargs) -> BaseLLMClient:
    """
    Convenience function to create LLM client
    
    Supported models:
    - Qwen: qwen-flash, qwen-plus, qwen-max (aliyun)
    - DeepSeek-R1: deepseek-r1 (aliyun, with thinking support)
    - DeepSeek: deepseek-v3 (aliyun, same as Qwen)
    - Llama: meta-llama/llama-3.3-70b-instruct (novita)
    - GPT-4: gpt-4o, gpt-4-turbo, etc. (openai)
    - GPT-5: gpt-5, gpt-5.1, gpt-5.2, gpt-5-nano (openai, with reasoning effort=high)
    """
    model_lower = model_name.lower()
    
    # Detect GPT-5 models (gpt-5, gpt-5.1, gpt-5.2, gpt-5-nano, etc.)
    if 'gpt-5' in model_lower:
        # Remove temperature and seed if provided (GPT-5 doesn't use them)
        kwargs.pop('temperature', None)
        kwargs.pop('seed', None)
        return GPT5Client(model_name, **kwargs)
    
    # Detect DeepSeek-R1 models
    elif 'deepseek-r1' in model_lower:
        return DeepSeekR1Client(model_name, **kwargs)
    
    # Default: use universal LLM client (handles Qwen, DeepSeek-V3, Llama, GPT-4)
    else:
        return LLMClient(model_name, **kwargs)

