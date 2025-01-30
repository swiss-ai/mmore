from vllm import LLM, SamplingParams
from langchain_core.prompt_values import StringPromptValue, ChatPromptValue
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
from langchain_core.language_models.chat_models import BaseChatModel
from typing import List, Sequence, Dict, Any
from dataclasses import dataclass
import tiktoken
from typing import Optional
from vllm.distributed.parallel_state import destroy_model_parallel
import torch
import gc

class vLLMWrapper(BaseChatModel):
    """A chat model wrapper for vLLM that implements the LangChain BaseChatModel interface."""
    model_name: str
    gpu_memory_utilization: float = 0.8
    tensor_parallel_size: int = 4
    chat_template: Optional[str] = None
    client: Optional[LLM] = None

    def __init__(
        self,
        model_name: str,
        *,
        gpu_memory_utilization: float = 0.8,
        tensor_parallel_size: int = 4,
        chat_template: Optional[str] = None,
    ):
        """Initialize the vLLM client after dataclass initialization."""
        super().__init__(model_name=model_name, gpu_memory_utilization=gpu_memory_utilization, tensor_parallel_size=tensor_parallel_size, chat_template=chat_template)
        self.client = LLM(
            model=self.model_name,
            gpu_memory_utilization=self.gpu_memory_utilization,
            tensor_parallel_size=self.tensor_parallel_size
        )

    @property
    def _llm_type(self) -> str:
        """Return identifier for the model type."""
        return self.model_name
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model_name,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "tensor_parallel_size": self.tensor_parallel_size
        }
    
    def _format_messages(self, messages: List[BaseMessage]) -> str:
        """Format a list of messages into a prompt string.
        
        Args:
            messages: List of BaseMessage objects to format
            
        Returns:
            Formatted string combining all messages
        """
        if self.chat_template:
            return self.chat_template.format(messages=messages)
        
        formatted_messages = []
        for message in messages:
            if isinstance(message, SystemMessage):
                formatted_messages.append(f"System: {message.content}")
            elif isinstance(message, HumanMessage):
                formatted_messages.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                formatted_messages.append(f"Assistant: {message.content}")
            elif isinstance(message, ChatMessage):
                formatted_messages.append(f"{message.role}: {message.content}")
            else:
                formatted_messages.append(message.content)
        return "\n".join(formatted_messages)

    def generate(
            self, 
            messages: List[List[BaseMessage]], 
            stop: List[str] = None, 
            callbacks = None, 
            *, 
            tags = None, 
            metadata = None, 
            run_name = None, 
            run_id = None, 
            **kwargs) -> LLMResult:
        """Generate a chat response for the given messages."""
        prompts = [self._format_messages(message) for message in messages]

        params = SamplingParams(
            temperature=kwargs.get("temperature", 0.2),
            min_p=kwargs.get("min_p", 0.15),
            top_p=kwargs.get("top_p", 0.9),
            max_tokens=kwargs.get("max_new_tokens", 512),
            stop=stop
        )

        raw_batch = self.client.generate(prompts, params)
        
        results: List[ChatResult] = []
        for response in raw_batch:
            message = AIMessage(content=response.outputs[0].text)
            gen = ChatGeneration(message=message)
            res = ChatResult(generations=[gen])
            results.append(res)

        llm_output = self._combine_llm_outputs([res.llm_output for res in results])
        generations = [res.generations for res in results]
        output = LLMResult(generations=generations, llm_output=llm_output)  # type: ignore[arg-type]
        return output

    


    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat response for the given messages.
        
        Args:
            messages: The list of messages to generate a response for
            stop: Optional list of stop sequences
            run_manager: Optional callback manager
            **kwargs: Additional keyword arguments
            
        Returns:
            ChatResult containing the generated response
        """
        prompt = self._format_messages(messages)
        
        params = SamplingParams(
            temperature=kwargs.get("temperature", 0.2),
            min_p=kwargs.get("min_p", 0.15),
            top_p=kwargs.get("top_p", 0.9),
            max_tokens=kwargs.get("max_new_tokens", 512),
            stop=stop
        )

        # Generate response
        raw_response = self.client.generate(prompt, params)
        
        # Convert to ChatResult format
        generations = []
        for response in raw_response:
            message = AIMessage(content=response.outputs[0].text)
            gen = ChatGeneration(message=message)
            generations.append(gen)
            
        return ChatResult(generations=generations)

    
    
    def __del__(self):
        if (not hasattr(self, 'client')):
            return
        destroy_model_parallel()
        del self.client.llm_engine.model_executor.driver_worker
        del self.client # Isn't necessary for releasing memory, but why not
        gc.collect()
        torch.cuda.empty_cache()