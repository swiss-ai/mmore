from vllm import LLM, SamplingParams
from langchain_core.prompt_values import StringPromptValue, ChatPromptValue
from typing import List, Sequence, Dict, Any
from dataclasses import dataclass
import tiktoken
from vllm.distributed.parallel_state import destroy_model_parallel
import torch
import gc

class vLLMWrapper():
    def __init__(self, model: str, gpu_memory_utilization: float = .8, tensor_parallel_size: int = 4):
        self.client = LLM(model=model, gpu_memory_utilization=gpu_memory_utilization, tensor_parallel_size=tensor_parallel_size)


    def __call__(self, prompts: str | Sequence[str] | StringPromptValue | Sequence[StringPromptValue], modalities: List[Dict[str, Any]] = None,  max_tokens: int = 512, stop: str = "<|COMPLETE|>") -> List[str]:
        # We ignore modalities for a VLLM model
        params = SamplingParams(temperature=0.2, min_p=0.15, top_p=0.9, max_tokens=max_tokens, stop=stop)

        if (isinstance(prompts, StringPromptValue)):
            prompts = prompts.text

        chat_responses = self.client.generate(prompts, params)
        
        return [chat_response.outputs[0].text for chat_response in chat_responses]
    
    def __del__(self):
        if (not hasattr(self, 'client')):
            return
        destroy_model_parallel()
        del self.client.llm_engine.model_executor.driver_worker
        del self.client # Isn't necessary for releasing memory, but why not
        gc.collect()
        torch.cuda.empty_cache()