"""Simple LLM implementation for websearch."""
from dataclasses import dataclass

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from langchain_core.messages import BaseMessage

@dataclass
class LLMConfig:
    """Simple LLM configuration."""
    llm_name: str
    max_new_tokens: int = 2000
    temperature: float = 0.2

class LLM:
    """Simple LLM wrapper for websearch."""
    def __init__(self, model, tokenizer, config: LLMConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    @classmethod
    def from_config(cls, config: LLMConfig):
        """Create LLM from configuration."""
        tokenizer = AutoTokenizer.from_pretrained(config.llm_name)
        model = AutoModelForCausalLM.from_pretrained(
            config.llm_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        return cls(model, tokenizer, config)

    def invoke(self, messages: list[BaseMessage]) -> BaseMessage:
        """Generate response for the given messages."""
        # Convert messages to prompt
        prompt = ""
        for msg in messages:
            if msg.type == "system":
                prompt += f"System: {msg.content}\n"
            elif msg.type == "human":
                prompt += f"Human: {msg.content}\n"
            elif msg.type == "assistant":
                prompt += f"Assistant: {msg.content}\n"
        prompt += "Assistant: "

        # Generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        return BaseMessage(content=response, type="assistant") 