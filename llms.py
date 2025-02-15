"""
Hold all LLMs 

Make it easier to expland to

I envision expanding this to many types of model - even GPT model as evaluatiors 
So want to keep it here 

"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_llm_tokenizer(model_name, device): 
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=None, 
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # Add this line to ensure model's config matches tokenizer
    model.config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer

