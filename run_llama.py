import transformers
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
MODEL_CACHE_DIR = "model_cache"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

def load_or_create_model():
    model_path = os.path.join(MODEL_CACHE_DIR, "model")
    pipeline_path = os.path.join(MODEL_CACHE_DIR, "pipeline")
    print(f"loading model from cache directory: {MODEL_CACHE_DIR}")
    
    if os.path.exists(model_path) and os.path.exists(pipeline_path):
        # Load cached objects
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_path,
            tokenizer=model_path
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        print("Loaded model from cache")
    else:
        # Create new objects
        model_id = "meta-llama/Llama-3.2-1B"
        pipeline = transformers.pipeline("text-generation", model=model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        
        # Save all components
        pipeline.save_pretrained(pipeline_path)
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        print("Created and cached new model")
    
    return pipeline, tokenizer, model

def generate_examples(pipeline):
    examples = [
        "Once upon a time in a land far away,",
        # "The future of artificial intelligence is",
        # "In a world where technology and nature coexist,",
    ]
    
    for example in examples:
        print(f"Prompt: {example}")
        output = pipeline(example, max_length=50, num_return_sequences=1)
        print(f"Generated: {output[0]['generated_text']}\n")
    
    print("model loaded successfully!")
    print("-" * 50)
