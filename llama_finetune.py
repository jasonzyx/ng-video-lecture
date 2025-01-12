import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from run_llama import load_or_create_model, generate_examples
import json
import os
import urllib
from functools import partial
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from utils import calc_loss_batch, train_model_simple, evaluate_model, generate_and_print_sample, calc_loss_loader
import torch.multiprocessing as mp

# Setup multiprocessing and environment
mp.set_start_method('spawn', force=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set device (CUDA > MPS > CPU)
device = torch.device("cuda" if torch.cuda.is_available() else 
                     "mps" if torch.backends.mps.is_available() else "cpu")

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = item['input_ids'].clone()
        return item

    def __len__(self):
        return len(self.encodings.input_ids)

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text, add_special_tokens=True)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)

def test_model(model, tokenizer, device):
    try:
        prompts = [
            "Write a short poem about artificial intelligence.",
            "Tell me a story about a robot learning to be human."
        ]
        
        print("\nTesting model with example prompts:")
        print("=" * 50 + "\n")
        
        for prompt in prompts:
            print(f"Prompt: {prompt}")
            print("-" * 30)
            try:
                generate_and_print_sample(model, tokenizer, prompt, device)
            except Exception as e:
                print(f"Error generating sample: {str(e)}")
            print()
            
    except Exception as e:
        print(f"Error in test_model: {str(e)}")

def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    with open(file_path, "r") as file:
        data = json.load(file)

    return data

def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    batch_max_length = max(len(item)+1 for item in batch)
    if allowed_max_length:
        batch_max_length = min(batch_max_length, allowed_max_length)

    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        if allowed_max_length:
            new_item = new_item[:allowed_max_length]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    plot_name = "loss-plot.pdf"
    print(f"Plot saved as {plot_name}")
    plt.savefig(plot_name)

def model_train(model, tokenizer, train_loader, val_loader, val_data, max_iter=None):
    print("Initial losses")
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)

    print("   Training loss:", train_loss)
    print("   Validation loss:", val_loss)

    start_time = time.time()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], 
        lr=1e-5, 
        weight_decay=0.1,
        betas=(0.9, 0.95)
    )

    num_epochs = 2
    torch.manual_seed(123)
    
    # Use a simple string prompt for generation samples
    example_prompt = "Write a short story about a robot learning to be human."
    
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=example_prompt,
        tokenizer=tokenizer, 
        max_iter=max_iter
    )

    execution_time_minutes = (time.time() - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    print(50*"-")

    model_name = "finetuned_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, model_name)
    print(f"Model saved as {model_name}")

    return model, tokenizer

def set_model_params_for_tuning(model, num_unfrozen_layers):
    # Freeze all parameters initially
    for param in model.parameters():
        param.requires_grad = False
    
    total_layers = len(model.model.layers)
    
    # Unfreeze last few layers
    for i in range(total_layers - num_unfrozen_layers, total_layers):
        layer = model.model.layers[i]
        for param in layer.post_attention_layernorm.parameters():
            param.requires_grad = True
    
    # Print model statistics
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel preparation:")
    print(f"Total layers: {total_layers}")
    print(f"Unfrozen layers: {num_unfrozen_layers}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")
    
    return model

def finetune_model(model, tokenizer, device, max_iter=None, num_unfrozen_layers=1):

    def setup_data():
        # Load and prepare the dataset
        file_path = "instruction-data.json"
        url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
        data = download_and_load_file(file_path, url)

        # Split data into train/test/val sets
        torch.manual_seed(42)
        data = list(data)
        torch.randperm(len(data))

        train_portion = int(len(data) * 0.85)
        test_portion = int(len(data) * 0.1)

        train_data = data[:train_portion]
        test_data = data[train_portion:train_portion + test_portion]
        val_data = data[train_portion + test_portion:]

        print("Dataset sizes:")
        print(f"Training: {len(train_data)}")
        print(f"Validation: {len(val_data)}")
        print(f"Test: {len(test_data)}")
        print(50*"-")
        
        return train_data, test_data, val_data

    def create_data_loaders(train_data, val_data):
        # Setup data loaders with custom collation
        customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=1024)
        loader_config = {
            'batch_size': 8,
            'num_workers': 0,
            'pin_memory': True,
            'collate_fn': customized_collate_fn
        }
        
        torch.manual_seed(123)
        
        train_dataset = InstructionDataset(train_data, tokenizer)
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            drop_last=True,
            **loader_config
        )

        val_dataset = InstructionDataset(val_data, tokenizer)
        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            drop_last=False,
            **loader_config
        )
        
        return train_loader, val_loader

    # Prepare model and move to device
    model = set_model_params_for_tuning(model, num_unfrozen_layers)
    model = model.to(device)
    
    # Setup and execute training pipeline
    train_data, test_data, val_data = setup_data()
    train_loader, val_loader = create_data_loaders(train_data, val_data)

    print("\nStarting fine-tuning process...")
    model, tokenizer = model_train(
        model, 
        tokenizer, 
        train_loader, 
        val_loader, 
        val_data,
        max_iter=max_iter
    )
    print("\nFine-tuning completed!")
    
    return model, tokenizer

def main(is_fine_tune=False, max_iter=10):
    # Load the pretrained model
    pipeline, tokenizer, model = load_or_create_model()

    # Configure tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # Test base model
    print("\nTesting base model capabilities:")
    generate_examples(pipeline)
    if is_fine_tune:    # Fine-tune if requested
        model, tokenizer = finetune_model(
            model, tokenizer, device, 
            max_iter=max_iter, 
            num_unfrozen_layers=1
        )

    # Test final model
    test_model(model, tokenizer, device)

if __name__ == "__main__":
    # Example usage:
    # python llama_finetune.py --fine-tune --max-iter 200
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fine-tune', action='store_true', help='Enable model fine-tuning')
    parser.add_argument('--max-iter', type=int, default=100, help='Maximum number of training iterations')
    args = parser.parse_args()
    main(is_fine_tune=True, max_iter=10)
