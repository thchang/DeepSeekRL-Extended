"""
Implementation of GRPO, DeepSeek style training without external libraries 
"""
import argparse
import os
import json
import torch
from datasets import load_dataset, Dataset
from tqdm import tqdm
from typing import Any 
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import defaultdict

import llms
import rldatasets
import evaluator
import utils


def eval_on_test_set(model, tokenizer, test_loader, device, args, round_num): 
    print("Running evaluation on test set...")
    
    # Track metrics across all test examples
    total_scores = {
        'total_score': 0.0,
        'correctness': 0.0,
        'int_format': 0.0, 
        'strict_format': 0.0,
        'soft_format': 0.0,
        'xml_count': 0.0
    }
    num_examples = 0
    num_correct = 0

    # Create log file for this evaluation round
    log_file = os.path.join(args.output_dir, f'generation_log_{round_num}.txt')
    test_loader.reset()
    with open(log_file, 'w') as f:
        # Run through test set
        for question, answer in tqdm(test_loader, desc="Evaluating on test set"):
            # Generate model response
            full_prompt = test_loader.pre_prompt + question
            inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
            outputs = model.generate(**inputs, max_length=args.max_completion_length)
            generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_answer = generated_answer[len(test_loader.pre_prompt + question):]

            # Evaluate the response
            score, metrics = evaluator.evaluate_answer(question, generated_answer, answer)
            
            # Track correctness
            if metrics['correctness'] == 2.0:  # Full correctness score
                num_correct += 1
                
            # Accumulate metrics
            total_scores['total_score'] += score
            for k, v in metrics.items():
                total_scores[k] += v
            num_examples += 1

            # Log this example
            f.write("\n==================================================\n")
            f.write(f"Q# {num_examples}\n")
            f.write(f"Question: {question}\n")
            f.write(f"Response: {generated_answer}\n")
            f.write(f"Ground Truth: {answer}\n")
            f.write("Metrics:\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value}\n")
            f.write(f"Total Score: {score}\n")

    # Calculate averages
    avg_scores = {k: v/num_examples for k,v in total_scores.items()}
    accuracy = num_correct / num_examples * 100

    # Save metrics
    metrics_path = os.path.join(args.output_dir, f'eval_metrics_{round_num}.json')
    with open(metrics_path, 'w') as f:
        json.dump({**avg_scores, 'accuracy': accuracy}, f, indent=4)

    if args.verbose:
        print("\nEvaluation Results:")
        print("-" * 20)
        print(f"Accuracy: {accuracy:.2f}%")
        for metric, value in avg_scores.items():
            print(f"{metric:15s}: {value:.4f}")
        print("-" * 20)

    return avg_scores, accuracy

import torch
import torch.nn.functional as F
import argparse
from transformers import PreTrainedModel, PreTrainedTokenizerBase

def grpo_loss(
        model: PreTrainedModel,
        base_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        question: str,
        answer: str,
        device: str,
        args: argparse.Namespace
) -> tuple[torch.Tensor, dict, float]:
    """
    Compute GRPO loss between the current model and base model.

    Args:
        model: The fine-tuned model.
        base_model: The reference (baseline) model.
        tokenizer: Tokenizer used for encoding.
        question: The input prompt.
        answer: The ground truth answer.
        device: Device to run the model on.
        args: Training arguments.

    Returns:
        total_loss: Computed GRPO loss.
        total_metrics: Evaluation metrics.
        accuracy: Percentage of correct responses.
    """

    # Track evaluation metrics
    total_metrics = {
        'total_score': 0.0,
        'correctness': 0.0,
        'int_format': 0.0,
        'strict_format': 0.0,
        'soft_format': 0.0,
        'xml_count': 0.0
    }
    num_correct = 0

    ##### **1. Generate Responses from Model and Base Model**
    with torch.inference_mode():
        # Prepare prompt
        full_prompt = train_loader.pre_prompt + question

        # Tokenize prompt and send to device
        inputs = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True).to(device)

        # Repeat inputs for multiple answer generations
        inputs = {k: v.repeat(args.num_generations, 1) for k, v in inputs.items()}

        # Generate responses from the model (fine-tuned)
        outputs = model.generate(
            **inputs, max_length=args.max_completion_length, temperature=0.7, top_p=0.9, do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        # Decode responses and remove the original prompt from outputs
        responses = [tokenizer.decode(o, skip_special_tokens=True)[len(full_prompt):] for o in outputs]

    ##### **2. Compute Rewards & Normalize Advantages**
    all_scores = []
    for response in responses:
        total_score, metrics = evaluator.evaluate_answer(question, response, answer)
        all_scores.append(total_score)

        # Accumulate total metrics for logging
        for k, v in metrics.items():
            total_metrics[k] += v
        
        total_metrics["total_score"] += total_score

        # Track correctness
        if metrics['correctness'] == 2.0:
            num_correct += 1

    # Average the accumulated metrics
    for k in total_metrics:
        total_metrics[k] /= len(responses)

    # Convert all scores to tensor and normalize advantage
    all_scores = torch.tensor(all_scores, device=device)
    mean_rewards = all_scores.view(-1, args.num_generations).mean(dim=1, keepdim=True)
    std_rewards = all_scores.view(-1, args.num_generations).std(dim=1, keepdim=True)
    advantage = (all_scores - mean_rewards) / (std_rewards + 1e-4)

    ##### **3. Compute Per-Token Log Probabilities**
    input_ids = tokenizer(responses, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
    outputs = model(input_ids)
    log_probs = F.log_softmax(outputs.logits, dim=-1)

    # Base model is just model 
    with torch.no_grad():
        base_outputs = model(input_ids)
        base_log_probs = F.log_softmax(base_outputs.logits, dim=-1)

    ##### **4. Extract Response-Specific Log Probabilities**
    full_prompt_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(device)
    prompt_length = full_prompt_ids.shape[1]

    # Ensure we extract the response part correctly
    response_ids = input_ids[:, prompt_length:]  # Only the generated response tokens

    # Shift logits to align with the next-token prediction
    response_logits = log_probs[:, prompt_length - 1:-1, :]
    token_log_probs = torch.gather(response_logits, dim=-1, index=response_ids.unsqueeze(-1)).squeeze(-1)

    # Extract old model probabilities
    old_response_logits = base_log_probs[:, prompt_length - 1:-1, :]
    old_token_log_probs = torch.gather(old_response_logits, dim=-1, index=response_ids.unsqueeze(-1)).squeeze(-1)

    ##### **5. Compute KL Divergence Term**
    kl_div = torch.exp(old_token_log_probs - token_log_probs) - (old_token_log_probs - token_log_probs) - 1

    ##### **6. Create Completion Mask (Fix EOS Handling)**
    is_eos = input_ids == tokenizer.eos_token_id
    eos_idx = is_eos.int().argmax(dim=1, keepdim=True)

    # Correctly mask only response tokens, stopping at EOS
    sequence_indices = torch.arange(input_ids.size(1), device=device).expand(input_ids.size(0), -1)
    completion_mask = (sequence_indices >= prompt_length) & (sequence_indices < eos_idx)
    completion_mask = completion_mask.float().to(device)

    # Ensure completion mask matches response_ids shape
    sequence_length = response_ids.shape[1]  # Only count response tokens
    completion_mask = completion_mask[:, prompt_length:prompt_length + sequence_length]  # Fix misalignment

    ##### **7. Compute GRPO Loss**
    beta = 0.04  # KL coefficient
    per_token_logps = token_log_probs - old_token_log_probs
    advantages = advantage.view(-1, 1).expand_as(token_log_probs)

    # Correct log prob calculation
    per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages
    per_token_loss = -(per_token_loss - beta * kl_div)

    # Corrected final loss with masking
    total_loss = per_token_loss.mean()
    ##### **8. Return Loss & Accuracy**
    return total_loss, total_metrics, num_correct / args.num_generations * 100

def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training arguments")
    
    # Training hyperparameters
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.99)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_generations", type=int, default=16)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--max_completion_length", type=int, default=786)
    parser.add_argument("--num_train_iters", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7111994)
    parser.add_argument("--eval_iterations", type=int, default=100)
    parser.add_argument("--dataset_name", type=str, default="gsm8k")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    # Get all args 
    args = parse_args() 

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    # Save args to json file
    args_dict = vars(args)
    args_path = os.path.join(args.output_dir, 'args.json')
    with open(args_path, 'w') as f:
        json.dump(args_dict, f, indent=4)
    running_metrics = defaultdict(list)  # For accumulating metrics between logging steps

    # Create eval logs directory
    eval_log_dir = os.path.join(args.output_dir, 'eval_logs')
    os.makedirs(eval_log_dir, exist_ok=True)

    # Seed everything 
    utils.seed_everything(args.seed)
    torch.set_float32_matmul_precision('high') 

    # Set device and enable bf16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    
    # Setup network 
    model, tokenizer = llms.get_llm_tokenizer(args.model_name, device)
    base_model, _ = llms.get_llm_tokenizer(args.model_name, device)
    # Get data set 
    train_loader, test_loader = rldatasets.get_dataloaders(args.dataset_name)

    # Dictionary to store metrics over time
    metrics_history = {
        'loss': {},
        'train_metrics': {},
        'eval_metrics': {},
        'train_accuracy': {},
        'eval_accuracy': {}
    }

    # Setup optimizer for trainer agent with GRPO config settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
        eps=1e-8
    )

    # Add cosine learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.num_train_iters,
        eta_min=args.learning_rate * 0.1
    )

    # Begin training 
    accumulated_loss = 0
    optimizer.zero_grad()
    for round_num in tqdm(range(args.num_train_iters), desc="Training Progress"):
    
        # Evaluate on test set every so often 
        if False: #round_num % args.eval_iterations == 0:
            eval_metrics, eval_accuracy = eval_on_test_set(model, tokenizer, test_loader, device, args, round_num)
            metrics_history['eval_metrics'][round_num] = eval_metrics
            metrics_history['eval_accuracy'][round_num] = eval_accuracy

        # Get next question
        question, answer = next(train_loader)

        # Get total loss and metrics
        total_loss, train_metrics, train_accuracy = grpo_loss(model, base_model, tokenizer, question, answer, device, args)
        running_metrics['loss'].append(total_loss.item())
        running_metrics['train_accuracy'].append(train_accuracy)
        for k, v in train_metrics.items():
            running_metrics[f'train_{k}'].append(v)
                
        # Gradient accumulation
        total_loss = total_loss / args.gradient_accumulation_steps
        total_loss.backward()
        accumulated_loss += total_loss.item()
        
        # Step optimizer and scheduler
        if (round_num + 1) % args.gradient_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step() 
            optimizer.zero_grad()
            accumulated_loss = 0  # Reset accumulated loss

            # Save the metrics history
            avg_metrics = {k: sum(v)/len(v) for k, v in running_metrics.items()}
            
            # Update metrics history
            metrics_history['loss'][round_num] = avg_metrics['loss']
            metrics_history['train_accuracy'][round_num] = avg_metrics['train_accuracy']
            
            # Initialize train_metrics dict for this round if it doesn't exist
            if round_num not in metrics_history['train_metrics']:
                metrics_history['train_metrics'][round_num] = {}
                
            for k, v in avg_metrics.items():
                if k.startswith('train_'):
                    metrics_history['train_metrics'][round_num][k[6:]] = v  # Remove 'train_' prefix
            # Clear running metrics
            running_metrics.clear()
            with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics_history, f, indent=4)

            if args.verbose:
                print(f"\nTraining Progress - Round {round_num}:")
                print("-" * 20)
                print(f"Loss: {total_loss.item():.8f}")
                print(f"Training Accuracy: {train_accuracy:.8f}%")
                print("Training Metrics:")
                for metric, value in train_metrics.items():
                    print(f"{metric:15s}: {value:.8f}")
                print("\nEvaluation Accuracy: {eval_accuracy:.8f}%")
                if round_num % args.eval_iterations == 0:
                    print("Evaluation Metrics:")
                    for metric, value in eval_metrics.items():
                        print(f"{metric:15s}: {value:.8f}")
                print("-" * 20)

            total_loss = 0