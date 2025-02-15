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

import torch
import torch.nn.functional as F
import argparse
from transformers import PreTrainedModel, PreTrainedTokenizerBase


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
            prompt = [
                {'role': 'system', 'content': test_loader.system_prompt},
                {'role': 'user', 'content': question}
            ]
            inputs = tokenizer.apply_chat_template(prompt, return_tensors="pt").to(device)
            outputs = model.generate(**inputs, max_length=args.max_completion_length)
            generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_answer = generated_answer[len(tokenizer.apply_chat_template(prompt)):]

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


def grpo_loss(
        model: PreTrainedModel,
        base_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        question: str,
        answer: str,
        device: str,
        round_num: int,
        training_log_dir: str, 
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



    ##############################
    # PRODUCE CHAINS AND SCORES  # 
    ##############################


    ##### **1. Generate Responses from Model and Base Model**
    with torch.inference_mode():



        # 1.  Prepare prompt
        # Setup actual text structure 
        prompt = [
            {'role': 'system', 'content': train_loader.system_prompt},
            {'role': 'user', 'content': question}
        ]

        # Get text prompt - and prompt ids/mask
        prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False)
        prompt_inputs = tokenizer(prompt_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        
        # Log the prompt to file
        log_file = os.path.join(training_log_dir, f'{round_num}_generations.txt')
        with open(log_file, 'w') as f:
            f.write('###### ORIGINAL PROMPT #####\n\n')
            f.write(prompt_text)
            f.write('\n\n#### ANS ####\n\n')
            f.write(answer)


        # Truncate prompt to max length and repeat for number of generations
        prompt_ids = prompt_ids[:, -args.max_prompt_length:]
        prompt_mask = prompt_mask[:, -args.max_prompt_length:]
        
        # Repeat for number of chains/generations
        prompt_ids = prompt_ids.repeat(args.num_chains, 1)
        prompt_mask = prompt_mask.repeat(args.num_chains, 1)



        # Tokenize prompt and send to device
        inputs = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True).to(device)

        # Repeat inputs for multiple answer generations
        inputs = {k: v.repeat(args.num_chains, 1) for k, v in inputs.items()}

        # Generate responses from the model (fine-tuned)
        outputs = model.generate(
            **inputs, max_length=args.max_completion_length, temperature=0.9, do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        # Decode responses and remove the original prompt from outputs
        responses = [tokenizer.decode(o, skip_special_tokens=True)[len(full_prompt):] for o in outputs]


    ##### **2. Compute Rewards & Normalize Advantages**
    all_scores = []
    for response in responses:
        total_score, metrics = evaluator.evaluate_answer(question, response, answer)
        all_scores.append(total_score)

        # Log scores for this response
        with open(log_file, 'a') as f:
            f.write(f"\n#### GENERATION {len(all_scores)} RESPONSE ####\n\n")
            f.write(response)
            f.write(f"\n\n#### GENERATION {len(all_scores)} SCORES ####\n")
            f.write(f"Total Score: {total_score}\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")

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
    # Add metrics counts to total metrics
    total_metrics["num_correct"] = num_correct
    total_metrics["num_total"] = len(responses)
    total_metrics["accuracy"] = num_correct / len(responses) if len(responses) > 0 else 0



    # Convert all scores to tensor and normalize advantage
    all_scores = torch.tensor(all_scores, device=device)
    with open(log_file, 'a') as f:
        f.write("\n#### SCORES ####\n")
        f.write(f"{all_scores}\n")

    mean_rewards = all_scores.view(-1, args.num_chains).mean(dim=1, keepdim=True)
    with open(log_file, 'a') as f:
        f.write("\n#### MEAN REWARDS ####\n")
        f.write(f"{mean_rewards}\n")

    std_rewards = all_scores.view(-1, args.num_chains).std(dim=1, keepdim=True)
    with open(log_file, 'a') as f:
        f.write("\n#### STD REWARDS ####\n")
        f.write(f"{std_rewards}\n")

    advantage = (all_scores - mean_rewards) / (std_rewards + 1e-4)
    with open(log_file, 'a') as f:
        f.write("\n#### ADVANTAGE ####\n")
        f.write(f"{advantage}\n")

    print(f"Raw rewards range: {all_scores.min().item():.3f} to {all_scores.max().item():.3f}")
    print(f"Advantage range: {advantage.min().item():.3f} to {advantage.max().item():.3f}")



    ######################
    # COMPUTE GRPO LOSS # 
    #####################

    # 1. Get the tokens from just the prompt  [1 question x question length]
    prompt_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(device)
    
    # 2. Get the responses from each generation [N generatation x max(generation_length)] (it is padded so all same length)
    response_ids = tokenizer(responses, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
    logits_to_keep = response_ids.size(1)  # we only need to compute the logits for the completion tokens

    # 3. Combine to be [N generation, prompt_length + response_length]
    combined_ids = torch.cat([prompt_ids.expand(response_ids.size(0), -1), response_ids], dim=1)

    # 4. Want to calculate a completion mask (because each response is padded - we dont care about padding at the end)
    # will use it logits producing (to not waste computation) and in computing final loss 
    is_eos = response_ids == tokenizer.eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
    eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
    completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
    prompt_mask = torch.ones((response_ids.size(0), prompt_ids.size(1)), dtype=torch.int, device=device)
    combined_attention_mask = torch.cat([prompt_mask, completion_mask], dim=1) 

    
    # # 5. Get logits of model - trim to just be responses 
    # logits = model(input_ids=combined_ids, attention_mask=combined_attention_mask).logits 
    # logits = logits[:, prompt_ids.size(1):, :]

    # # 6. Calculate per token log probabilities 
    # per_token_logps = []
    # for logits_row, input_ids_row in zip(logits, combined_ids[:, -logits_to_keep:]):
    #     log_probs = logits_row.log_softmax(dim=-1)
    #     token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
    #     per_token_logps.append(token_log_prob)
    # training_model_per_token_logps = torch.stack(per_token_logps)
    # print(f"Log probs range: {training_model_per_token_logps.min().item():.3f} to {training_model_per_token_logps.max().item():.3f}")

    # # 7. Now we need to also calculate the per token log probabilites for the refence model 
    # # at first will be same model of course - but then will diverge as training model is trained
    # with torch.inference_mode():
    #     base_logits = base_model(input_ids=combined_ids, attention_mask=combined_attention_mask).logits 
    #     base_logits = base_logits[:, prompt_ids.size(1):, :]

    #     # 5. Calculate log probabilities 
    #     per_token_logps = []
    #     for logits_row, input_ids_row in zip(base_logits, combined_ids[:, -logits_to_keep:]):
    #         log_probs = logits_row.log_softmax(dim=-1)
    #         token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
    #         per_token_logps.append(token_log_prob)
    #     ref_model_per_token_logps = torch.stack(per_token_logps)


    logits = model(input_ids=combined_ids, attention_mask=combined_attention_mask).logits 
    logits = logits[:, prompt_ids.size(1):, :]  # Keep only completion logits

    # 6. Calculate per token log probabilities 
    per_token_logps = []
    for logits_row, input_ids_row, mask_row in zip(logits, response_ids, completion_mask):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        # Apply mask to zero out padding
        token_log_prob = token_log_prob * mask_row
        per_token_logps.append(token_log_prob)
    training_model_per_token_logps = torch.stack(per_token_logps)

    # 7. Now for reference model (same changes)
    with torch.inference_mode():
        base_logits = base_model(input_ids=combined_ids, attention_mask=combined_attention_mask).logits 
        base_logits = base_logits[:, prompt_ids.size(1):, :]

        per_token_logps = []
        for logits_row, input_ids_row, mask_row in zip(base_logits, response_ids, completion_mask):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            token_log_prob = token_log_prob * mask_row
            per_token_logps.append(token_log_prob)
        ref_model_per_token_logps = torch.stack(per_token_logps)

    # 8. Now let compute KL divergence between the live model and the reference model
    per_token_kl = torch.exp(ref_model_per_token_logps - training_model_per_token_logps) - (ref_model_per_token_logps - training_model_per_token_logps) - 1
    print(f"KL range: {per_token_kl.min().item():.3f} to {per_token_kl.max().item():.3f}")
    
    # 9. Now compute the advantage gradient weighted by the per token difference 
    # Now in this case the model used to generate the tokens is precisly the one we are updated - the ratio will always be 1 
    # The advantage is currently [1, 16] - we need it to be [16, 609] to match per-token dimensions
    advantages = advantage.squeeze(0)  # Remove batch dimension to get [16]
    advantages = advantages.unsqueeze(1).expand(-1, training_model_per_token_logps.size(1))  # Shape: [16, 609]
    per_token_loss = torch.exp(training_model_per_token_logps - training_model_per_token_logps.detach()) * advantages
    # 10. Now combine with the weighted kl 
    per_token_loss = -(per_token_loss - args.kl_weight_beta * per_token_kl)

    # 11. Finally mask out the padded tokens and sume 
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    print(f"Per token loss range: {per_token_loss.min().item():.7f} to {per_token_loss.max().item():.7f}")
    print(f"Final loss: {loss.item():.7f}")
    #######################
    # Now just some logs #
    ######################
    mean_kl = (((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()).mean().item()
    total_metrics["total_loss"] = loss.item() 
    total_metrics["mean_kl_loss"] = mean_kl

    return loss, total_metrics

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
    parser.add_argument("--num_chains", type=int, default=16)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--max_completion_length", type=int, default=786)
    parser.add_argument("--num_train_iters", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7111994)
    parser.add_argument("--eval_iterations", type=int, default=100)
    parser.add_argument("--dataset_name", type=str, default="gsm8k")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--kl_weight_beta", type=float, default=0.04)
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
    
    # Setup network and reference model 
    model, tokenizer = llms.get_llm_tokenizer(args.model_name, device)
    base_model, _ = llms.get_llm_tokenizer(args.model_name, device)

    # Get data set 
    train_loader, test_loader = rldatasets.get_dataloaders(args.dataset_name)

    # Setup optimizer for trainer agent with GRPO config settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
        eps=1e-8
    )

    # Add linear warmup learning rate scheduler
    warmup_steps = int(0.18 * args.num_train_iters)
    def get_lr(step):
        if step < warmup_steps:
            return (step / warmup_steps)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=get_lr
    )

    # Create training logs directory
    train_log_dir = os.path.join(args.output_dir, 'training_logs')
    os.makedirs(train_log_dir, exist_ok=True)

    # Create training logs directory
    eval_log_dir = os.path.join(args.output_dir, 'eval_logs')
    os.makedirs(eval_log_dir, exist_ok=True)

    # Begin training 
    accumulated_loss = 0
    optimizer.zero_grad()
    train_metrics_total = {}
    for round_num in tqdm(range(args.num_train_iters), desc="Training Progress"):
    
        # # Evaluate on test set every so often 
        # if False: #round_num % args.eval_iterations == 0:
        #     eval_metrics, eval_accuracy = eval_on_test_set(model, tokenizer, test_loader, device, args, round_num)
        #     metrics_history['eval_metrics'][round_num] = eval_metrics
        #     metrics_history['eval_accuracy'][round_num] = eval_accuracy

        # if (round_num+1) % args.ref_model_sync_steps == 0:
        #     with torch.no_grad():
        #         for param, ref_param in zip(model.parameters(), base_model.parameters()):
        #             ref_param.data = args.ref_model_mixup_alpha * param.data + (1 - args.ref_model_mixup_alpha) * ref_param.data


        # Get next question
        question, answer = next(train_loader)

        # Do GRPO - generate chains, score, compute advantage, compute loss 
        total_loss, train_metrics = grpo_loss(model, base_model, tokenizer, question, answer, device, round_num, train_log_dir, args)
        # Add current learning rate to metrics
        train_metrics["learning_rate"] = scheduler.get_last_lr()[0]
        train_metrics_total[round_num] = train_metrics

        # Log metrics for this iteration
        with open(os.path.join(train_log_dir, "train_logs.json"), "w") as f:
            json.dump(train_metrics_total, f, indent=4)
        # Gradient accumulation
        total_loss = total_loss / args.gradient_accumulation_steps
        total_loss.backward()
        accumulated_loss += total_loss.item()

        # Step optimizer and scheduler
        if (round_num + 1) % args.gradient_accumulation_steps == 0:
            # Then do the actual clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()    
        scheduler.step()
