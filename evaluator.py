"""
Hold all kinds of evaluation criteria, right now just for gsmk8k 
"""
import re


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()
# Original reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    # print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses] 
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses] 
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

def evaluate_answer(question: str, generated_answer: str, ground_truth: str) -> tuple[float, dict[str, float]]:
    """
    Evaluates a generated answer against ground truth using multiple reward functions.
    
    Args:
        question: The input question
        generated_answer: The model's generated answer
        ground_truth: The ground truth answer
    
    Returns:
        tuple[float, dict[str, float]]: Total reward score and dictionary of individual scores
    """
    # Format inputs as expected by reward functions
    mock_prompt = [{'content': question}]
    mock_completion = [{'content': generated_answer}]
    mock_prompts = [mock_prompt]
    mock_completions = [mock_completion]
    mock_answer = [ground_truth]

    # Run all reward functions
    correctness_score = correctness_reward_func(mock_prompts, mock_completions, mock_answer)[0]
    int_score = int_reward_func(mock_completions)[0] 
    strict_format_score = strict_format_reward_func(mock_completions)[0]
    soft_format_score = soft_format_reward_func(mock_completions)[0]
    xml_count_score = xmlcount_reward_func(mock_completions)[0]

    # Create scores dictionary
    scores = {
        'correctness': correctness_score,  # Up to 2.0
        'int_format': int_score,          # Up to 0.5
        'strict_format': strict_format_score, # Up to 0.5
        'soft_format': soft_format_score,    # Up to 0.5
        'xml_count': xml_count_score         # Up to 0.5
    }

    # Sum up total score
    total_score = sum(scores.values())

    return total_score, scores
