import json
import tiktoken

def load_prompts_from_json(file_path):
    """Load prompts from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def count_tokens(prompt, tokenizer):
    """Count the number of tokens in a given prompt."""
    tokens = tokenizer.encode(prompt)
    return len(tokens)

def calculate_average_tokens(prompts, tokenizer):
    """Calculate the average number of tokens per prompt."""
    total_tokens = sum(count_tokens(prompt, tokenizer) for prompt in prompts)
    return total_tokens / len(prompts) if prompts else 0

def main(json_file_path):
    """Main function to compute average tokens from a JSON file of prompts."""
    # Load the tokenizer for GPT models
    tokenizer = tiktoken.get_encoding("o200k_base")  # Adjust model name as necessary

    # Load prompts from JSON
    prompts = load_prompts_from_json(json_file_path)

    # Calculate average number of tokens
    average_tokens = calculate_average_tokens(prompts, tokenizer)

    print(f"Average number of tokens per prompt: {average_tokens:.2f}")

# Example usage:
# main('/home/dasheth/unlearning/useful_runs/gpt-4o-mini-prompt4/all_prompts_20241203-230656.json')
# main('/home/dasheth/unlearning/useful_runs/gpt-4o-mini-prompt1/all_prompts_20241203-194355.json')
# main('/home/dasheth/unlearning/useful_runs/gpt-4o-mini-prompt2/all_prompts_20241203-201452.json')
main('/home/dasheth/unlearning/useful_runs/gpt-4o-mini-prompt3/all_prompts_20241203-220135.json')