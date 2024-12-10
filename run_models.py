import pandas as pd
import subprocess
import shlex
import zipfile
import os
import json
import datetime
import re
from vllm import LLM, SamplingParams
from ftfy import fix_text
from transformers import AutoTokenizer

from openai import OpenAI
from tqdm import tqdm
from databench_eval import Runner, Evaluator, utils
from prompts import prompt_0, prompt_1, prompt_2, prompt_3, prompt_4, prompt_5, prompt_6, prompt_7, prompt_8

def clean_and_encode_text(text):
    fixed_text = fix_text(text)
    return fixed_text

def call_gguf_model(prompts):
    results = []
    for p in prompts:
        escaped = p.replace('"', '\\"')
        cmd = f'llama-cli -m ./models/stable-code-3b.Q4_K_M.gguf -p "{escaped}" -c 1024 -n 128'
        args = shlex.split(cmd)
        try:
            result = subprocess.run(args, capture_output=True, text=True, check=True)
            results.append(result.stdout)
        except Exception as e:
            results.append(f"__CODE_GEN_ERROR__: {e.stderr}")

    return results

def initialize_vllm_model(model_path):
    """Initialize the vLLM model."""
    # if model_path.startswith("Qwen"):
    return LLM(model=model_path,
                seed=42,
                download_dir="/data/datasets/hf_cache",
                tensor_parallel_size=1)

def call_vllm_model(prompts):#, vllm_model=vllm_model, tokenizer=tokenizer):
    """Generate responses using vLLM."""
    results = []
    samplingparams = SamplingParams(temperature=0.8, max_tokens=500)
    cleaned_prompts = []
    for i, prompt in enumerate(prompts):
        prompt = clean_and_encode_text(prompt)
        if not model_name.startswith("bigcode"):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            cleaned_prompts.append(formatted_prompt)
        else:
            cleaned_prompts.append(prompt)
    outputs = vllm_model.generate(cleaned_prompts, samplingparams)
    for output in tqdm(outputs):
        # print("Prompt = ", prompt)
        # try:
        response = output.outputs[0].text
        results.append(response.strip())  # Get the first output and strip whitespace
        # except Exception as e:
        #     results.append(f"__CODE_GEN_ERROR__: {str(e)}")
    all_prompts.extend(cleaned_prompts)
    all_generations.extend(results)
    return results

def chat_completions(prompts, max_tokens=500):
  api_key = os.environ.get('OPENAI_API_KEY')
  client = OpenAI(api_key=api_key)
  generations = []
  cleaned_prompts = []
  for i, prompt in enumerate(prompts):
    prompt = clean_and_encode_text(prompt)
    cleaned_prompts.append(prompt)
  for prompt in tqdm(prompts):
    response = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=max_tokens
    )
    generations.append(response.choices[0].message.content.strip())  # Remove leading/trailing whitespace
  all_prompts.extend(cleaned_prompts)
  all_generations.extend(generations)
  return generations


##############################################################
##############################################################

os.environ["HF_HOME"] = "/data/datasets/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/data/datasets/hf_cache"
model_name = "bigcode/starcoder2-7b"
eval_dataset = "test"
# model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
# model_name = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
# model_name = "codellama/CodeLlama-7b-Instruct-hf"
# model_name="gpt-4o-mini"
if model_name.startswith("gpt"):
    model_call_function=chat_completions
elif model_name.startswith("Qwen") or model_name.startswith("deepseek") or model_name.startswith("bigcode") or model_name.startswith("codellama"):
    model_call_function=call_vllm_model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vllm_model = initialize_vllm_model(model_name)
else:
    raise ValueError("Model not supported")
prompt_to_use = prompt_6 # prompt_6 for pretrained, prompt_4 for instruction-tuned
all_prompts = []
all_generations = []
all_execs = []

##############################################################
##############################################################


def example_generator(row: dict) -> str:
    """IMPORTANT:
    **Only the question and dataset keys will be available during the actual competition**.
    You can, however, try to predict the answer type or columns used
    with another modeling task if that helps, then use them here.
    """
    dataset = row["dataset"]
    question = row["question"]
    df = utils.load_table(dataset)
    formatted_prompt = prompt_to_use.format(question=question, row_type=row["type"], list_df_columns=list(df.columns))
    return formatted_prompt

def extract_solution(input_string):
    # Define the regex pattern to match the desired format
    pattern = r'\{\s*\"solution\":\s*\"(.*?)\"\s*\}'
    # Search for the first occurrence of the pattern
    match = re.search(pattern, input_string)
    # If a match is found, return the value of 'solution'
    if match:
        return match.group(1)
    return None

def postprocess_prompt_1(response: str, dataset: str, loader):
    df = loader(dataset)
    lead = """
def answer(df):
    return """
    response_formatted = response.replace("```json\n", "").replace("```", "").strip()
    # response_formatted = extract_solution(response_formatted)
    print(response_formatted)
    try:    
        # generated_code = json.loads(response_formatted)["solution"].replace("return ", "")
        generated_code = extract_solution(response_formatted).replace("return ", "")
        to_exec = "global ans\n" + lead + generated_code + f"\nans = answer(df)"
        exec(
            to_exec
        )
        all_execs.append(to_exec)
        # print("exec = ", to_exec)
        # print("ans = ", ans)
        return ans
    except Exception as e:
        all_execs.append(response_formatted)
        return f"__CODE_ERROR__: {e}"

def example_postprocess(response: str, dataset: str, loader):
    try:
        df = loader(dataset)
        lead = """
def answer(df):
    return """
        exec(
            "global ans\n"
            + lead
            + response.split("return")[2]
            .split("\n")[0]
            .strip()
            .replace("[end of text]", "")
            + f"\nans = answer(df)"
        )
        # no true result is > 1 line atm, needs 1 line for txt format
        return ans.split("\n")[0] if "\n" in str(ans) else ans
    except Exception as e:
        return f"__CODE_ERROR__: {e}"


def main():
    qa = utils.load_qa(name="semeval", split="dev")
    # qa = qa.shuffle(seed=42)
    qa_split = qa.train_test_split(test_size=0.5, shuffle=False)
    qa_dev = qa_split["train"]
    qa_test = qa_split["test"]
    qa.to_csv("/home/dasheth/unlearning/data/qa_all.csv")
    qa_dev.to_csv("/home/dasheth/unlearning/data/qa_dev.csv")
    qa_test.to_csv("/home/dasheth/unlearning/data/qa_test.csv")
    if eval_dataset == "dev":
        qa = qa_dev
    elif eval_dataset == "test":
        qa = qa_test
    # qa_valtest = qa.train_test_split(test_size=0.5, seed=42, shuffle=True)
    # qa_val = qa_valtest["train"]
    # qa_test = qa_valtest["test"]
    # print(f"Validation set size: {len(qa_val)}")
    # print(f"Test set size: {len(qa_test)}")
    evaluator = Evaluator(qa=qa)

    runner_lite = Runner(
        # model_call=call_gguf_model,
        model_call=model_call_function,
        # model_call=chat_completions,
        prompt_generator=example_generator,
        postprocess=lambda response, dataset: postprocess_prompt_1(
            response, dataset, utils.load_sample
        ),
        qa=qa,
        batch_size=32,
    )

    datetime_now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.mkdir("/home/dasheth/unlearning/{}".format(datetime_now))
    output_dir = "/home/dasheth/unlearning/{}".format(datetime_now)
    # responses = runner.run(save="predictions.txt")
    responses_lite = runner_lite.run(save=os.path.join(output_dir, "predictions_lite_{}.txt".format(datetime_now)))
    # print(f"DataBench accuracy is {evaluator.eval(responses)}")  # ~0.15
    accuracy = evaluator.eval(responses_lite, lite=True)
    print(f"DataBench_lite accuracy is {accuracy}")  # ~0.07

    with open(os.path.join(output_dir, "all_prompts_{}.json".format(datetime_now)), "w") as f:
        json.dump(all_prompts, f, indent=4)

    with open(os.path.join(output_dir, "all_generations_{}.json".format(datetime_now)), "w") as f:
        json.dump(all_generations, f, indent=4)

    with open(os.path.join(output_dir, "all_execs_{}.json".format(datetime_now)), "w") as f:
        json.dump(all_execs, f, indent=4)

    with open(os.path.join(output_dir, "accuracy_{}.txt".format(datetime_now)), "w") as f:
        f.write("Model used is {}".format(model_name) + "\n")
        f.write("DataBench_lite accuracy is {}".format(accuracy) + "\n")
        f.write("Dataset split: {}".format(eval_dataset) + "\n")
        f.write("\nPrompt used is:\n\n{}".format(fix_text(prompt_to_use)) + "\n")

if __name__ == "__main__":
    main()