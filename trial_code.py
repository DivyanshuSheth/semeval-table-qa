import os
import json
import pandas as pd
from datasets import load_dataset, Dataset

def example_postprocess(response: str, dataset: str, loader):
    # try:
        df = loader(dataset)
        lead = """
def answer(df):
    return """
        to_exec = "global ans\n" + lead + response.split("return")[2].split("\n")[0].strip().replace("[end of text]", "") + f"\nans = answer(df)"
        print("exec = ", to_exec)
        exec(
            to_exec
        )
        # no true result is > 1 line atm, needs 1 line for txt format
        to_return = ans.split("\n")[0] if "\n" in str(ans) else ans
        return to_return
    # except Exception as e:
    #     print(f"__CODE_ERROR__: {e}")
    #     return f"__CODE_ERROR__: {e}"
    

with open("/home/dasheth/unlearning/all_generations_20241203-141437.json", "r") as f:
    all_generations = json.load(f)

def load_sample(name):
    return pd.read_parquet(
        f"hf://datasets/cardiffnlp/databench/data/{name}/sample.parquet"
    )

def load_qa(**kwargs) -> Dataset:
    return load_dataset(
        "cardiffnlp/databench", **{"name": "qa", "split": "train", **kwargs}
    )

qa = load_qa(name="semeval", split="dev")
qa = qa.take(160)
i = 2
example = all_generations[i]
example_dataset = qa[i]["dataset"]
print("Example:", example)
result = example_postprocess(example, example_dataset, load_sample)
print("#" * 100)
print("Result:", result)
print("#" * 100)