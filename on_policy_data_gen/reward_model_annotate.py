"""
python on_policy_data_gen/reward_model_annotate.py --generation_file /scratch/ssd004/scratch/snajafi/datasets/llama-3.2-1b-instruct-data/all_outputs.json --output_dir /scratch/ssd004/scratch/snajafi/datasets/llama-3.2-1b-instruct-data --reward_model /scratch/ssd004/scratch/snajafi/model-weights/ArmoRM-Llama3-8B-v0.1

python on_policy_data_gen/reward_model_annotate.py --generation_file /scratch/ssd004/scratch/snajafi/datasets/llama-3.2-1b-offline-preference-data/all_outputs.json --reward_model /scratch/ssd004/scratch/snajafi/model-weights/ArmoRM-Llama3-8B-v0.1 --output_dir /scratch/ssd004/scratch/snajafi/datasets/llama-3.2-1b-offline-preference-data
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import os
import argparse
import tqdm
import numpy as np
import datasets
import random

parser = argparse.ArgumentParser()
parser.add_argument("--generation_file", type=str, default="datasets/gemma2_ultrafeedback/all_outputs.json", help="Path to the output generation file")
parser.add_argument("--reward_model", type=str, default="RLHFlow/ArmoRM-Llama3-8B-v0.1", help="Path to reward model")
parser.add_argument("--output_dir", type=str, default="datasets/gemma2_ultrafeedback/", help="Path to output directory")
args = parser.parse_args()

print(args)

generation_file = args.generation_file
with open(generation_file, 'r') as f:
    output_data = json.load(f)


inputs = [data["prompt"] for data in output_data]
candidates_texts = [data["all_generated_responses"] for data in output_data]

model = AutoModelForSequenceClassification.from_pretrained(args.reward_model, 
                                                           device_map="cuda:0",
                                                           attn_implementation="flash_attention_2",
                                                           trust_remote_code=True, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(args.reward_model, use_fast=True)

for data in tqdm.tqdm(output_data):
    prompt = data["prompt"]
    candidates = data["all_generated_responses"]
    scores = []
    for candidate in candidates:
        messages = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": candidate}]
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = model(input_ids)
            score = output.score.float().item()
            scores.append(score)
    data["all_rm_scores"] = scores

file_name = os.path.basename(args.generation_file).split('.json')[0] + "_rm.json"
with open(os.path.join(args.output_dir, file_name), 'w') as f:
    json.dump(output_data, f, indent=4)

print(f"Annotated outputs saved to {os.path.join(args.output_dir, file_name)}")

# Binarize data: win = highest scoring reponse; lose = lowest scoring response
for data in output_data:
    chosen_idx = np.argmax(data["all_rm_scores"])
    rejected_idx = np.argmin(data["all_rm_scores"])
    chosen = []
    chosen.append({
        "role": "user",
        "content": data["prompt"]
    })
    chosen.append({
        "role": "assistant",
        "content": data["all_generated_responses"][chosen_idx]
    })
    rejected = []
    rejected.append({
        "role": "user",
        "content": data["prompt"]
    })
    rejected.append({
        "role": "assistant",
        "content": data["all_generated_responses"][rejected_idx]
    })
    data.update({
        "chosen": chosen,
        "rejected": rejected,
    })

output_file = os.path.basename(args.generation_file).split('.json')[0] + "_bin.json"
with open(os.path.join(args.output_dir, output_file), 'w') as f:
    json.dump(output_data, f, indent=4)
print(f"Binarized outputs saved to {output_file}")

random.shuffle(output_data)
test_output_data = output_data[:len(output_data) // 10]
train_output_data = output_data[len(output_data) // 10:]

# Convert the data to Hugging Face datasets format
test_dataset = datasets.Dataset.from_list(test_output_data)
test_output_path = os.path.join(args.output_dir, "test")
test_dataset.save_to_disk(test_output_path)
print(f"Binarized test dataset saved to {test_output_path}")

# Convert the data to Hugging Face datasets format
train_dataset = datasets.Dataset.from_list(train_output_data)
train_output_path = os.path.join(args.output_dir, "train")
train_dataset.save_to_disk(train_output_path)
print(f"Binarized train dataset saved to {train_output_path}")