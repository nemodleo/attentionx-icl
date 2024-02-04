import os
import csv
import asyncio

import fire
import openai
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

from .ickd_sst2_prompt import PROMPT

openai.api_key = os.getenv("OPENAI_API_KEY")


async def get_probs(sentence):
    try:
        messages = [
            {
                "role": "system",
                "content": "You are an helpful assistant."
            },
            {
                "role": "user",
                "content": PROMPT.format(sentence=sentence)
            }
        ]
        response = await penai.ChatCompletion.create(
            model="gpt-4-turbo-preview",
            messages=messages,
            max_tokens=1,
            logprobs=True,
            top_logprobs=5,
            n=1,
            stop=None,
        )
        
        # print("content:", response["choices"][0]["message"]["content"])
        # print("logprobs:", response["choices"][0]["logprobs"]["content"][0]["logprob"])
        # print("top_logprobs:", response["choices"][0]["logprobs"]["content"][0]["top_logprobs"])

        token_prob_dict = {}
        for top_logprob in response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]:
            token = top_logprob["token"].lower().strip()
            logprob = top_logprob["logprob"]
            if token not in token_prob_dict:
                token_prob_dict[token] = np.exp(logprob)
            else:
                token_prob_dict[token] += np.exp(logprob)

        positive_prob = token_prob_dict.get('positive', -1)
        negative_prob = token_prob_dict.get('negative', -1)

        return positive_prob, negative_prob
    except Exception as e:
        print(f"Error processing sentence: {sentence}. Error: {e}")
        return -1, -1


async def main_async(
    start_idx: int = 0,
    max_num: int = 10000,
):
    dataset = load_dataset('glue', 'sst2')

    end_idx = start_idx + max_num

    tasks = []
    for idx, instance in enumerate(tqdm(dataset['train'])):
        if idx < start_idx:
            continue
        if idx >= end_idx:
            break

        sentence = instance['sentence']
        tasks.append(asyncio.create_task(get_probs(sentence)))
    
    results = await asyncio.gather(*tasks)

    with open(f'ickd_sst2_probs_n{max_num}_s{start_idx}_e{end_idx}.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["index", "sentence", "label", "positive_prob", "negative_prob"])

        tasks = []
        for idx, instance in enumerate(tqdm(dataset['train'])):
            if idx < start_idx:
                continue
            if idx >= start_idx + max_num:
                break

            positive_prob, negative_prob = results[idx - start_idx]

            index = instance['idx']
            sentence = instance['sentence']
            label = instance['label']
            writer.writerow([index, sentence, label, positive_prob, negative_prob])


def main(
    start_idx: int = 0,
    max_num: int = 10000,
):
    asyncio.run(main_async(start_idx, max_num))


if __name__ == "__main__":
    fire.Fire(main)
