import os
import csv

import fire
import openai
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

from ickd_sst2_prompt import PROMPT

openai.api_key = os.getenv("OPENAI_API_KEY")


def get_probs(sentence):
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
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
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

        prob_positive = token_prob_dict.get('positive', -1)
        prob_negative = token_prob_dict.get('negative', -1)

        return prob_positive, prob_negative
    except Exception as e:
        print(f"Error processing sentence: {sentence}. Error: {e}")
        return -1, -1


def main(
    start_idx: int = 0,
    max_num: int = 1000000,
):
    dataset = load_dataset('glue', 'sst2')
    train_dataset = dataset['train']

    end_idx = min(start_idx + max_num, len(train_dataset))
    max_num = end_idx - start_idx
    with open(f'poc/openai_api/ickd__gpt-3-5-turbo__sst2___n{max_num}_s{start_idx}_e{end_idx}.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["index", "sentence", "label", "positive_prob", "negative_prob"])

        for idx, instance in enumerate(tqdm(train_dataset)):
            if idx < start_idx:
                continue
            if idx >= start_idx + max_num:
                break

            index = instance['idx']
            sentence = instance['sentence']
            label = instance['label']
            prob_positive, prob_negative = get_probs(sentence)
            writer.writerow([index, sentence, label, prob_positive, prob_negative])


if __name__ == "__main__":
    fire.Fire(main)
