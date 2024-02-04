import os
import csv

import openai
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

openai.api_key = os.getenv("OPENAI_API_KEY")


PROMPT = """
Classify the sentiment of the sentence into two classes: "positive" or "negative".
If the sentence expresses a positive sentiment, use the word "positive" to indicate the sentiment.
If the sentence expresses a negative sentiment, use the word "negative" to indicate the sentiment.
Consider the overall tone and specific words used in the sentence.

Example1)
Sentence: A warm, funny, engaging film.
Sentiment: positive

Example2)
Sentence: A three-hour cinema master class.
Sentiment: negative

Example3)
Sentence: Brilliantly crafted and remarkably insightful.
Sentiment: positive

Example4)
Sentence: An utterly unconvincing plot.
Sentiment: negative

Consider the following sentence and classify its sentiment.
You can use the words "positive" or "negative" to indicate the sentiment.
Never use other words except "positive" or "negative".

Sentence: {sentence}
Sentiment: 
""".strip()


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

        prob_positive = token_prob_dict.get('positive', -1)
        prob_negative = token_prob_dict.get('negative', -1)

        return prob_positive, prob_negative
    except Exception as e:
        print(f"Error processing sentence: {sentence}. Error: {e}")
        return None, None
    

prob_positive, prob_negative = get_probs("great movie")
prob_positive, prob_negative

start_idx = 0
max_num = 10000
end_idx = start_idx + max_num

dataset = load_dataset('glue', 'sst2')

with open(f'ickd_sst2_probs_n{max_num}_s{start_idx}_e{end_idx}.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["index", "sentence", "label", "positive_prob", "negative_prob"])

    for idx, instance in enumerate(tqdm(dataset['train'])):
        if idx < start_idx:
            continue
        if idx >= start_idx + max_num:
            break

        index = instance['idx']
        sentence = instance['sentence']
        label = instance['label']
        prob_positive, prob_negative = get_probs(sentence)
        writer.writerow([index, sentence, label, prob_positive, prob_negative])
