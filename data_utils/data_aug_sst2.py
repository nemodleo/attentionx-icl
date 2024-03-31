import json
import pandas as pd
import numpy as np

columns = ['0', '1']
file_path = 'data/sst2/train_spaced_sst2.jsonl'
output_file_path = 'data/sst2/train_spaced_sst2.jsonl'


def rec_softmax(x):
    print(x)
    e_x = np.exp(x)
    e_x = 1 / e_x
    return e_x / e_x.sum(axis=0)


def process_jsonl(file_path, output_file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data)

    df[columns] = df[columns].apply(rec_softmax, axis=1)

    df.to_json(output_file_path, orient='records', lines=True)
    return df


if __name__ == '__main__':
    df = process_jsonl(file_path, output_file_path)
