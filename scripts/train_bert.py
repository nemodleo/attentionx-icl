import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse

from iclx.soft_label_generator.train_bert import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="sst2")
    args = parser.parse_args()
    dataset = args.dataset

    train(
        dataset=dataset,
        n_gpus=1,
        max_epochs=100,
        batch_size=32,
        lr=2e-5,
    )
