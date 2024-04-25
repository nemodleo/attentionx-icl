import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse

from iclx.soft_label_generator.train_bert import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="sst2")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--sampling_rate", type=float, default=1.0)
    parser.add_argument("--max_token_len", type=int, default=512)
    parser.add_argument("--n_gpus", type=int, default=1)
    args = parser.parse_args()

    train(
        dataset=args.dataset,
        lr=args.lr,
        batch_size=args.batch_size,
        sampling_rate=args.sampling_rate,
        max_token_len=args.max_token_len,
        n_gpus=args.n_gpus,
        max_epochs=100,
    )
