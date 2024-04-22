import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse

from iclx.soft_label_generator.infer_bert import infer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--dataset", type=str, default="sst2")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_token_len", type=int, default=512)
    parser.add_argument("--file_name", type=str, default="sst2-train.jsonl")
    args = parser.parse_args()

    infer(
        checkpoint_path=args.checkpoint_path,
        dataset=args.dataset,
        batch_size=args.batch_size,
        max_token_len=args.max_token_len,
        file_name=args.file_name,
    )
