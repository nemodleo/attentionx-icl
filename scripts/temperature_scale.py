import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse

from iclx.soft_label_generator.calculate_temperature import calculate_temperature


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_dataset_name", type=str) # dataset name from HF. ex) ICKD/agnews-bert-scaled
    parser.add_argument("--target_split", type=str) # train, valid, test
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--optimization_iter", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()

    calculate_temperature(
        infer_dataset_name=args.infer_dataset_name,
        target_split=args.target_split,
        output_folder=args.output_folder,
        max_iter=args.optimization_iter,
        batch_size=args.batch_size
    )
