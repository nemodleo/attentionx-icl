import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from iclx.soft_label_generator.infer_bert import infer


if __name__ == "__main__":
    infer(
        dataset_name="sst2",
        dataset_split="train",
        batch_size=512,
        output_path="result_sst2.csv"
    )
