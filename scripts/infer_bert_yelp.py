import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from iclx.soft_label_generator.infer_bert import infer


if __name__ == "__main__":
    infer(
        checkpoint_path="",  # Add the path to the checkpoint
        dataset_name="yelp",
        batch_size=512,
        file_name="yelp-jsonl.jsonl"
    )
