import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from iclx.soft_label_generator.infer_bert import infer


if __name__ == "__main__":
    infer(
        checkpoint_path="/home/alan-k/workspace/fork/attentionx-icl/lightning_logs/version_0/checkpoints/epoch=14-val_loss=0.15.ckpt",
        dataset_name="sst2",
        batch_size=512,
        output_path="json_sst2.jsonl"
    )
