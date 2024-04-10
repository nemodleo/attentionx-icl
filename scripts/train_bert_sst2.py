import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from iclx.soft_label_generator.train_bert import train


if __name__ == "__main__":
    train(
        dataset="sst2",
        max_epochs=100,
        n_gpus=8,
        batch_size=32,
        lr=2e-5,
    )
