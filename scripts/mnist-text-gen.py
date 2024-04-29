# https://huggingface.co/datasets/Fraser/mnist-text-default/blob/main/mnist-text-default.py

"""MNIST text dataset."""

from __future__ import absolute_import, division, print_function

import json
import os
import math

import numpy as np
import datasets

from datasets import load_dataset, DatasetDict
import torch
import torch.nn.functional as F


_DESCRIPTION = """\
MNIST dataset adapted to a text-based representation.

This allows testing interpolation quality for Transformer-VAEs.

System is heavily inspired by Matthew Rayfield's work https://youtu.be/Z9K3cwSL6uM

Works by quantising each MNIST pixel into one of 64 characters.
Every sample has an up & down version to encourage the model to learn rotation invarient features.

Use `.array_to_text(` and `.text_to_array(` methods to test your generated data.

Data format:
- text: (30 x 28 tokens, 840 tokens total): Textual representation of MNIST digit, for example:
```
00 down ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !
01 down ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !
02 down ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !
03 down ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !
04 down ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !
05 down ! ! ! ! ! ! ! ! ! ! ! ! ! % % % @ C L ' J a ^ @ ! ! ! !
06 down ! ! ! ! ! ! ! ! ( * 8 G K ` ` ` ` ` Y L ` ] Q 1 ! ! ! !
07 down ! ! ! ! ! ! ! - \ ` ` ` ` ` ` ` ` _ 8 5 5 / * ! ! ! ! !
08 down ! ! ! ! ! ! ! % W ` ` ` ` ` R N ^ ] ! ! ! ! ! ! ! ! ! !
09 down ! ! ! ! ! ! ! ! 5 H ; ` ` T # ! + G ! ! ! ! ! ! ! ! ! !
10 down ! ! ! ! ! ! ! ! ! $ ! G ` 7 ! ! ! ! ! ! ! ! ! ! ! ! ! !
11 down ! ! ! ! ! ! ! ! ! ! ! C ` P ! ! ! ! ! ! ! ! ! ! ! ! ! !
12 down ! ! ! ! ! ! ! ! ! ! ! # P ` 2 ! ! ! ! ! ! ! ! ! ! ! ! !
13 down ! ! ! ! ! ! ! ! ! ! ! ! ) ] Y I < ! ! ! ! ! ! ! ! ! ! !
14 down ! ! ! ! ! ! ! ! ! ! ! ! ! 5 ] ` ` > ' ! ! ! ! ! ! ! ! !
15 down ! ! ! ! ! ! ! ! ! ! ! ! ! ! , O ` ` F ' ! ! ! ! ! ! ! !
16 down ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! % 8 ` ` O ! ! ! ! ! ! ! !
17 down ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! _ ` _ 1 ! ! ! ! ! ! !
18 down ! ! ! ! ! ! ! ! ! ! ! ! ! ! , A N ` ` T ! ! ! ! ! ! ! !
19 down ! ! ! ! ! ! ! ! ! ! ! ! * F Z ` ` ` _ N ! ! ! ! ! ! ! !
20 down ! ! ! ! ! ! ! ! ! ! ' = X ` ` ` ` S 4 ! ! ! ! ! ! ! ! !
21 down ! ! ! ! ! ! ! ! & 1 V ` ` ` ` R 5 ! ! ! ! ! ! ! ! ! ! !
22 down ! ! ! ! ! ! % K W ` ` ` ` Q 5 # ! ! ! ! ! ! ! ! ! ! ! !
23 down ! ! ! ! . L Y ` ` ` ` ^ B # ! ! ! ! ! ! ! ! ! ! ! ! ! !
24 down ! ! ! ! C ` ` ` V B B % ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !
25 down ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !
26 down ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !
27 down ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !
```
- label: Just a number with the texts matching label.

"""

_CITATION = """\
@dataset{dataset,
    author = {Fraser Greenlee},
    year = {2021},
    month = {1},
    pages = {},
    title = {MNIST text dataset.},
    doi = {}
}
"""

_TRAIN_DOWNLOAD_URL = "https://raw.githubusercontent.com/Fraser-Greenlee/my-huggingface-datasets/master/data/mnist-text/train.json.zip"
_TEST_DOWNLOAD_URL = "https://raw.githubusercontent.com/Fraser-Greenlee/my-huggingface-datasets/master/data/mnist-text/test.json"

LABELS = list(range(10))
CUSTOM_METHODS = ['array_to_text', 'text_to_array']
IMG_SIZE = (30, 28)


class MnistText(datasets.GeneratorBasedBuilder):
    """MNIST represented by text."""

    def as_dataset(self, *args, **kwargs):
        f"""
            Return a Dataset for the specified split.

            Modified to add custom methods {CUSTOM_METHODS} to the dataset.
            This allows rendering the text as images & vice versa.
        """
        a_dataset = super().as_dataset(*args, **kwargs)
        for method in CUSTOM_METHODS:
            setattr(a_dataset, f'custom_{method}', getattr(self, method))
        return a_dataset

    @staticmethod
    def array_to_text(pixels: np.array):
        '''
            Takes a 2D array of pixel brightnesses and converts them to text.
            Uses 64 tokens to represent all brightness values.
        '''
        width = pixels.shape[0]
        height = pixels.shape[1]

        lines = []

        for y in range(height):
            split = ['%02d down' % y]

            for x in range(width):
                brightness = pixels[y, x]

                mBrightness = math.floor(brightness * 64)
                s = chr(mBrightness + 33)

                split.append(s)

            lines.append(' '.join(split))

        reversed = []
        for line in lines:
            reversed.insert(0, (line.replace(' down ', ' up ', 1)))

        return ['\n'.join(lines), '\n'.join(reversed)]

    @staticmethod
    def text_to_array(text: str):
        '''
            Takes a text sequences and tries to convert it into a 2D numpy array of brightnesses.
            If parts of the text don't match the format they will be skipped.
        '''
        lines = text.strip().split('\n')
        pixels = np.zeros((IMG_SIZE[1], IMG_SIZE[0] - 2))
        tokens = None
        for y in range(min(IMG_SIZE[1], len(lines))):
            line = lines[y].strip()
            tokens = line.split(' ')
            for i in range(2, min(IMG_SIZE[0], len(tokens))):
                token = tokens[i]
                if len(token) == 1:
                    tkn_v = (ord(token) - 33)
                    if tkn_v >= 0 and tkn_v <= 64:
                        pixels[y, i - 2] = tkn_v / 64

        if not lines:
            return pixels

        if tokens and len(tokens) > 1 and tokens[1] == 'up':
            pixels = pixels[::-1]

        return pixels
    
    @staticmethod
    def array_to_text_binary(pixels: np.array):
        '''
            Takes a 2D array of pixel brightnesses and converts them to text.
            Uses 64 tokens to represent all brightness values.
        '''
        width = pixels.shape[0]
        height = pixels.shape[1]

        lines = []

        for y in range(height):
            split = ['%02d down' % y]

            for x in range(width):
                brightness = pixels[y, x]

                mBrightness = math.floor(brightness * 64)
                # s = chr(mBrightness + 33)
                s = '1' if mBrightness > 0 else '0'

                split.append(s)

            lines.append(' '.join(split))

        reversed = []
        for line in lines:
            reversed.insert(0, (line.replace(' down ', ' up ', 1)))

        return ['\n'.join(lines), '\n'.join(reversed)]


    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    'label': datasets.features.ClassLabel(names=LABELS),
                    'text': datasets.Value("string"),
                }
            ),
            homepage="https://github.com/Fraser-Greenlee/my-huggingface-datasets",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": os.path.join(train_path, 'train.json')}
            ),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": test_path}),
        ]

    def _generate_examples(self, filepath):
        """Generate examples."""
        with open(filepath, encoding="utf-8") as json_lines_file:
            data = []
            for line in json_lines_file:
                data.append(json.loads(line))

            for id_, row in enumerate(data):
                yield id_, row


def make_small_mnist_text(x, return_binary=False):
    x = MnistText.text_to_array(x)
    x = torch.tensor(x.copy()).unsqueeze(0).unsqueeze(0)
    x = F.interpolate(x, size=(14, 14), mode='nearest-exact')
    x = x.squeeze(0).squeeze(0).numpy()
    if return_binary:
        return MnistText.array_to_text_binary(x)[0]
    return x


if __name__ == "__main__":
    dataset = load_dataset("Fraser/mnist-text-default")
    small_dataset_test = dataset['test'].map(
        lambda batch: {'label': batch['label'], 'text': make_small_mnist_text(batch['text'], return_binary=False)},
        batched=False,
    )
    small_dataset_train = dataset['train'].map(
        lambda batch: {'label': batch['label'], 'text': make_small_mnist_text(batch['text'], return_binary=False)},
        batched=False,
    )
    dataset = DatasetDict({'train': small_dataset_train, 'test': small_dataset_test})
    dataset.push_to_hub("ICKD/mnist-text-small")
    print(dataset)

    dataset = load_dataset("Fraser/mnist-text-default")
    small_dataset_test = dataset['test'].map(
        lambda batch: {'label': batch['label'], 'text': make_small_mnist_text(batch['text'], return_binary=True)},
        batched=False,
    )
    small_dataset_train = dataset['train'].map(
        lambda batch: {'label': batch['label'], 'text': make_small_mnist_text(batch['text'], return_binary=True)},
        batched=False,
    )
    dataset = DatasetDict({'train': small_dataset_train, 'test': small_dataset_test})
    dataset.push_to_hub("ICKD/mnist-text-small-binary")
    print(dataset)

