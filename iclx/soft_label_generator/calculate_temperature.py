from functools import partial
import json
import os

from datasets import load_dataset
from loguru import logger
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


def temperature_scale(logits, temperature):
    """
    Perform temperature scaling on logits
    """
    # Expand temperature to match the size of logits
    reshaped_temperature = temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
    return logits / reshaped_temperature


def calculate_temperature_from_dataloader(dataloader, max_iter, device):
    """
    Tune the tempearature of the model (using the validation set).
    We're going to set it to optimize NLL.
    dataloader (DataLoader): data loader
    """
    # Create a tensor and make sure it's on the correct device before wrapping it as a Parameter
    initial_tensor = torch.ones(1)
    initial_tensor = initial_tensor.to(device)  # Move tensor to the device first
    temperature = nn.Parameter(initial_tensor)  # Create Parameter from the already device-assigned tensor

    # Set up the optimizer with the temperature parameter
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)

    nll_criterion = nn.CrossEntropyLoss().to(device)
    ece_criterion = _ECELoss().to(device)

    # First: collect all the logits and labels for the validation set
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for probs, label in dataloader:
            logits = torch.log(probs)
            logits_list.append(logits)
            labels_list.append(label)
        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device)

    # Calculate NLL and ECE before temperature scaling
    before_temperature_nll = nll_criterion(logits, labels).item()
    before_temperature_ece = ece_criterion(logits, labels).item()
    logger.info('Before temperature - NLL: %.6f, ECE: %.6f' % (before_temperature_nll, before_temperature_ece))

    # Next: optimize the temperature w.r.t. NLL
    def eval():
        optimizer.zero_grad()
        loss = nll_criterion(temperature_scale(logits, temperature), labels)
        loss.backward()
        return loss
    optimizer.step(eval)

    # Calculate NLL and ECE after temperature scaling
    after_temperature_nll = nll_criterion(temperature_scale(logits, temperature), labels).item()
    after_temperature_ece = ece_criterion(temperature_scale(logits, temperature), labels).item()
    logger.info('Optimal temperature: %.6f' % temperature.item())
    logger.info('After temperature - NLL: %.6f, ECE: %.6f' % (after_temperature_nll, after_temperature_ece))

    return temperature.detach().cpu().numpy().item()


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


class WrapperDataset(Dataset):
    def __init__(self, dataset, label_col="label"):
        super().__init__()
        self.dataset = dataset
        self.label_col = label_col

    def __getitem__(self, idx):
        example = self.dataset[idx]
        probs = []
        i = 0
        while str(i) in example.keys():
           probs.append(float(example[str(i)]))
           i+=1
        return (torch.tensor(probs), example[self.label_col])

    def __len__(self):
        return len(self.dataset)




def calculate_temperature(infer_dataset_name: str,
                          target_split: str,
                          output_folder: str,
                          max_iter: int,
                          batch_size: int):
    os.makedirs(output_folder, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    infer_dataset = load_dataset(infer_dataset_name)
    dataset = WrapperDataset(infer_dataset[target_split])
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=32)
    temp = calculate_temperature_from_dataloader(dataloader, max_iter, device)

    def scale_row(row, temperature):
        i=0
        probs = []
        while str(i) in row.keys():
            probs.append(float(row[str(i)]))
            i+=1
        logits = np.log(probs) / temperature
        scaled_res = np.exp(logits - np.max(logits))
        scaled_res /= np.sum(scaled_res)

        for i in range(len(scaled_res)):
            row[str(i)] = scaled_res[i]
        return row

    def save_to_jsonl(dataset, filename):
        with open(filename, 'w', encoding='utf-8') as file:
            for record in dataset:
                json_record = json.dumps(record, ensure_ascii=False)
                file.write(json_record + '\n')

    for split, dataset in infer_dataset.items():
        dataset = dataset.map(partial(scale_row, temperature=temp), load_from_cache_file=False)
        save_path = f"{output_folder}/{split}.jsonl"
        save_to_jsonl(dataset, save_path)
        logger.info(f"Saved {split} split to {save_path}")
