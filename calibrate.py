import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sacred import Experiment

EXPERIMENT_NAME = "calibrate"
ex = Experiment(EXPERIMENT_NAME, ingredients=[])


# https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
class ECELoss(nn.Module):
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
        super(ECELoss, self).__init__()
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


def plot_calibration_error(logits, targets):
    confidences = F.softmax(logits, -1).max(-1).values.detach().numpy()
    accuracies = logits.argmax(-1).eq(targets).numpy()
    print(confidences)
    print(accuracies)

    n_bins = 15
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    max_err = 0.0

    plot_acc = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) * (confidences <= bin_upper)
        prop_in_bin = in_bin.astype(np.float32).mean()

        if prop_in_bin > 0.0:
            accuracy_in_bin = accuracies[in_bin].astype(np.float32).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            if np.abs(avg_confidence_in_bin - accuracy_in_bin) > max_err:
                max_err = np.abs(avg_confidence_in_bin - accuracy_in_bin)

            plot_acc.append(accuracy_in_bin)
        else:
            plot_acc.append(0.0)

    plt.bar(
        bin_lowers, plot_acc, bin_uppers[0], align="edge", linewidth=1, edgecolor="k"
    )
    plt.plot([0.0, 1.0], [0.0, 1.0], c="orange", lw=2)
    plt.text(
        0.02,
        0.93,
        "acc:  {:0.4f}\nece:  {:0.4f}\nmce: {:0.4f}".format(
            accuracies.astype(np.float32).mean(), ece, max_err
        ),
        fontsize=16,
    )

    plt.xlabel("confidence")
    plt.ylabel("accuracy")
    plt.savefig("temp.png", bbox_inches="tight")


@ex.config
def get_config():
    # where evaluation runs are located
    run_dir = "runs/test"

    job_id = None


@ex.automain
def main(run_dir, job_id, _run):
    assert job_id is not None, "must specify a job id"

    results = torch.load(
        os.path.join(run_dir, str(job_id), "results.pth"), map_location="cpu"
    )

    logits = torch.cat([result["stats"]["logits"] for result in results], 0)
    targets = torch.cat([result["stats"]["targets"] for result in results], 0)

    ece_module = ECELoss()
    print(ece_module.forward(logits, targets))

    plot_calibration_error(logits, targets)

