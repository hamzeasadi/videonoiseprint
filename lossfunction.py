from typing import Optional, List
import os
import math

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch import Tensor
import numpy as np


# dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PI = torch.tensor(math.pi)




class AdaMagfaceLoss(nn.Module):
    """
    MagfaceLoss with Adaptive Scale strategy

    References:

        - https://arxiv.org/pdf/2103.06627.pdf
    """

    def __init__(self, NumClasses, InputFeatures, dev:torch.device) -> None:
        super().__init__()
        # From the output size of the Network
        self.NumClasses = NumClasses
        self.InputFeatures = InputFeatures
        self.dev = dev
        # Multiplier might be set in the range of [0.5 , 2] in ternarization/quantization, Default is 1.0
        self.S = math.sqrt(2)
        if self.NumClasses > 1:
            self.S = self.S * math.log(self.NumClasses - 1)
        # Maximum allowed scale for Arcface Loss
        self.S_Max = math.sqrt(2) * self.S

        self.LowerMargin = 0.35
        self.UpperMargin = 0.75
        self.LowerBound = 0
        self.UpperBound = 90

        self.EasyMargin = True

        # self.weights = torch.nn.Parameter(torch.Tensor(self.NumClasses, self.InputFeatures))
        self.weights = torch.nn.Parameter(torch.Tensor(self.NumClasses, self.InputFeatures), requires_grad=True)
        self.softmax = CrossEntropyLoss()
        torch.nn.init.xavier_uniform_(self.weights)
        # self.weights.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        # torch.nn.init.kaiming_normal_(self.weights)

    # Adapted from https://github.com/IrvingMeng/MagFace/blob/main/models/magface.py
    def _margin(self, features_mag):
        """
            generates Magnitude-aware adaptive margin
        """
        margin = (self.UpperMargin - self.LowerMargin) / \
                 (self.UpperBound - self.LowerBound) * (features_mag - self.LowerBound) + self.LowerMargin
        return margin

    def forward(self, outputs, labels,step: Optional[int] = None) -> torch.Tensor:
        """
        Perform a forward pass of the loss function. Gather metrics to track loss values during training.

        Args:
            outputs: Network output dictionary
            labels: Ground truth dictionary
            step: Current Solver step

        Returns:
            Loss
        """

        # Allow loss function inputs to be dictionaries containing metadata
        if isinstance(outputs, dict):
            features = outputs["data"]
        else:
            features = outputs

        if isinstance(labels, dict):
            class_labels = labels["label"]
        else:
            class_labels = labels

        batch_size = features.size(0)
        Magnitude_Mean = torch.mean(torch.norm(features, p=2, dim=1))  # type: ignore

        features_mag = torch.norm(features, dim=1, keepdim=True).clamp(self.LowerBound, self.UpperBound)

        Magnitude_Variance = torch.var(features_mag, unbiased=False)

        Margin = self._margin(features_mag)  # To generate the adaptive margin batch-wise

        # Constants to speed up forward pass
        self.cos_m = torch.cos(Margin)
        self.sin_m = torch.sin(Margin)
        self.th = torch.cos(PI - Margin)
        self.mm = torch.sin(PI - Margin) * Margin

        # cos(theta_i)
        cosine = torch.nn.functional.linear(torch.nn.functional.normalize(features),
                                            torch.nn.functional.normalize(self.weights))  # cos(theta_i)
        # cosine = cosine.clamp(-1, 1)  # for numerical stability
        theta = torch.acos(cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7))

        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))  # sin(theta_i)

        # Convert to one hot labels

        one_hot_labels = torch.zeros(batch_size, self.NumClasses, device=self.dev)
        one_hot_labels[torch.arange(batch_size), class_labels] = 1

        cosine_s_exp = torch.exp(self.S * cosine)
        if cosine_s_exp.dtype != cosine.dtype:
            cosine_s_exp = cosine_s_exp.to(cosine.dtype)

        # Adaptive Computation of Scale parameter
        with torch.no_grad():  # type: ignore
            B_avg = torch.where(one_hot_labels < 1, cosine_s_exp, torch.zeros_like(cosine))
            B_avg = torch.sum(B_avg) / batch_size
            theta_med = torch.median(theta[one_hot_labels == 1])
            self.S = float(torch.log(B_avg) / torch.cos(torch.min(math.pi / 4 * torch.ones_like(theta_med),
                                                                  theta_med)))  # Cast to a float to ensure self.S maintains the same type
            if self.S > self.S_Max:
                self.S = self.S_Max

        # cos(theta_i+m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # Ensure that phi dtype matches the input dtype.
        # This is used when Auto Mixed Precision is enabled so that the float32 tensor gets downcast to float16 to match the network output.
        if phi.dtype != cosine.dtype:
            phi = phi.to(cosine.dtype)

        if self.EasyMargin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # only apply the margin to the corresponding logit for each class
        logits = (one_hot_labels * phi) + ((1.0 - one_hot_labels) * cosine)

        # Scale by s
        logits = logits * self.S

        # Compute loss
        loss: Tensor = self.softmax(logits, class_labels)

        return loss
    

if __name__ == "__main__":
    torch.manual_seed(42)
    print(os.path.basename(__file__))
    weights = torch.nn.Parameter(torch.Tensor(3, 3))
    print(weights)
    torch.nn.init.xavier_uniform_(weights)
    print(weights)