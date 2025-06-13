# Custom Model for Exponential Fitting (user-provided)
import torch
import torch.nn as nn


class ExponentialModel(nn.Module):
    def __init__(self):
        super(ExponentialModel, self).__init__()
        # Initialize parameters a0, a1, b0, b1
        self.a_params = nn.Parameter(
            torch.tensor([-1, -10], dtype=torch.float64)
        )  # For a0, a1
        self.b_params = nn.Parameter(
            torch.tensor([1, 10], dtype=torch.float64)
        )  # For b0, b1

    def forward(self, x_input_dict):
        # x_input_dict is expected to have "rij" and "d" keys
        # Assuming x_input_dict["d"] and x_input_dict["rij"] are per-sample tensors (e.g., shape [N] or [N,1])
        # a_effective = a0*d + a1
        # b_effective = b0*d + b1
        # result = exp(a_effective * rij + b_effective)

        # Ensure operations are element-wise if inputs are 1D tensors per sample
        # If x_input_dict["d"] is shape (N,) or (N,1)
        a_eff = self.a_params[0] * x_input_dict["d"] + self.a_params[1]
        b_eff = self.b_params[0] * x_input_dict["d"] + self.b_params[1]

        return torch.exp(a_eff * x_input_dict["rij"] + b_eff)


class MaskedExponentialModel(nn.Module):
    def __init__(self):
        super(MaskedExponentialModel, self).__init__()
        self.a_params = nn.Parameter(torch.tensor([-1, -10], dtype=torch.float64))
        self.b_params = nn.Parameter(torch.tensor([1, 10], dtype=torch.float64))

    def forward(self, x_input_dict):
        # Create a copy to avoid modifying the original dict during evaluation
        input_dict = x_input_dict.copy()

        # 1. Pop the mask from the dictionary so it's not treated as a feature
        mask = input_dict.pop("mask")

        # --- The rest of your forward pass is the same ---
        a_eff = self.a_params[0] * input_dict["d"] + self.a_params[1]
        b_eff = self.b_params[0] * input_dict["d"] + self.b_params[1]

        result = torch.exp(a_eff * input_dict["rij"] + b_eff)

        # 2. Apply the mask to the output. This zeros out predictions for padded values.
        # The Y tensor is also padded with zeros, so these positions won't contribute to MSE loss.
        # This assumes your result and mask have shape (Batch, n_pairs, ...), so element-wise multiplication works.
        masked_result = result * mask

        return masked_result
