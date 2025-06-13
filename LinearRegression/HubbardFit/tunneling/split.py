import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset

from sklearn.model_selection import KFold
from sklearn.linear_model import LassoCV


from .loss import mse_criterion, R2_eval_metric
from .train import train_fold_lbfgs


class PaddedDictDataset(Dataset):
    def __init__(self, samples_X, samples_Y, max_pairs):
        # samples_X: list of dicts, e.g. [{'rij': tensor, 'd': tensor}, ...]
        # samples_Y: list of tensors
        self.samples_X = samples_X
        self.samples_Y = samples_Y
        self.max_pairs = max_pairs

    def __len__(self):
        return len(self.samples_Y)

    def __getitem__(self, idx):
        x_item_dict = self.samples_X[idx]
        y_item = self.samples_Y[idx]

        # --- Padding Logic ---
        padded_x_dict = {}
        original_len = 0
        for key, tensor in x_item_dict.items():
            original_len = tensor.shape[0]  # Assume shape is (n_pairs, ...)
            padding_size = self.max_pairs - original_len
            # Pad the first dimension (n_pairs)
            padded_x_dict[key] = nn.functional.pad(
                tensor, (0, 0, 0, padding_size), "constant", 0
            )

        # Also pad the Y tensor
        padding_size_y = self.max_pairs - y_item.shape[0]
        padded_y = nn.functional.pad(y_item, (0, 0, 0, padding_size_y), "constant", 0)

        # --- Mask Creation ---
        mask = torch.zeros(self.max_pairs, dtype=torch.bool)
        mask[:original_len] = True

        # Add mask to the dictionary to pass it to the model
        padded_x_dict["mask"] = mask

        return padded_x_dict, padded_y


def cross_validate_with_padded_dataset(
    model_class,
    # Pass the unified dataset instance
    full_dataset,
    l1_lambdas,
    k_folds=5,
    lbfgs_params=None,
    device="cpu",
):
    if lbfgs_params is None:
        lbfgs_params = {
            "lr": 0.005,
            "max_iter_per_step": 20,
            "epochs": 50,
            "tol_grad": 1e-7,
            "tol_change": 1e-9,
            "tol_loss_change": 1e-9,
        }

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    avg_val_r2_scores_per_lambda = []
    print("Starting Padded Dataset K-Fold Cross-Validation...")

    for l1_lambda in l1_lambdas:
        fold_specific_val_r2_scores = []
        # 2. Split the full dataset
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
            # 3. Create subsets for the fold
            train_subset = Subset(full_dataset, train_idx)
            val_subset = Subset(full_dataset, val_idx)

            # 4. Manually build full tensors for LBFGS from subsets
            train_X_list, train_Y_list = zip(
                *[train_subset[i] for i in range(len(train_subset))]
            )
            val_X_list, val_Y_list = zip(
                *[val_subset[i] for i in range(len(val_subset))]
            )

            # Consolidate list of dicts into dict of lists, then stack
            train_X_fold_dict = {
                key: torch.stack([d[key] for d in train_X_list])
                for key in train_X_list[0]
            }
            train_Y_fold_tensor = torch.stack(train_Y_list)

            val_X_fold_dict = {
                key: torch.stack([d[key] for d in val_X_list]) for key in val_X_list[0]
            }
            val_Y_fold_tensor = torch.stack(val_Y_list)

            # Move to device
            train_X_fold_dict = {k: v.to(device) for k, v in train_X_fold_dict.items()}
            train_Y_fold_tensor = train_Y_fold_tensor.to(device)
            val_X_fold_dict = {k: v.to(device) for k, v in val_X_fold_dict.items()}
            val_Y_fold_tensor = val_Y_fold_tensor.to(device)

            # 5. Proceed with your existing training and evaluation functions
            current_model_instance = model_class().to(device).double()

            # The model's forward pass needs to accept the mask!
            # Example: def forward(self, x_input_dict):
            #   mask = x_input_dict.pop('mask') # pop to avoid using it as a feature
            #   ... use mask to nullify padded values ...
            train_fold_lbfgs(
                current_model_instance,
                train_X_fold_dict,
                train_Y_fold_tensor,
                l1_lambda,
                lbfgs_params,
            )

            val_r2 = R2_eval_metric(
                current_model_instance, val_X_fold_dict, val_Y_fold_tensor
            )
            fold_specific_val_r2_scores.append(val_r2)

        avg_r2 = np.mean(fold_specific_val_r2_scores)
        avg_val_r2_scores_per_lambda.append(avg_r2)
        print(f"  L1 Lambda: {l1_lambda:.3e} -> Avg Validation R2: {avg_r2:.4f}")

    if not avg_val_r2_scores_per_lambda:
        print("Warning: No validation scores recorded.")
        return l1_lambdas[0] if l1_lambdas else 0.0, []

    # ... find and return best lambda
    best_lambda_idx = np.argmax(avg_val_r2_scores_per_lambda)
    best_l1_lambda = l1_lambdas[best_lambda_idx]

    print(
        f"\nBest L1 lambda from CV: {best_l1_lambda:.3e} (Avg R2: {avg_val_r2_scores_per_lambda[best_lambda_idx]:.4f})"
    )

    return best_l1_lambda, avg_val_r2_scores_per_lambda


def cross_validate_lbfgs(
    model_class,
    full_X_torch_dict,
    full_Y_torch_tensor,
    l1_lambdas,
    k_folds=5,
    lbfgs_params=None,
    device="cpu",
):
    if lbfgs_params is None:
        lbfgs_params = {
            "lr": 0.005,
            "max_iter_per_step": 20,
            "epochs": 50,
            "tol_grad": 1e-7,
            "tol_change": 1e-9,
            "tol_loss_change": 1e-9,
        }

    # KFold splits along the first dimension of the tensors
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    # full_Y_torch_tensor has shape (N, P, F_y), indices are for N
    indices = torch.arange(full_Y_torch_tensor.shape[0])

    avg_val_r2_scores_per_lambda = []
    print("Starting Cross-Validation...")

    for l1_lambda in l1_lambdas:
        fold_specific_val_r2_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
            # train_idx and val_idx select from the first dimension (N)
            train_X_fold_dict = {
                key: full_X_torch_dict[key][train_idx].to(device)
                for key in full_X_torch_dict
            }
            train_Y_fold_tensor = full_Y_torch_tensor[train_idx].to(device)

            val_X_fold_dict = {
                key: full_X_torch_dict[key][val_idx].to(device)
                for key in full_X_torch_dict
            }
            val_Y_fold_tensor = full_Y_torch_tensor[val_idx].to(device)

            current_model_instance = model_class().to(device)
            if hasattr(current_model_instance, "double"):
                current_model_instance.double()

            train_fold_lbfgs(
                model_instance=current_model_instance,
                train_X_dict=train_X_fold_dict,
                train_Y_tensor=train_Y_fold_tensor,
                l1_lambda=l1_lambda,
                lbfgs_params=lbfgs_params,
            )

            val_r2 = R2_eval_metric(
                current_model_instance, val_X_fold_dict, val_Y_fold_tensor
            )
            fold_specific_val_r2_scores.append(val_r2)

        avg_r2_for_this_lambda = (
            np.mean(fold_specific_val_r2_scores)
            if fold_specific_val_r2_scores
            else -float("inf")
        )
        avg_val_r2_scores_per_lambda.append(avg_r2_for_this_lambda)
        print(
            f"  L1 Lambda: {l1_lambda:.3e} -> Avg Validation R2: {avg_r2_for_this_lambda:.4f}"
        )

    if not avg_val_r2_scores_per_lambda:
        print("Warning: No validation scores recorded.")
        return l1_lambdas[0] if l1_lambdas else 0.0, []

    best_lambda_idx = np.argmax(avg_val_r2_scores_per_lambda)
    best_l1_lambda = l1_lambdas[best_lambda_idx]

    print(
        f"\nBest L1 lambda from CV: {best_l1_lambda:.3e} (Avg R2: {avg_val_r2_scores_per_lambda[best_lambda_idx]:.4f})"
    )
    return best_l1_lambda, avg_val_r2_scores_per_lambda
