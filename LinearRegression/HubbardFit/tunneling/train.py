import numpy as np

import torch
from torch import nn
import torch.optim as optim

from sklearn.model_selection import KFold
from sklearn.linear_model import LassoCV

from .loss import mse_criterion, R2_eval_metric


def train_fold_lbfgs(
    model_instance, train_X_dict, train_Y_tensor, l1_lambda, lbfgs_params
):
    model_instance.train()
    optimizer = optim.LBFGS(
        model_instance.parameters(),
        lr=lbfgs_params.get("lr", 0.005),
        max_iter=lbfgs_params.get("max_iter_per_step", 20),
        tolerance_grad=lbfgs_params.get("tol_grad", 1e-7),
        tolerance_change=lbfgs_params.get("tol_change", 1e-9),
        history_size=lbfgs_params.get("history_size", 100),
    )

    def closure():
        optimizer.zero_grad()
        outputs = model_instance(train_X_dict)
        data_loss = mse_criterion(outputs, train_Y_tensor)

        l1_penalty_val = 0.0
        if l1_lambda > 0:
            for param in model_instance.parameters():
                l1_penalty_val += torch.abs(param.reshape(-1)).sum()

        total_loss = data_loss + l1_lambda * l1_penalty_val
        total_loss.backward()
        return total_loss

    num_epochs = lbfgs_params.get("epochs", 50)
    tol_loss_change = lbfgs_params.get("tol_loss_change", 1e-9)
    loss_old = torch.inf

    for epoch in range(num_epochs):
        loss_new = optimizer.step(closure)
        if loss_new is None:  # Should not happen
            print("Warning: LBFGS step returned None in fold training.")
            break
        current_loss_val = loss_new.item()
        if abs(loss_old - current_loss_val) < tol_loss_change and epoch > 0:
            break
        loss_old = current_loss_val
    return model_instance


def train_final_model_with_padded_dataset(
    model_class,
    full_dataset,  # <-- The PaddedDictDataset instance containing ALL data
    best_l1_lambda,
    lbfgs_params=None,
    device="cpu",
):
    """
    Trains the final model on the entire padded dataset using the best L1 lambda.
    """
    if lbfgs_params is None:
        lbfgs_params = {
            "lr": 0.005,
            "max_iter_per_step": 50,
            "epochs": 200,
            "tol_grad": 1e-9,
            "tol_change": 1e-11,
            "tol_loss_change": 1e-11,
        }

    print(f"\nTraining final model with best L1 lambda: {best_l1_lambda:.3e}")

    # --- Data Preparation Step ---
    print("Preparing full padded dataset for final training...")
    # 1. Iterate through the full_dataset to get all padded samples
    all_X_list, all_Y_list = zip(*[full_dataset[i] for i in range(len(full_dataset))])

    # 2. Consolidate list of dicts into a single dict of stacked tensors
    # This dictionary will now include the 'mask' tensor.
    full_X_padded_dict = {
        key: torch.stack([d[key] for d in all_X_list]) for key in all_X_list[0]
    }
    full_Y_padded_tensor = torch.stack(all_Y_list)

    # 3. Move all data to the correct device
    X_data_device_dict = {
        key: val.to(device) for key, val in full_X_padded_dict.items()
    }
    Y_data_device_tensor = full_Y_padded_tensor.to(device)

    # --- Model Training Step (largely unchanged) ---
    final_model_instance = model_class().to(device)
    if hasattr(final_model_instance, "double"):
        final_model_instance.double()

    # Your existing training function works perfectly with the padded & masked data
    trained_final_model = train_fold_lbfgs(
        model_instance=final_model_instance,
        train_X_dict=X_data_device_dict,
        train_Y_tensor=Y_data_device_tensor,
        l1_lambda=best_l1_lambda,
        lbfgs_params=lbfgs_params,
    )

    # --- Final Evaluation (unchanged) ---
    trained_final_model.eval()
    with torch.no_grad():
        predictions = trained_final_model(X_data_device_dict)
        final_loss_mse_part = mse_criterion(predictions, Y_data_device_tensor)

        l1_penalty_val = 0.0
        if best_l1_lambda > 0:
            for param in trained_final_model.parameters():
                l1_penalty_val += torch.abs(param.reshape(-1)).sum()
        total_final_loss = final_loss_mse_part + best_l1_lambda * l1_penalty_val
        final_r2 = R2_eval_metric(
            trained_final_model, X_data_device_dict, Y_data_device_tensor
        )

    print(f"Final model training finished.")
    print(f"  Parameters (a_params): {trained_final_model.a_params.data.cpu().numpy()}")
    print(f"  Parameters (b_params): {trained_final_model.b_params.data.cpu().numpy()}")
    print(f"  Final Model Total Loss (MSE_norm + L1): {total_final_loss.item():.6f}")
    print(f"  Final Model Normalized MSE part: {final_loss_mse_part.item():.6f}")
    print(f"  Final Model R2 on all data: {final_r2:.4f}")
    return trained_final_model


def train_final_model_lbfgs(
    model_class,
    full_X_torch_dict,
    full_Y_torch_tensor,
    best_l1_lambda,
    lbfgs_params=None,
    device="cpu",
):
    if lbfgs_params is None:
        lbfgs_params = {
            "lr": 0.005,
            "max_iter_per_step": 50,
            "epochs": 200,
            "tol_grad": 1e-9,
            "tol_change": 1e-11,
            "tol_loss_change": 1e-11,
        }

    print(f"\nTraining final model with best L1 lambda: {best_l1_lambda:.3e}")
    final_model_instance = model_class().to(device)
    if hasattr(final_model_instance, "double"):
        final_model_instance.double()

    X_data_device_dict = {
        key: full_X_torch_dict[key].to(device) for key in full_X_torch_dict
    }
    Y_data_device_tensor = full_Y_torch_tensor.to(device)

    trained_final_model = train_fold_lbfgs(
        model_instance=final_model_instance,
        train_X_dict=X_data_device_dict,
        train_Y_tensor=Y_data_device_tensor,
        l1_lambda=best_l1_lambda,
        lbfgs_params=lbfgs_params,
    )

    trained_final_model.eval()
    with torch.no_grad():
        predictions = trained_final_model(X_data_device_dict)
        final_loss_mse_part = mse_criterion(predictions, Y_data_device_tensor)

        l1_penalty_val = 0.0
        if best_l1_lambda > 0:
            for param in trained_final_model.parameters():
                l1_penalty_val += torch.abs(param.reshape(-1)).sum()
        total_final_loss = final_loss_mse_part + best_l1_lambda * l1_penalty_val
        final_r2 = R2_eval_metric(
            trained_final_model, X_data_device_dict, Y_data_device_tensor
        )

    print(f"Final model training finished.")
    print(f"  Parameters (a_params): {trained_final_model.a_params.data.cpu().numpy()}")
    print(f"  Parameters (b_params): {trained_final_model.b_params.data.cpu().numpy()}")
    print(f"  Final Model Total Loss (MSE_norm + L1): {total_final_loss.item():.6f}")
    print(f"  Final Model Normalized MSE part: {final_loss_mse_part.item():.6f}")
    print(f"  Final Model R2 on all data: {final_r2:.4f}")
    return trained_final_model
