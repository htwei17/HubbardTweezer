import torch
import torch.nn as nn


# Helper function for the MSE loss criterion used in LBFGS optimization
def mse_criterion(outputs, targets):
    mse_loss_fn = nn.MSELoss()  # Averages over all elements
    # targets might be (B, P, 1). torch.var computes variance over all elements.
    var_targets_val = torch.var(targets)
    loss = mse_loss_fn(outputs, targets) / (
        var_targets_val.item() if var_targets_val.item() != 0 else 1.0
    )
    return loss


# Helper function for R2 Score Evaluation
def R2_eval_metric(model: nn.Module, X_data_dict, Y_data_tensor):
    model.eval()
    with torch.no_grad():
        predictions = model(X_data_dict)

    # Flatten to compute metrics over all individual data points (N*P)
    Y_data_flat = Y_data_tensor.reshape(-1)
    predictions_flat = predictions.reshape(-1)

    sst = torch.var(Y_data_flat)
    if sst.item() == 0:
        return -float("inf")

    ssr = torch.mean((Y_data_flat - predictions_flat) ** 2)
    r2 = 1 - (ssr / sst)
    return r2.item()
