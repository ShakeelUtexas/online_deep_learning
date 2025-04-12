import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from homework.models import load_model, save_model
from homework.datasets.road_dataset import load_data
from homework.metrics import PlannerMetric
from homework import train_mlp,train_tp,train_cnn
"""
Usage:
    python3 -m homework.train_planner --your_args here
"""
 

def train(
    model_name: str,
    transform_pipeline: str,
    num_workers: int,
    lr: float,
    batch_size: int,
    num_epoch: int,
    device: str = None,
):
    """
    Train a model with the given parameters.

    Args:
        model_name (str): Name of the model to train (e.g., "linear_planner").
        transform_pipeline (str): Data transformation pipeline to use.
        num_workers (int): Number of workers for the DataLoader.
        lr (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.
        num_epoch (int): Number of epochs to train.
        device (str): Device to use for training ("cuda", "mps", or "cpu").
    """
    #  # Automatically determine the best available device if not provided
    # if device is None:
    #     if torch.cuda.is_available():
    #         device = "cuda"
    #     elif torch.backends.mps.is_available():
    #         device = "mps"
    #     else:
    #         device = "cpu"
    # print(f"Using device: {device}")
    # Load training and validation datasets
    train_loader = load_data(
        dataset_path="drive_data/train",
        transform_pipeline=transform_pipeline,
        return_dataloader=True,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = load_data(
        dataset_path="drive_data/val",
        transform_pipeline=transform_pipeline,
        return_dataloader=True,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
    )
    # Load the model



    if model_name == "mlp_planner":
        train_mlp.train_model(
                model_name=model_name,
                train_loader=train_loader,
                val_loader=val_loader,
                n_epochs=num_epoch,
                learning_rate=lr,
                device=device,
            )
    elif model_name == "transformer_planner":
        train_tp.train_model(
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=num_epoch,
            learning_rate=lr,
            device=device,
        )
    elif model_name == "cnn_planner":
        train_cnn.train_model(
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=num_epoch,
            learning_rate=lr,
            device=device,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

if __name__ == "__main__":
 
    for lr in [1e-2, 1e-3, 1e-4]:
        train(
            model_name="transformer_planner", # 
            #transform_pipeline="state_only", for mlp
            transform_pipeline="state_only", # for cnn
            num_workers=4,
            lr=lr,
            batch_size=128,
            num_epoch=40,
         )


# #working for cnn and mlp
# # this is just how it's structured in the solution
# for lr in [1e-3]:#, 1e-3, 1e-4]:
#     train(
#         model_name="cnn_planner",
#         transform_pipeline="default",
#         num_workers=4,
#         lr=lr,
#         batch_size=128,
#         num_epoch=40,
#     )