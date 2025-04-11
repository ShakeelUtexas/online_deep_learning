"""
Usage:
    python3 -m homework.train_planner --your_args here
"""
import torch
from torch.utils.data import DataLoader
from homework.models import load_model, save_model
from homework.datasets.road_dataset import load_data
from homework.metrics import PlannerMetric


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
    # Automatically determine the best available device if not provided
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Using device: {device}")

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
    model = load_model(model_name, with_weights=False).to(device)

    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epoch):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            # Move data to device
            inputs = batch["image"].to(device)
            targets = batch["waypoints"].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        longitudinal_error = 0.0
        lateral_error = 0.0
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                inputs = batch["image"].to(device)
                targets = batch["waypoints"].to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                # Calculate metrics
                metric = PlannerMetric()
                metric.add(outputs, targets, labels_mask=torch.ones_like(targets[..., 0]))
                metrics = metric.compute()
                longitudinal_error += metrics["longitudinal_error"]
                lateral_error += metrics["lateral_error"]

        val_loss /= len(val_loader)
        longitudinal_error /= len(val_loader)
        lateral_error /= len(val_loader)

        # Logging
        print(
            f"Epoch {epoch + 1}/{num_epoch} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Longitudinal Error: {longitudinal_error:.4f} | "
            f"Lateral Error: {lateral_error:.4f}"
        )

    # Save the trained model
    save_path = save_model(model)
    print(f"Model saved to {save_path}")