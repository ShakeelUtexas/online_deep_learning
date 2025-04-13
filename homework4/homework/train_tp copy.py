import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from homework.models import load_model, save_model
from homework.datasets.road_dataset import load_data
from homework.metrics import PlannerMetric

def train_model(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 20,
    learning_rate: float = 1e-3,
    n_track: int = 10,  # Number of track points
    n_waypoints: int = 3,  # Number of waypoints
    device: str = None,
):
    """
    Train a model and evaluate it on validation data.

    Args:
        model_name (str): Name of the model to train (e.g., "mlp_planner", "transformer_planner", "cnn_planner").
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        n_epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate for the optimizer.
        n_track (int): Number of track points.
        n_waypoints (int): Number of waypoints to predict.
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

    # Load the model with the correct configuration
    model = load_model(model_name, n_track=n_track, n_waypoints=n_waypoints).to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error for waypoint prediction
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )

    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            #print(f"Targets shape: {batch['waypoints'].shape}")
            # Move data to device
            track_left = batch["track_left"].to(device)
            track_right = batch["track_right"].to(device)
            targets = batch["waypoints"].to(device)

            # Debug shapes
            #print(f"Outputs shape: {model(track_left, track_right).shape}")
            #print(f"Targets shape: {targets.shape}")

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(track_left=track_left, track_right=track_right)
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
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                targets = batch["waypoints"].to(device)

                # Forward pass
                outputs = model(track_left=track_left, track_right=track_right)
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
            f"Epoch {epoch + 1}/{n_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Longitudinal Error: {longitudinal_error:.4f} | "
            f"Lateral Error: {lateral_error:.4f}"
        )

    # Save the trained model
    save_path = save_model(model)
    print(f"Model saved to {save_path}")