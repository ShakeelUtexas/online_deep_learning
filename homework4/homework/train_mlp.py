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

    # Load the model
    model = load_model(model_name, with_weights=False).to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error for waypoint prediction
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch in train_loader:
           # Move data to device
            track_left = batch["track_left"].to(device)  # Extract track_left from batch
            track_right = batch["track_right"].to(device)  # Extract track_right from batch
            targets = batch["waypoints"].to(device)  # Extract waypoints (targets) from batch

            # Forward pass
            #outputs = model(track_left=track_left, track_right=track_right)  # Pass both track_left and track_right


            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(track_left=track_left, track_right=track_right)  # Pass both track_left and track_right
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
                track_left = batch["track_left"].to(device)  # Extract track_left from batch
                track_right = batch["track_right"].to(device)  # Extract track_right from batch
                targets = batch["waypoints"].to(device)  # Extract waypoints (targets) from batch

                # Forward pass
                outputs = model(track_left=track_left, track_right=track_right)  # Pass both track_left and track_right
            
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


if __name__ == "__main__":
    # Load training and validation datasets
    train_loader = load_data(
        dataset_path="drive_data/train",  # Path to the training data
        transform_pipeline="default",
        return_dataloader=True,
        num_workers=4,
        batch_size=32,
        shuffle=True,
    )

    val_loader = load_data(
        dataset_path="drive_data/val",  # Path to the validation data
        transform_pipeline="default",
        return_dataloader=True,
        num_workers=4,
        batch_size=32,
        shuffle=False,
    )

    # Train the model
    train_model(
        model_name="mlp_planner",    # Change to "MLPPlanner" or "TransformerPlanner" as needed
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=50,
        learning_rate=1e-3,
    )