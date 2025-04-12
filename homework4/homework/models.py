from pathlib import Path

import torch
import torch.nn as nn
import math

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        # Define a simple MLP
        self.mlp = nn.Sequential(
            nn.Flatten(),  # Flatten the input
            nn.Linear(n_track * 2 * 2, 128),  # Input: (track_left + track_right) flattened
            nn.ReLU(),
            nn.Linear(128, n_waypoints * 2),  # Output: (n_waypoints, 2)
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # Concatenate left and right track boundaries along the last dimension
        x = torch.cat([track_left, track_right], dim=-1)  # Shape: (b, n_track, 4)

        # Pass through the MLP
        x = self.mlp(x)

        # Reshape to (batch_size, n_waypoints, 2)
        return x.view(x.size(0), self.n_waypoints, 2)

 

class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,  # Changed default to match dataset
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        # Input embedding for track points
        self.input_embedding = nn.Linear(2, d_model)
        
        # Query embeddings for waypoints
        self.query_embed = nn.Embedding(n_waypoints, d_model)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # Output layer
        self.output_layer = nn.Linear(d_model, 2)

    def forward(self, track_left: torch.Tensor, track_right: torch.Tensor) -> torch.Tensor:
        batch_size = track_left.size(0)
        
        # Concatenate track boundaries
        track_points = torch.cat([track_left, track_right], dim=1)
        
        # Embed track points
        track_embeddings = self.input_embedding(track_points)
        
        # Transpose for Transformer input (seq_len, batch, d_model)
        memory = track_embeddings.transpose(0, 1)
        
        # Get waypoint queries and expand for batch
        queries = self.query_embed.weight.unsqueeze(1).expand(-1, batch_size, -1)
        
        # Decoder forward pass
        waypoint_features = self.transformer_decoder(
            tgt=queries,
            memory=memory
        )
        
        # Transpose back and predict waypoints
        waypoints = self.output_layer(waypoint_features.transpose(0, 1))
        
        return waypoints

    def _add_positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add sinusoidal positional encoding to the input embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape (B, seq_len, d_model).

        Returns:
            torch.Tensor: Tensor with positional encoding added.
        """
        seq_len, d_model = x.size(1), x.size(2)
        position = torch.arange(seq_len, device=x.device).unsqueeze(1)  # Shape: (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=x.device) * -(math.log(10000.0) / d_model))  # Shape: (d_model // 2)

        pe = torch.zeros(1, seq_len, d_model, device=x.device)
        pe[0, :, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[0, :, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices

        return x + pe

    def _add_positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add sinusoidal positional encoding to the input embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape (B, seq_len, d_model).

        Returns:
            torch.Tensor: Tensor with positional encoding added.
        """
        seq_len, d_model = x.size(1), x.size(2)
        position = torch.arange(seq_len, device=x.device).unsqueeze(1)  # Shape: (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=x.device) * -(math.log(10000.0) / d_model))  # Fixed missing parenthesis

        pe = torch.zeros(1, seq_len, d_model, device=x.device)
        pe[0, :, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[0, :, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices

        return x + pe

    def predict(self, track_left: torch.Tensor, track_right: torch.Tensor) -> torch.Tensor:
        """Convenience method for inference with optional preprocessing"""
        with torch.no_grad():
            return self.forward(track_left, track_right)

class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        # Define CNN layers
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((4, 4)),  # Reduce to a fixed size
        )

        # Fully connected layers to predict waypoints
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 4 * 4, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, n_waypoints * 2),  # Output (n_waypoints, 2)
        )


    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        # Normalize the input image
        x = (image - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Pass through convolutional layers
        x = self.conv_layers(x)

        # Pass through fully connected layers
        x = self.fc_layers(x)

        # Reshape to (batch_size, n_waypoints, 2)
        return x.view(x.size(0), self.n_waypoints, 2)


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    n_track: int = 10, # number of track points for Transormer
    n_waypoints: int = 3,   # number of waypoints to predict for Transormer
    **model_kwargs
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """

    if model_name == "transformer_planner":
        model = TransformerPlanner(
            n_track=n_track,
            n_waypoints=n_waypoints,
            d_model=128,
            nhead=8,
            num_layers=4
        ) 
    else:
        model = MODEL_FACTORY[model_name](**model_kwargs)

        if with_weights:
            model_path = HOMEWORK_DIR / f"{model_name}.th"
            assert model_path.exists(), f"{model_path.name} not found"

            try:
                m.load_state_dict(torch.load(model_path, map_location="cpu"))
            except RuntimeError as e:
                raise AssertionError(
                    f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
                ) from e

        # limit model sizes since they will be zipped and submitted
        model_size_mb = calculate_model_size_mb(m)

        if model_size_mb > 20:
            raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return model


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
