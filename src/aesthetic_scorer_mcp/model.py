import torch
import torch.nn as nn


class AestheticScorer(nn.Module):
    """
    Fine-tuned CLIP model to predict aesthetic scores (e.g., light, depth, composition) based on the PARA dataset.
    """
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

        # Define the scoring heads
        hidden_dim = backbone.config.hidden_size
        self.aesthetic_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

        self.composition_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

        self.light_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

        self.color_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

        self.dof_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

        self.content_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pixel_values):
        features = self.backbone(pixel_values).pooler_output
        return (
            self.aesthetic_head(features),
            self.quality_head(features),
            self.composition_head(features),
            self.light_head(features),
            self.color_head(features),
            self.dof_head(features),
            self.content_head(features)
        )
