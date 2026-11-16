
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureTokenizer(nn.Module):
    def __init__(self, num_features, d_model):
        super().__init__()
        self.weights = nn.Parameter(
            torch.randn(num_features, d_model) * 0.02
        )
        self.biases = nn.Parameter(
            torch.zeros(num_features, d_model)
        )
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, d_model)
        )

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.weights.unsqueeze(0) \
                 + self.biases.unsqueeze(0)

        cls = self.cls_token.expand(x.size(0), -1, -1)

        return torch.cat([cls, tokens], dim=1)


class DualHeadRiskTransformer(nn.Module):
    def __init__(
        self,
        num_features,
        num_classes=4,
        d_model=64,
        nhead=4,
        num_layers=3,
        dropout=0.25
    ):
        super().__init__()

        self.tokenizer = FeatureTokenizer(num_features, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(d_model)

        self.classifier_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

        self.anomaly_decoder = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, num_features)
        )

        self.register_buffer(
            "normal_center",
            torch.zeros(d_model)
        )

    def forward(self, x):
        encoded = self.encoder(self.tokenizer(x))

        cls_repr = self.norm(encoded[:, 0, :])

        logits = self.classifier_head(cls_repr)
        reconstructed = self.anomaly_decoder(cls_repr)

        return logits, reconstructed, cls_repr

    def compute_anomaly_score(self, x):
        logits, reconstructed, cls_repr = self.forward(x)

        recon_err = F.mse_loss(
            reconstructed, x, reduction="none"
        ).mean(dim=-1)

        hypersphere_dist = torch.norm(
            cls_repr - self.normal_center,
            p=2,
            dim=-1
        )

        energy = -torch.logsumexp(logits, dim=-1)

        anomaly_score = (
            0.5 * recon_err
            + 0.3 * hypersphere_dist
            + 0.2 * energy
        )

        return logits, anomaly_score


def load_model(checkpoint_path, device):
    model = DualHeadRiskTransformer(
        num_features=122,
        num_classes=4,
        d_model=64,
        nhead=4,
        num_layers=3,
        dropout=0.25
    )

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model
