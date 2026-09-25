"""
Temporal Evidence Neural Model (PyTorch GRU with Multi-Head Attention)
=======================================================================
Deep recurrent encoder for continuous trajectory representation, phase detection,
and latent temporal feature extraction from noisy scientific observations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.ode_fitting import MechanisticGrowthModels


class TemporalAttentionPooling(nn.Module):
    """
    Self-attention pooling mechanism that learns to focus on key dynamic inflection
    points (e.g. lag exit, maximum growth velocity, stationary transition).
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [batch_size, seq_len, hidden_dim]
        B, T, D = x.shape
        q = self.query_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B, heads, T, T]
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        # Average across heads for pooled context
        pooled_scores = attn_weights.mean(dim=1)  # [B, T, T]
        temporal_importance = pooled_scores.mean(dim=1)  # [B, T]
        
        weighted = torch.bmm(temporal_importance.unsqueeze(1), x).squeeze(1)  # [B, D]
        return weighted, temporal_importance


class TemporalEvidenceGRU(nn.Module):
    """
    Bidirectional GRU Network for Temporal Feature Extraction & Kinetic Evidence Encoding.
    Outputs latent trajectory embeddings, model classification logits, and phase timing estimates.
    """

    def __init__(
        self,
        input_dim: int = 4,  # [OD, time_norm, dOD_dt, d2OD_dt2]
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 4,
        latent_dim: int = 32,
        dropout: float = 0.15,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1
        self.gru_out_dim = hidden_dim * self.directions

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attention_pool = TemporalAttentionPooling(self.gru_out_dim, num_heads=4)
        
        # Latent representation projection
        self.latent_head = nn.Sequential(
            nn.Linear(self.gru_out_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )

        # Classification head for candidate model prior probabilities
        self.classifier_head = nn.Sequential(
            nn.Linear(self.gru_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        # Kinetic parameter prior estimator (approximate mu_max, lambda_lag, carrying_capacity)
        self.param_estimator = nn.Sequential(
            nn.Linear(self.gru_out_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # [mu_max_norm, lambda_lag_norm, A_norm]
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # x: [B, T, input_dim]
        h_proj = self.input_layer(x)
        gru_out, _ = self.gru(h_proj)
        
        pooled_feat, attn_weights = self.attention_pool(gru_out, mask)
        latent_embedding = self.latent_head(pooled_feat)
        class_logits = self.classifier_head(pooled_feat)
        param_estimates = self.param_estimator(pooled_feat)

        return {
            "latent_embedding": latent_embedding,
            "logits": class_logits,
            "class_probs": F.softmax(class_logits, dim=-1),
            "param_estimates": param_estimates,
            "attention_weights": attn_weights,
            "pooled_features": pooled_feat,
        }


class SyntheticTrajectoryGenerator:
    """
    Controlled generator for synthetic microbial growth curves with known ground truth
    parameters, heteroscedastic sensor noise, and variable sampling frequencies.
    """

    MODEL_NAMES = ["Gompertz", "Logistic", "Richards", "Baranyi"]

    @classmethod
    def generate_single_trajectory(
        cls,
        model_name: Optional[str] = None,
        duration: float = 48.0,
        num_points: int = 49,
        snr_db: float = 25.0,
        seed: Optional[int] = None,
    ) -> Dict[str, any]:
        rng = np.random.RandomState(seed)
        t = np.linspace(0.0, duration, num_points)

        if model_name is None:
            model_name = rng.choice(cls.MODEL_NAMES)

        # Randomize biologically realistic parameter sets
        y0 = rng.uniform(0.02, 0.12)
        A = rng.uniform(0.8, 2.5)
        mu_max = rng.uniform(0.15, 0.95)
        lambda_lag = rng.uniform(1.5, 12.0)
        nu = rng.uniform(0.3, 3.5) if model_name == "Richards" else 1.0

        if model_name == "Gompertz":
            y_clean = MechanisticGrowthModels.modified_gompertz(t, y0, A, mu_max, lambda_lag)
        elif model_name == "Logistic":
            y_clean = MechanisticGrowthModels.logistic(t, y0, A, mu_max, lambda_lag)
        elif model_name == "Richards":
            y_clean = MechanisticGrowthModels.richards(t, y0, A, mu_max, lambda_lag, nu)
        elif model_name == "Baranyi":
            y_clean = MechanisticGrowthModels.baranyi_roberts(t, y0, A, mu_max, lambda_lag)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Add Gaussian noise with heteroscedastic scale
        signal_power = np.mean(y_clean ** 2)
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        noise_std = math.sqrt(noise_power)
        # Heteroscedastic factor: higher noise at higher OD
        hetero_std = noise_std * (0.5 + 0.5 * (y_clean / (np.max(y_clean) + 1e-6)))
        noise = rng.normal(0.0, hetero_std, size=len(t))
        y_noisy = np.clip(y_clean + noise, 0.001, None)

        ground_truth_params = {
            "y0": float(y0),
            "A": float(A),
            "mu_max": float(mu_max),
            "lambda_lag": float(lambda_lag),
        }
        if model_name == "Richards":
            ground_truth_params["nu"] = float(nu)

        return {
            "time": t,
            "od600_noisy": y_noisy,
            "od600_clean": y_clean,
            "true_model": model_name,
            "true_model_idx": cls.MODEL_NAMES.index(model_name),
            "ground_truth_params": ground_truth_params,
            "snr_db": float(snr_db),
            "duration": float(duration),
        }

    @classmethod
    def generate_batch(
        cls,
        count: int = 500,
        snr_db: float = 25.0,
        seed: int = 42,
    ) -> List[Dict[str, any]]:
        trajectories = []
        for i in range(count):
            traj = cls.generate_single_trajectory(
                snr_db=snr_db,
                seed=seed + i,
            )
            trajectories.append(traj)
        return trajectories


class TrajectoryDataset(Dataset):
    """PyTorch Dataset wrapping synthesized or empirical time-series trajectories."""

    def __init__(self, data_list: List[Dict[str, any]]):
        self.data_list = data_list

    def __len__(self) -> int:
        return len(self.data_list)

    @staticmethod
    def extract_features(t: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Constructs 4-dimensional temporal tensor [OD, time_norm, dy_dt, d2y_dt2]."""
        t_norm = t / (np.max(t) + 1e-6)
        dy = np.gradient(y, t)
        d2y = np.gradient(dy, t)
        features = np.stack([y, t_norm, dy, d2y], axis=-1)
        return features.astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data_list[idx]
        t = item["time"]
        y = item["od600_noisy"]
        features = self.extract_features(t, y)
        label = item["true_model_idx"]

        return {
            "features": torch.tensor(features, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
        }


class TemporalEvidenceEncoder:
    """Convenience wrapper for inference and neural feature extraction."""

    def __init__(self, model: Optional[TemporalEvidenceGRU] = None, device: str = "cpu"):
        self.device = device
        if model is None:
            self.model = TemporalEvidenceGRU().to(device)
            self.model.eval()
        else:
            self.model = model.to(device)

    def extract_evidence(self, t: np.ndarray, y: np.ndarray) -> Dict[str, any]:
        """Runs the neural encoder on a single trajectory and returns structured evidence."""
        features = TrajectoryDataset.extract_features(t, y)
        feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(feat_tensor)
            probs = output["class_probs"].squeeze(0).cpu().numpy()
            embedding = output["latent_embedding"].squeeze(0).cpu().numpy()
            attn = output["attention_weights"].squeeze(0).cpu().numpy()

        class_names = SyntheticTrajectoryGenerator.MODEL_NAMES
        prob_dict = {name: float(probs[i]) for i, name in enumerate(class_names)}
        predicted_class = class_names[int(np.argmax(probs))]
        confidence = float(np.max(probs))

        return {
            "predicted_model_prior": predicted_class,
            "confidence": confidence,
            "class_probabilities": prob_dict,
            "latent_vector": [float(x) for x in embedding[:8]],  # Truncated summary
            "peak_attention_time_hours": float(t[int(np.argmax(attn))]),
        }
