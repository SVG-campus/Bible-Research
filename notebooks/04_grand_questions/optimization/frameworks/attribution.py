# FIXED ATTRIBUTION FRAMEWORK
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings('ignore')
torch.manual_seed(2025)
np.random.seed(2025)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class UniversalAttributionValidator:
    def __init__(self, X_raw=None, y_raw=None, feature_names=None):
        print(f"🚀 INITIALIZING TRUTH ENGINE on {DEVICE}...")
        self.model = None
        self.X_tensor = None
        self.y_tensor = None
        if X_raw is not None and y_raw is not None:
            self.fit(X_raw, y_raw, feature_names)

    def fit(self, X_raw, y_raw, feature_names=None):
        self.X_raw = X_raw
        self.y_raw = y_raw
        self.features = feature_names if feature_names else list(X_raw.columns)
        self.scaler = StandardScaler()
        self.X_tensor = torch.FloatTensor(self.scaler.fit_transform(self.X_raw)).to(DEVICE)
        y_values = self.y_raw.values
        if len(y_values.shape) == 1: y_values = y_values.reshape(-1, 1)
        self.y_tensor = torch.FloatTensor(y_values).to(DEVICE)
        self.n_feat = self.X_tensor.shape[1]
        print(f" >> [DATA] Loaded {len(self.X_raw)} samples with {self.n_feat} features.")
        self._train_robust_proxy()

    def _train_robust_proxy(self):
        print(" >> [MODEL] Training Robust Neural Proxy...")
        self.model = nn.Sequential(
            nn.Linear(self.n_feat, 48), nn.ReLU(), nn.Dropout(0.1), 
            nn.Linear(48, 24), nn.ReLU(), nn.Linear(24, 1)
        ).to(DEVICE)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005, weight_decay=1e-3)
        loss_fn = nn.MSELoss()
        for epoch in range(500):
            optimizer.zero_grad()
            noise = torch.randn_like(self.X_tensor) * 0.05
            y_pred = self.model(self.X_tensor + noise)
            loss = loss_fn(y_pred, self.y_tensor)
            loss.backward()
            optimizer.step()
        r2 = 1 - (loss.item() / (torch.var(self.y_tensor).item() + 1e-9))
        print(f" ✓ Model Converged (R²: {r2:.3f}) - Stability Locked.")

    def compute_attribution(self, steps=100):
        if self.model is None: raise ValueError("Model not trained. Call .fit() first.")
        print("
>> [1/4] Computing Integrated Gradients...")
        baseline = torch.zeros_like(self.X_tensor)
        attributions = []
        batch_size = 2000
        for i in range(0, len(self.X_tensor), batch_size):
            bs = min(batch_size, len(self.X_tensor) - i)
            batch_X = self.X_tensor[i:i+bs]
            batch_base = baseline[i:i+bs]
            alphas = torch.linspace(0, 1, steps).to(DEVICE)
            path = batch_base.unsqueeze(0) + alphas.view(-1, 1, 1) * (batch_X - batch_base).unsqueeze(0)
            path.requires_grad = True
            preds = self.model(path.reshape(-1, self.n_feat))
            grads = torch.autograd.grad(torch.sum(preds), path)[0]
            attr = (batch_X - batch_base) * torch.mean(grads, dim=0)
            attributions.append(attr.detach().cpu().numpy())
        self.ig_scores = pd.DataFrame(np.vstack(attributions), columns=self.features)
        return self.ig_scores.mean().sort_values(ascending=False)

    def compute_interactions(self, top_n=5):
        if self.model is None: return
        print("
>> [2/4] Scanning for Overlaps...")
        idx = np.random.choice(len(self.X_tensor), min(500, len(self.X_tensor)), replace=False)
        X_s = self.X_tensor[idx]
        base = torch.mean(self.X_tensor, dim=0)
        interactions = {}
        corr = self.X_raw.corr().abs()
        pairs = [(c, r) for c in corr.columns for r in corr.columns if c < r and corr.loc[c,r] > 0.3]
        for name_a, name_b in pairs:
            idx_a, idx_b = self.features.index(name_a), self.features.index(name_b)
            X_00, X_11 = X_s.clone(), X_s.clone()
            X_10, X_01 = X_s.clone(), X_s.clone()
            X_00[:, [idx_a, idx_b]] = base[[idx_a, idx_b]]
            X_10[:, idx_b] = base[idx_b]
            X_01[:, idx_a] = base[idx_a]
            with torch.no_grad():
                val = (self.model(X_11) - self.model(X_10) - self.model(X_01) + self.model(X_00))
                interactions[f"{name_a} + {name_b}"] = val.mean().item()
        self.inter_scores = pd.Series(interactions).sort_values(key=abs, ascending=False).head(top_n)

    def detect_regimes(self):
        if self.model is None: return
        print("
>> [3/4] Scanning for Regime Changes...")
        attrs = self.ig_scores.values
        if len(attrs) < 10: return
        kmeans = KMeans(n_clusters=2, random_state=42).fit(attrs)
        score = silhouette_score(attrs, kmeans.labels_)
        self.regime_status = "STABLE" if score < 0.5 else "MULTI-REGIME"
        print(f" ✓ Clustering Score: {score:.3f} ({self.regime_status})")

    def generate_certificate(self):
        if self.model is None: return
        print("
>> [4/4] Generating Validation Certificate...")
        noise = torch.randn_like(self.X_tensor) * 0.1
        with torch.no_grad():
            p1 = self.model(self.X_tensor).flatten()
            p2 = self.model(self.X_tensor + noise).flatten()
            stability = torch.corrcoef(torch.stack([p1, p2]))[0,1].item()
        delta_y = (self.model(self.X_tensor) - self.model(torch.zeros_like(self.X_tensor))).mean().item()
        sum_attr = self.ig_scores.sum(axis=1).mean()
        completeness = abs(delta_y - sum_attr)
        passed = stability > 0.90 and completeness < 0.05
        status_icon = "✅ VERIFIED" if passed else "❌ FAILED"
        print(f"STATUS: {status_icon}")
        print(f"1. Robustness Score: {stability:.4f} (Target > 0.90)")
        print(f"2. Completeness Err: {completeness:.5f} (Target < 0.05)")
        print(f"3. Regime Stability: {self.regime_status}")
        print("
ATTRIBUTION DRIVERS:")
        print(self.ig_scores.mean().sort_values(ascending=False).to_string())
