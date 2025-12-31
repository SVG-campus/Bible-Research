# ============================================================
# UNIFIED INTERVENTION FRAMEWORK (UIF) - PLATINUM EDITION
# ============================================================

import warnings
import numpy as np
import pandas as pd
import networkx as nx
import xgboost as xgb
import torch
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

# GPU Configuration
warnings.filterwarnings("ignore")
USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"
print(f"✅ UIF PLATINUM ONLINE | Device: {DEVICE}")

class PlatinumCausalEngine:
    def __init__(self, causal_graph, data: pd.DataFrame, verbose=True):
        self.G = self._build_graph(causal_graph)
        self.df = data.copy()
        self.verbose = verbose
        self.nodes = list(nx.topological_sort(self.G))
        self.models = {}
        self.model_types = {}
        self.residuals = pd.DataFrame(index=self.df.index)

        if self.verbose: print("⚙️ FITTING ADAPTIVE MODELS...")
        self._fit_adaptive_models()
        self._compute_residuals()
        if self.verbose: print(" -> Models Fitted & Residuals Computed.")

    def _build_graph(self, g_input):
        if isinstance(g_input, list):
            G = nx.DiGraph()
            G.add_edges_from(g_input)
            return G
        return g_input.copy()

    def _fit_adaptive_models(self):
        for node in self.nodes:
            parents = list(self.G.predecessors(node))
            if not parents:
                self.residuals[node] = self.df[node] 
                continue

            X = self.df[parents]
            y = self.df[node]

            lin_scores, xgb_scores = [], []
            kf = KFold(n_splits=3, shuffle=True, random_state=42)

            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                lin = LinearRegression().fit(X_train, y_train)
                lin_scores.append(lin.score(X_val, y_val))

                xg = xgb.XGBRegressor(n_estimators=50, max_depth=4, device=DEVICE, enable_categorical=True)
                xg.fit(X_train, y_train)
                xgb_scores.append(xg.score(X_val, y_val))

            avg_lin, avg_xgb = np.mean(lin_scores), np.mean(xgb_scores)

            if avg_xgb > avg_lin + 0.05:
                self.models[node] = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, device=DEVICE)
                self.models[node].fit(X, y)
                self.model_types[node] = "XGBoost (Non-Linear)"
            else:
                self.models[node] = LinearRegression()
                self.models[node].fit(X, y)
                self.model_types[node] = "Linear (Robust)"

            if self.verbose:
                print(f" - {node}: Selected {self.model_types[node]} (R2: {max(avg_lin, avg_xgb):.3f})")

    def _compute_residuals(self):
        for node in self.nodes:
            parents = list(self.G.predecessors(node))
            if not parents: continue
            X = self.df[parents]
            pred = self.models[node].predict(X)
            self.residuals[node] = self.df[node] - pred

    def simulate_intervention(self, treatment: dict, target: str, n_boot=0) -> dict:
        mu = self._run_simulation_pass(self.df, self.residuals, treatment, target)

        result = {
            "E_y_do": mu,
            "std_error": 0.0,
            "ci_lower": mu, "ci_upper": mu,
            "model_used": self.model_types.get(target, "Direct"),
            "treatment": treatment
        }

        if n_boot > 0:
            estimates = []
            for _ in range(n_boot):
                res_boot = self.residuals.sample(frac=1.0, replace=True)
                df_boot = self.df.loc[res_boot.index].copy()
                val = self._run_simulation_pass(df_boot, res_boot, treatment, target)
                estimates.append(val)

            result["ci_lower"] = np.percentile(estimates, 2.5)
            result["ci_upper"] = np.percentile(estimates, 97.5)
            result["std_error"] = np.std(estimates)

        return result

    def _run_simulation_pass(self, df_base, res_base, treatment, target):
        df_sim = df_base.copy()
        for t_var, t_val in treatment.items():
            df_sim[t_var] = t_val
        for node in self.nodes:
            if node in treatment: continue
            parents = list(self.G.predecessors(node))
            if not parents: continue
            X = df_sim[parents]
            base_val = self.models[node].predict(X)
            df_sim[node] = base_val + res_base[node].values
        return df_sim[target].mean()
