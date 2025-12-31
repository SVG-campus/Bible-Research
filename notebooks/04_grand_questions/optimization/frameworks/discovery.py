# ==============================================================================
# THE UNIFIED DISCOVERY FRAMEWORK (PRODUCTION EDITION)
# ==============================================================================
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import ruptures as rpt  # pip install ruptures
import warnings

warnings.filterwarnings('ignore')

class UnifiedDiscoveryEngine:
    def __init__(self, df, target_col, date_col='Date_Index'):
        self.raw_df = df.copy()
        self.target = target_col
        self.date_col = date_col
        self.known_features = [c for c in df.columns if c not in [target_col, date_col]]
        print(f"🚀 ENGINE ONLINE. Target: '{self.target}'")
        print(f"   Initial Knowledge Base: {self.known_features}")

    # PHASE 1: DISCOVERY & PHYSICS SCAN
    def scan_environment(self):
        print("\n>> [PHASE 1] SCANNING ENVIRONMENT (Structure & Time)...")
        X = self.raw_df[self.known_features]
        y = self.raw_df[self.target]

        # Baseline Model (XGBoost for non-linearity)
        self.model = xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
        self.model.fit(X, y)
        baseline_r2 = self.model.score(X, y)
        print(f"   Baseline Model R²: {baseline_r2:.4f}")

        # Residual Analysis
        preds = self.model.predict(X)
        residuals = y - preds

        # Temporal Scan
        max_corr = 0
        best_lag = 0
        for l in [1, 2, 3, 7, 30]: 
            res_shift = np.roll(residuals, l)
            if len(residuals) > l:
                corr = pearsonr(residuals[l:], res_shift[l:])[0]
                if abs(corr) > max_corr:
                    max_corr = abs(corr)
                    best_lag = l

        if max_corr > 0.15:
            print(f"   ⚠️  TEMPORAL GAP: Lag detected (Corr: {max_corr:.2f}). Suggesting Lag-{best_lag}.")
            return f"Sales_Lag_{best_lag}"
        return None

    def add_feature(self, name, data):
        self.raw_df[name] = data
        if name not in self.known_features:
            self.known_features.append(name)
        print(f"   ➕ ADDED: '{name}'")

    # PHASE 2: CAUSAL DOMINANCE
    def verify_causality_nonlinear(self):
        print("\n>> [PHASE 2] CAUSAL DOMINANCE (Non-Linear Proxy Kill)...")
        anchors = [f for f in self.known_features if "Lag" in f or "Price" in f or "Competitor" in f]
        suspects = [f for f in self.known_features if f not in anchors]

        if not anchors: 
            print("   (Skipping: No Anchors found to test against).")
            return

        survivors = list(anchors)
        for suspect in suspects:
            X_anchor = self.raw_df[anchors]
            y_suspect = self.raw_df[[suspect]]

            m_check = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
            m_check.fit(X_anchor, y_suspect.values.ravel())
            pred_suspect = m_check.predict(X_anchor)

            explained_variance = r2_score(y_suspect, pred_suspect)
            print(f"   ⚔️  AUDITING '{suspect}'...")
            print(f"       -> Redundancy Score (RF-R²): {explained_variance:.4f}")

            if explained_variance > 0.90:
                resid_suspect = y_suspect.values.ravel() - pred_suspect
                m_sales = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
                m_sales.fit(X_anchor, self.raw_df[self.target])
                resid_sales = self.raw_df[self.target] - m_sales.predict(X_anchor)

                corr_resid = pearsonr(resid_sales, resid_suspect)[0]
                print(f"       -> Independent Signal (Resid Corr): {corr_resid:.4f}")

                if abs(corr_resid) < 0.15:
                    print(f"       ❌ REJECTED: Pure Proxy. Removing.")
                    if suspect in self.known_features:
                        self.known_features.remove(suspect)
                else:
                    print(f"       ⚠️  KEPT: High overlap, but unique signal exists.")
                    survivors.append(suspect)
            else:
                print("       ✅ VERIFIED: Unique Driver.")
                survivors.append(suspect)

        self.known_features = list(set(self.known_features) & set(survivors + anchors))

    # PHASE 3: REGIME CHANGE DETECTION
    def detect_regimes_and_drift(self):
        print("\n>> [PHASE 3] REGIME CHANGE DETECTION (Ruptures)...")
        X = self.raw_df[self.known_features]
        y = self.raw_df[self.target]

        try:
            global_model = Ridge().fit(X, y)
            residuals = (y - global_model.predict(X)).values.reshape(-1, 1)
            algo = rpt.Pelt(model="rbf").fit(residuals)
            result = algo.predict(pen=10)
        except:
            print("   (Ruptures scan failed or no library, skipping)")
            result = []

        if len(result) > 1:
            last_cp = result[-2]
            print(f"   📍 Detected Regime Change at Day {last_cp}")

            df_past = self.raw_df.iloc[:last_cp]
            df_recent = self.raw_df.iloc[last_cp:]

            if len(df_recent) < 20: 
                print("   (Recent regime too short for stability analysis)")
                return

            m_past = LinearRegression().fit(df_past[self.known_features], df_past[self.target])
            m_recent = LinearRegression().fit(df_recent[self.known_features], df_recent[self.target])

            print(f"   {'Feature':<15} | {'Past':<8} | {'Recent':<8} | {'Status'}")
            print("-" * 60)

            for i, feat in enumerate(self.known_features):
                past = m_past.coef_[i]
                recent = m_recent.coef_[i]

                mag_past = abs(past)
                mag_recent = abs(recent)

                if mag_past < 0.001: ratio = 1.0
                else: ratio = mag_recent / mag_past

                if ratio < 0.2: status = "💀 ZOMBIE"
                elif ratio < 0.5: status = "⚠️  DECAYING"
                elif ratio > 1.5: status = "🔥 SURGING"
                else: status = "✅ STABLE"

                print(f"   {feat:<15} | {past:<8.2f} | {recent:<8.2f} | {status}")
        else:
            print("   ✅ No Regime Change Detected (Stable World).")

    # PHASE 4: IV DISCOVERY
    def scan_for_instruments(self, potential_instruments):
        print("\n>> [PHASE 4] IV DISCOVERY (Endogeneity Scan)...")
        exog_controls = [f for f in self.known_features if "Lag" in f]
        if not exog_controls:
            print("   (No Lags found. Cannot perform Reduced Form check.)")
            return

        X_reduced = self.raw_df[exog_controls]
        y = self.raw_df[self.target]
        m_red = LinearRegression().fit(X_reduced, y)
        resid_reduced = y - m_red.predict(X_reduced)

        if 'My_Price' not in self.known_features: return
        price = self.raw_df['My_Price']

        print(f"   {'Candidate':<15} | {'Corr(Price)':<12} | {'Corr(Resid)':<12} | {'Verdict'}")
        print("-" * 65)

        for name, data in potential_instruments.items():
            corr_price = pearsonr(data, price)[0]
            corr_resid = pearsonr(data, resid_reduced)[0]

            status = "❌"
            if abs(corr_price) > 0.3:
                if abs(corr_resid) < abs(corr_price): 
                    status = "✅ VALID IV"

            print(f"   {name:<15} | {corr_price:<12.2f} | {corr_resid:<12.2f} | {status}")
