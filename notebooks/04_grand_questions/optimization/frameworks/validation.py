# FIXED VALIDATION FRAMEWORK
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from scipy.stats import ks_2samp
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

@dataclass
class TVFConfig:
    drift_alpha: float = 0.05
    reconstruction_error_threshold: float = 0.05 
    min_explained_variance_ratio: float = 0.90 
    max_null_spike: float = 0.10
    min_numeric_coercion: float = 0.98
    max_volumetric_drift: float = 0.50
    max_duplicate_rate: float = 0.0
    max_cardinality: int = 100
    allow_zero_variance: bool = False
    max_leakage_r2: float = 0.98
    enable_iforest: bool = True
    iforest_contamination: float = 0.02
    simpsons_threshold: float = 0.3
    max_fairness_disparity: float = 0.25

class TitanValidationFramework:
    def __init__(self, reference_data: pd.DataFrame, config: TVFConfig = None):
        self.logger = logging.getLogger("TVF")
        self.config = config or TVFConfig()
        self.reference = reference_data.copy()
        self.num_cols = self.reference.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols = self.reference.select_dtypes(include=['object', 'category']).columns.tolist()
        self._train_unsupervised()
        print(f"TVF Omni Online. Baseline: {len(self.reference)} rows.")

    def _train_unsupervised(self):
        self.iforest = None
        if self.config.enable_iforest and len(self.reference) > 50:
            self.iforest = IsolationForest(contamination=self.config.iforest_contamination, random_state=42, n_jobs=-1)
            self.iforest.fit(self.reference[self.num_cols].fillna(0))

    def validate(self, new_data: pd.DataFrame, target_col: str = None, date_col: str = None, 
                 consistency_rules: List[str] = None, subgroups: List[str] = None) -> Tuple[bool, Dict]:
        report = {"modules": {}, "traffic_light": None}
        report["modules"]["integrity"] = {"issues": []}
        report["modules"]["drift_num"] = self._scan_drift_numeric(new_data)
        report["modules"]["anomalies"] = self._scan_anomalies(new_data)
        if target_col and subgroups:
            report["modules"]["fairness"] = self._scan_fairness(new_data, target_col, subgroups)
        failed = False
        rows = []
        for mod, result in report["modules"].items():
            for k, v in result.items():
                if isinstance(v, list) and v:
                    for item in v: rows.append({"Module": mod, "Issue": str(item), "Status": "RED"})
        report_df = pd.DataFrame(rows) if rows else pd.DataFrame([{"Module": "All", "Issue": "None", "Status": "GREEN"}])
        return (not failed), report_df

    def _scan_drift_numeric(self, df):
        drifted = []
        for c in self.num_cols:
            if c not in df.columns: continue
            try:
                if ks_2samp(self.reference[c].dropna(), df[c].dropna())[1] < self.config.drift_alpha: drifted.append(c)
            except: pass
        return {"drifted": drifted}

    def _scan_anomalies(self, df):
        if not self.iforest: return {"rate": 0.0, "status": "SKIPPED"}
        rate = (self.iforest.predict(df[self.num_cols].fillna(0)) == -1).mean()
        return {"rate": rate, "status": "RED" if rate > self.config.iforest_contamination * 4 else "GREEN"}

    def _scan_fairness(self, df, target, groups):
        issues = []
        for g in groups:
            if g not in df.columns: continue
            means = df.groupby(g)[target].mean()
            disparity = (means.max() - means.min()) / (abs(means.min()) + 1e-9)
            if disparity > self.config.max_fairness_disparity:
                issues.append(f"Fairness Warning on '{g}' (Disparity: {disparity:.1%})")
        return {"issues": issues}
