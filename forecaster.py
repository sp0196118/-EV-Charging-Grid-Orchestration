"""
GridPulse — Spatial-Temporal Demand Forecaster
===============================================
Standalone training + evaluation of the GNN-style zone load predictor.

Usage:
    python forecaster.py              # train + evaluate + print results
    python forecaster.py --plot       # also save forecast plot (requires matplotlib)

Architecture (production):
    PyTorch Geometric GraphSAGE → per-zone load distribution
    Input features  : [util_t, neighbour_util_agg, load_t-1..t-8, hour_enc]
    Output          : μ(t+1..t+H), σ(t+1..t+H)  — mean + std per step
    Training        : MSE loss on μ + NLL on σ (Gaussian log-likelihood)

This module:
    Uses the same feature engineering + calibrated parametric fallback
    so results are directly comparable to the production GNN.
"""

import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# ── Inlined minimal zone definitions so this file is self-contained ────────

ZONE_SPECS = [
    ('Z1','Downtown Core',    620,'commercial'),
    ('Z2','Transit Hub',      510,'transit'),
    ('Z3','Retail District',  460,'retail'),
    ('Z4','Residential NW',   360,'residential'),
    ('Z5','Business Park',    560,'commercial'),
    ('Z6','Airport Corridor', 510,'transit'),
    ('Z7','University Campus',310,'institutional'),
    ('Z8','Industrial East',  410,'industrial'),
]

TYPE_RATES = {
    'commercial':   {'morning':1.6,'evening':1.3,'base':0.30},
    'transit':      {'morning':1.8,'evening':1.9,'base':0.35},
    'retail':       {'morning':0.8,'evening':1.6,'base':0.25},
    'residential':  {'morning':0.5,'evening':2.0,'base':0.20},
    'institutional':{'morning':1.2,'evening':0.8,'base':0.25},
    'industrial':   {'morning':1.5,'evening':0.6,'base':0.20},
}

def demand_rate(hour: float, zone_type: str) -> float:
    m = TYPE_RATES.get(zone_type, TYPE_RATES['commercial'])
    return (m['base']
            + np.exp(-((hour - 8.0)**2) / 7.0)  * m['morning']
            + np.exp(-((hour -18.0)**2) / 5.0)  * m['evening'])


# ═══════════════════════════════════════════════════════════════════════════
#  SYNTHETIC TRAINING DATA GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

class SyntheticDataset:
    """
    Generate N days × 720 ticks (2-min resolution) of realistic zone loads.
    Includes:
      - Inhomogeneous Poisson EV arrivals
      - Weekday/weekend demand variation
      - Random disruption events (parking lot events, grid faults)
      - Autocorrelated residual noise
    """

    def __init__(self, n_days: int = 14, seed: int = 0):
        np.random.seed(seed)
        self.n_days   = n_days
        self.ticks    = n_days * 720          # 2-min ticks per day
        self.zones    = [(zid, cap, ztype) for zid,_,cap,ztype in ZONE_SPECS]

    def generate(self) -> pd.DataFrame:
        rows = []
        # Persistent state: each zone has a drifting "current sessions" count
        sessions = {zid: 0.0 for zid,_,__ in self.zones}
        noise    = {zid: 0.0 for zid,_,__ in self.zones}  # AR(1) noise

        for t in range(self.ticks):
            hour    = (6.0 + t * 2/60) % 24
            day_of_week = (t // 720) % 7
            is_weekend  = day_of_week >= 5
            weekend_mod = 0.65 if is_weekend else 1.0

            row = {'tick': t, 'hour': round(hour, 3)}

            for zid, cap, ztype in self.zones:
                lam = demand_rate(hour, ztype) * weekend_mod

                # Sessions arrive and depart (simplified queue)
                arrivals   = np.random.poisson(lam * 0.04)
                departures = np.random.binomial(max(0,int(sessions[zid])), 0.025)
                sessions[zid] = max(0, sessions[zid] + arrivals - departures)

                # AR(1) residual noise
                noise[zid] = 0.7 * noise[zid] + np.random.randn() * cap * 0.03

                # Occasional surge event
                surge = cap * 0.25 if (np.random.rand() < 0.002) else 0.0

                base_load = sessions[zid] * (cap / (lam * 5 + 1)) * 0.40
                load = float(np.clip(base_load + noise[zid] + surge, 0, cap * 1.1))

                row[f'{zid}_load'] = round(load, 2)
                row[f'{zid}_util'] = round(load / cap, 4)

            rows.append(row)

        return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════

class FeatureEngineer:
    """
    Build GraphSAGE-compatible feature tensors from raw load history.

    Feature vector per (zone, tick):
        [ util_t,               # current utilisation
          util_t-1 .. t-7,      # lag features (8 steps = 16 min)
          neighbour_avg,        # spatial message-pass aggregation
          hour_sin, hour_cos,   # cyclic time encoding
          weekend_flag,
          rate_forecast ]       # parametric demand rate
    """

    LAG_STEPS  = 8
    NEIGHBOURS = {
        'Z1': ['Z2','Z4','Z5'],
        'Z2': ['Z1','Z3','Z5'],
        'Z3': ['Z2','Z5','Z6'],
        'Z4': ['Z1','Z5','Z7'],
        'Z5': ['Z1','Z2','Z3','Z4','Z6','Z7','Z8'],
        'Z6': ['Z3','Z5','Z8'],
        'Z7': ['Z4','Z5'],
        'Z8': ['Z5','Z6'],
    }

    def build(self, df: pd.DataFrame, zone_id: str,
              cap: float, ztype: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns X (n_samples, n_features), y (n_samples, 12) — 12-step ahead
        """
        HORIZON = 12
        col     = f'{zone_id}_util'
        nbrs    = self.NEIGHBOURS.get(zone_id, [])

        X_rows, y_rows = [], []
        lags = self.LAG_STEPS

        for t in range(lags, len(df) - HORIZON):
            row = df.iloc[t]
            hour = row['hour']

            # Lag features
            lag_feats = df[col].values[t-lags:t][::-1]   # most recent first

            # Neighbour aggregation (mean utilisation)
            nbr_vals = [df.iloc[t][f'{n}_util'] for n in nbrs if f'{n}_util' in df.columns]
            nbr_agg  = float(np.mean(nbr_vals)) if nbr_vals else row[col]

            # Cyclic time
            h_sin = np.sin(2*np.pi * hour / 24)
            h_cos = np.cos(2*np.pi * hour / 24)

            # Weekend
            wkend = float((df.iloc[t]['tick'] // 720) % 7 >= 5)

            # Parametric rate
            rate_f = demand_rate(hour, ztype)

            feat = np.concatenate([lag_feats, [nbr_agg, h_sin, h_cos, wkend, rate_f]])
            tgt  = df[col].values[t:t+HORIZON]

            X_rows.append(feat)
            y_rows.append(tgt)

        return np.array(X_rows), np.array(y_rows)


# ═══════════════════════════════════════════════════════════════════════════
#  LIGHTWEIGHT FORECASTER  (Ridge regression as stand-in for GraphSAGE)
# ═══════════════════════════════════════════════════════════════════════════

class ZoneForecaster:
    """
    Per-zone Ridge regression trained on lag + spatial features.
    In production: replaced by shared GraphSAGE weights across all zones.
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self._models : Dict[str, Tuple[np.ndarray, np.ndarray]] = {}  # weights, intercepts

    def _ridge_fit(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Closed-form Ridge: W = (X'X + αI)^{-1} X'y"""
        n_feat = X.shape[1]
        A = X.T @ X + self.alpha * np.eye(n_feat)
        b = X.T @ y
        W = np.linalg.solve(A, b)        # shape (n_feat, horizon)
        intercept = y.mean(axis=0) - X.mean(axis=0) @ W
        return W, intercept

    def train(self, X: np.ndarray, y: np.ndarray, zone_id: str):
        # Normalise X
        self._mu  = X.mean(axis=0)
        self._std = X.std(axis=0) + 1e-8
        Xn = (X - self._mu) / self._std
        W, b = self._ridge_fit(Xn, y)
        self._models[zone_id] = (W, b, self._mu, self._std)

    def predict(self, x: np.ndarray, zone_id: str) -> np.ndarray:
        W, b, mu, std = self._models[zone_id]
        xn = (x - mu) / std
        return np.clip(xn @ W + b, 0, 1)   # utilisation in [0,1]

    def n_params(self) -> int:
        return sum(W.size for W,b,*_ in self._models.values())


# ═══════════════════════════════════════════════════════════════════════════
#  EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

class ForecastEvaluator:

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
                 cap_kw: float) -> Dict[str, float]:
        """
        Returns per-zone evaluation metrics over the test set.
        y shapes: (n_samples, horizon)  — utilisation [0,1]
        """
        err   = y_true - y_pred
        mae   = np.abs(err).mean()
        rmse  = np.sqrt((err**2).mean())
        mape  = (np.abs(err) / (y_true + 1e-6)).mean() * 100

        # Critical-zone detection accuracy
        threshold = 0.85   # "critical" if util > 85%
        tp = ((y_true > threshold) & (y_pred > threshold)).sum()
        fp = ((y_true <= threshold) & (y_pred > threshold)).sum()
        fn = ((y_true > threshold) & (y_pred <= threshold)).sum()
        precision = tp / max(tp+fp, 1)
        recall    = tp / max(tp+fn, 1)
        f1        = 2*precision*recall / max(precision+recall, 1e-6)

        # Convert to kW
        mae_kw  = mae  * cap_kw
        rmse_kw = rmse * cap_kw

        return {
            'MAE_util':  round(mae, 4),
            'RMSE_util': round(rmse, 4),
            'MAPE_%':    round(mape, 2),
            'MAE_kW':    round(mae_kw, 1),
            'RMSE_kW':   round(rmse_kw, 1),
            'Critical_Precision': round(precision, 3),
            'Critical_Recall':    round(recall, 3),
            'Critical_F1':        round(f1, 3),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def main(do_plot: bool = False):
    print("═"*64)
    print("  GridPulse — Forecaster Training Pipeline")
    print("  Features: lag-8 + spatial aggregation + cyclic time encoding")
    print("═"*64)

    # 1. Generate synthetic data
    print("\n  [1/4] Generating synthetic dataset (14 days, 2-min resolution)…")
    dataset = SyntheticDataset(n_days=14).generate()
    print(f"        {len(dataset):,} ticks × {len(ZONE_SPECS)} zones")

    # 2. Train/test split (last 3 days = test)
    split   = len(dataset) - 3 * 720
    df_train, df_test = dataset.iloc[:split], dataset.iloc[split:]
    print(f"        Train: {len(df_train):,}  |  Test: {len(df_test):,}")

    # 3. Feature engineering + training
    print("\n  [2/4] Engineering features & training per-zone models…")
    fe  = FeatureEngineer()
    mdl = ZoneForecaster(alpha=0.5)

    all_results = []
    for zid, _, cap, ztype in ZONE_SPECS:
        X_train, y_train = fe.build(df_train, zid, cap, ztype)
        mdl.train(X_train, y_train, zid)

    print(f"        Total model parameters: {mdl.n_params():,}")

    # 4. Evaluate on test set
    print("\n  [3/4] Evaluating on held-out test set…")
    print(f"\n  {'Zone':<8}  {'MAE kW':>7}  {'RMSE kW':>8}  "
          f"{'MAPE%':>6}  {'Prec':>5}  {'Rec':>5}  {'F1':>5}")
    print("  " + "─"*55)

    evaluator = ForecastEvaluator()
    for zid, _, cap, ztype in ZONE_SPECS:
        X_test, y_test = fe.build(df_test, zid, cap, ztype)
        if len(X_test) == 0:
            continue
        y_pred = np.array([mdl.predict(x, zid) for x in X_test])
        metrics = evaluator.evaluate(y_test, y_pred, cap)
        all_results.append({'zone': zid, **metrics})
        print(f"  {zid:<8}  {metrics['MAE_kW']:>7.1f}  {metrics['RMSE_kW']:>8.1f}  "
              f"{metrics['MAPE_%']:>6.1f}  {metrics['Critical_Precision']:>5.3f}  "
              f"{metrics['Critical_Recall']:>5.3f}  {metrics['Critical_F1']:>5.3f}")

    # Aggregate
    res_df = pd.DataFrame(all_results)
    print("  " + "─"*55)
    print(f"  {'MEAN':<8}  {res_df['MAE_kW'].mean():>7.1f}  "
          f"{res_df['RMSE_kW'].mean():>8.1f}  "
          f"{res_df['MAPE_%'].mean():>6.1f}  "
          f"{res_df['Critical_Precision'].mean():>5.3f}  "
          f"{res_df['Critical_Recall'].mean():>5.3f}  "
          f"{res_df['Critical_F1'].mean():>5.3f}")

    res_df.to_csv('forecaster_eval.csv', index=False)

    # 5. Plot (optional)
    if do_plot:
        _plot_forecast_sample(df_test, fe, mdl)

    print("\n  [4/4] Done. Results saved → forecaster_eval.csv")
    print("═"*64 + "\n")
    return res_df


def _plot_forecast_sample(df_test, fe, mdl):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(16, 10))
        fig.patch.set_facecolor('#0b0f1a')
        gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

        COLORS = ['#3b82f6','#14b8a6','#a855f7','#f59e0b',
                  '#22c55e','#ef4444','#ec4899','#f97316']

        for idx, (zid, _, cap, ztype) in enumerate(ZONE_SPECS):
            ax = fig.add_subplot(gs[idx//4, idx%4])
            ax.set_facecolor('#111827')
            for spine in ax.spines.values():
                spine.set_edgecolor('#3d5270')

            X_test, y_test = fe.build(df_test, zid, cap, ztype)
            if len(X_test) < 30:
                continue

            sample = 60   # use 60 samples
            t_range = np.arange(sample)
            actual  = y_test[:sample, 0] * cap
            pred    = np.array([mdl.predict(x, zid) for x in X_test[:sample]])[:,0] * cap

            ax.fill_between(t_range, actual, alpha=0.3, color=COLORS[idx])
            ax.plot(t_range, actual, color=COLORS[idx],  lw=1.5, label='Actual')
            ax.plot(t_range, pred,   color='#ffffff',    lw=1,   alpha=0.8,
                    linestyle='--', label='Forecast')
            ax.axhline(cap * 0.85, color='#ef4444', lw=0.8, linestyle=':')

            ax.set_title(zid, color='#e8edf5', fontsize=10, fontweight='bold')
            ax.tick_params(colors='#3d5270', labelsize=8)
            ax.set_xlabel('Test sample', color='#3d5270', fontsize=8)
            ax.set_ylabel('kW', color='#3d5270', fontsize=8)

        fig.suptitle('GridPulse — Forecaster Sample Predictions (step +1)',
                     color='#e8edf5', fontsize=13, fontweight='bold', y=0.98)
        plt.savefig('forecast_sample.png', dpi=140, bbox_inches='tight',
                    facecolor='#0b0f1a')
        print("        Plot saved → forecast_sample.png")
    except Exception as e:
        print(f"        (Plot skipped: {e})")


if __name__ == '__main__':
    main(do_plot='--plot' in sys.argv)
