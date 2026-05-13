# %% [markdown]
# # GHI Forecasting — v4: LSTM Base Model + Adaptive Conformal Prediction
# 
# **Changes vs v3:**
# 
# | Component | v3 | v4 |
# |---|---|---|
# | MLP base model | `sklearn.MLPRegressor` (flat 864-d vector) | **PyTorch LSTM** (sequence-aware, 144×6 input) |
# | Prediction intervals | Static Split Conformal | **Adaptive Conformal Prediction** (Gibbs & Candès, 2021) |
# | Interval metric | PICP, MPIW | PICP, MPIW, **Winkler Score** |
# 
# **Why LSTM?** The flat-window MLP treats all 864 input elements identically — it has no
# intrinsic notion of temporal order. An LSTM processes the same data as a (144, 6) tensor
# through gated recurrent units, allowing it to selectively remember or forget past states.
# Recent lags are naturally weighted higher via the hidden state mechanism, which is physically
# appropriate for solar irradiance where the most recent cloud-cover state is the strongest predictor.
# 
# **Why ACP?** The static split conformal coverage guarantee requires the calibration and test
# distributions to be exchangeable (i.i.d.). Solar irradiance data has inter-annual variability,
# seasonal non-stationarity, and slow aerosol/climate drift — all of which violate exchangeability.
# ACP replaces the fixed quantile with a per-step adaptive quantile that self-corrects whenever
# coverage drifts, achieving a time-average coverage guarantee even under distributional shift.
# 

# %%
# ─────────────────────────────────────────────────────────────────────────
# CELL 0 — Libraries
# Changes vs v3:
#   • Added: torch, torch.nn, torch.utils.data (for LSTM)
#   • Removed: sklearn.neural_network.MLPRegressor
# ─────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── PyTorch (LSTM) ───────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ── sklearn (unchanged) ──────────────────────────────────────────────────────
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# ── SHAP ─────────────────────────────────────────────────────────────────────
import shap

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'PyTorch device : {DEVICE}')
print('All libraries loaded successfully.')


# %%
# ─────────────────────────────────────────────────────────────────────────
# CELL 1 — Data Loading & Chronological Split
#
# BUG FIX vs original: the scaler is now fit ONLY on training data.
# Original code accidentally re-fit the scaler on test_data, causing
# inverse-transform to use test-set statistics → data leakage.
#
# Split strategy (strictly chronological, no shuffling):
#   Train  : first 70%  → base model training + OOF meta-features
#   Cal    : next  15%  → conformal prediction calibration set
#   Test   : last  15%  → completely held-out evaluation
# ─────────────────────────────────────────────────────────────────────────

data = pd.read_csv('merged_data_new.csv')

data['DateTime'] = pd.to_datetime(data[['Year', 'Month', 'Day', 'Hour', 'Minute']])
data.drop(columns=['Year', 'Month', 'Day', 'Hour', 'Minute'], inplace=True)
data.set_index('DateTime', inplace=True)

# ── Global constants ────────────────────────────────────────────────────────
FEATURES   = ['DHI', 'DNI', 'Clearsky DHI', 'Clearsky DNI', 'Clearsky GHI']
TARGET     = 'GHI'
ALL_COLS   = FEATURES + [TARGET]   # 5 features + 1 target = 6 columns
N_FEAT     = len(FEATURES)         # 5
TARGET_IDX = len(FEATURES)         # index 5 in the scaled array

# Forecast horizons in time steps (assumes 1-hour resolution).
# h = 1  →  1-hour-ahead  (very-short-term)
# h = 3  →  3-hour-ahead  (short-term)
# h = 6  →  6-hour-ahead  (intra-day)
# h = 24 → 24-hour-ahead  (day-ahead)
HORIZONS   = [1, 3, 6, 24]
PAST_STEP  = 144   # 144-hour (6-day) look-back window

# ── Chronological split ─────────────────────────────────────────────────────
N          = len(data)
TRAIN_END  = int(N * 0.70)
CAL_END    = int(N * 0.85)

raw_array  = data[ALL_COLS].values   # shape (N, 6)

# Fit scaler on training data ONLY — this is critical for no leakage
scaler = MinMaxScaler()
scaler.fit(raw_array[:TRAIN_END])

# Apply (transform) to entire array using the training-fit scaler
all_scaled = scaler.transform(raw_array)   # shape (N, 6)

print(f'Total samples : {N:,}')
print(f'Train end idx : {TRAIN_END:,}  ({TRAIN_END/N*100:.1f}%)')
print(f'Cal   end idx : {CAL_END:,}  ({(CAL_END-TRAIN_END)/N*100:.1f}%)')
print(f'Test  samples : {N-CAL_END:,}  ({(N-CAL_END)/N*100:.1f}%)')


# %%
# ─────────────────────────────────────────────────────────────────────────
# CELL 2 — Exploratory Analysis: Correlation Heatmaps (original preserved)
# ─────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

wind_temp_cols  = ['Temperature', 'Wind Speed', 'Wind Direction', 'Pressure']
solar_cols      = ['DHI', 'DNI', 'GHI', 'Clearsky DHI', 'Clearsky DNI', 'Clearsky GHI']

sns.heatmap(data[wind_temp_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=axes[0])
axes[0].set_title('Correlation — Wind & Temperature Features')

sns.heatmap(data[solar_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=axes[1])
axes[1].set_title('Correlation — Solar Irradiance Features')

plt.tight_layout()
plt.savefig('correlation_heatmaps.jpeg', dpi=300)
plt.show()


# %%
# ─────────────────────────────────────────────────────────────────────────
# CELL 3 — Utility Functions, LSTM Architecture, and Model Factory
#
# Changes vs v3:
#   • build_windows_indexed   : UNCHANGED
#   • inverse_transform_target: UNCHANGED
#   • compute_metrics         : UNCHANGED
#   • NEW: GHI_LSTM           : PyTorch 2-layer LSTM with FC head
#   • NEW: LSTMWrapper        : sklearn-compatible wrapper (fit / predict API)
#   • get_base_models()       : 'MLP' replaced by 'LSTM' (LSTMWrapper)
#                               accepts lstm_epochs param for OOF vs final
# ─────────────────────────────────────────────────────────────────────────

# ── Unchanged utility functions ───────────────────────────────────────────────

def build_windows_indexed(scaled_full, past_step, horizon):
    """
    Constructs supervised windows from the full scaled array.
    For each index i:
      X_i     = scaled_full[i : i+past_step, :N_FEAT].flatten()  → (past_step*N_FEAT,)
      y_i     = scaled_full[i + past_step + horizon - 1, TARGET_IDX]
      y_pos_i = i + past_step + horizon - 1
    """
    X, y, y_pos = [], [], []
    max_i = len(scaled_full) - past_step - horizon + 1
    for i in range(max_i):
        X.append(scaled_full[i : i + past_step, :N_FEAT].flatten())
        pos = i + past_step + horizon - 1
        y.append(scaled_full[pos, TARGET_IDX])
        y_pos.append(pos)
    return np.array(X), np.array(y), np.array(y_pos)


def inverse_transform_target(scaler, y_scaled):
    """y_actual = y_scaled / scale_[TARGET_IDX] + data_min_[TARGET_IDX]"""
    return y_scaled / scaler.scale_[TARGET_IDX] + scaler.data_min_[TARGET_IDX]


def compute_metrics(y_true, y_pred):
    """Returns (MAE, MSE, RMSE, R²) in original W/m² scale."""
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2


# ─────────────────────────────────────────────────────────────────────────────
#  LSTM ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

class GHI_LSTM(nn.Module):
    """
    Two-layer stacked LSTM followed by a small fully-connected head.

    Forward pass:
      Input  : x  ∈ ℝ^{batch × seq_len × input_size}   (batch_first=True)
      LSTM   : processes full sequence; returns hidden states for all t
               h_t ∈ ℝ^{batch × seq_len × hidden_size}
      Readout: we use the LAST hidden state  h_{T-1} ∈ ℝ^{batch × hidden_size}
               (T-1 = most recent timestep), which encodes the full history
               via the LSTM's gating mechanism.
      Head   : Linear(hidden_size→64) → ReLU → Dropout → Linear(64→1)
      Output : scalar ŷ ∈ ℝ^{batch}  (scaled GHI)

    Why the last hidden state?
      The LSTM forget gate f_t = σ(W_f[h_{t-1}, x_t] + b_f) decides how
      much past information to retain at each step. By design, h_{T-1}
      already integrates information from all 144 previous steps with
      learned, horizon-appropriate forgetting weights. Using only h_{T-1}
      for prediction is therefore more principled than, e.g., averaging
      all hidden states.
    """
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            # dropout between LSTM layers (only meaningful when num_layers>1)
            dropout = dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _   = self.lstm(x)         # out: (batch, seq_len, hidden_size)
        last     = out[:, -1, :]        # last hidden state: (batch, hidden_size)
        return self.head(last).squeeze(-1)  # (batch,)


class LSTMWrapper:
    """
    sklearn-compatible wrapper around GHI_LSTM.

    Exposes .fit(X_flat, y) and .predict(X_flat) so the wrapper drops
    into the stacking OOF loop exactly like RandomForestRegressor or
    XGBRegressor — no changes needed to Cells 5–8.

    Internally it reshapes the flat input:
        X_flat  ∈ ℝ^{n × (past_step * N_FEAT)}     (e.g. n × 720)
        X_seq   ∈ ℝ^{n × past_step × N_FEAT}       (e.g. n × 144 × 5)
    before feeding to GHI_LSTM.

    Training details:
      Optimiser : Adam with lr=1e-3
      Loss      : MSELoss (appropriate for regression in scaled [0,1] space)
      Scheduler : ReduceLROnPlateau — halves lr after 5 epochs without
                  improvement; prevents overshooting in later training.
      Grad clip : max_norm=1.0 — prevents exploding gradients, which are
                  a known pathology in vanilla LSTM training on long sequences.
      shuffle   : False — preserves chronological order within each mini-batch,
                  consistent with our walk-forward validation philosophy.
    """
    def __init__(self, past_step, n_feat,
                 hidden_size=128, num_layers=2, dropout=0.2,
                 epochs=30, batch_size=512, lr=1e-3):
        self.past_step   = past_step
        self.n_feat      = n_feat
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.dropout     = dropout
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.lr          = lr
        self.model       = None   # set in fit()

    def _to_tensor(self, X_flat):
        """Reshape flat (n, past_step*n_feat) → (n, past_step, n_feat) tensor."""
        return torch.tensor(
            X_flat.reshape(-1, self.past_step, self.n_feat),
            dtype=torch.float32
        ).to(DEVICE)

    def fit(self, X_flat, y):
        X_t = self._to_tensor(X_flat)
        y_t = torch.tensor(y, dtype=torch.float32).to(DEVICE)

        self.model = GHI_LSTM(
            input_size  = self.n_feat,
            hidden_size = self.hidden_size,
            num_layers  = self.num_layers,
            dropout     = self.dropout
        ).to(DEVICE)

        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, patience=5, factor=0.5
        )
        loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.batch_size,
            shuffle=False   # chronological order
        )

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimiser.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimiser.step()
                epoch_loss += loss.item()
            scheduler.step(epoch_loss)
            if (epoch + 1) % 10 == 0:
                print(f'      [LSTM] Epoch {epoch+1:3d}/{self.epochs}  '
                      f'loss={epoch_loss/len(loader):.6f}')
        return self

    def predict(self, X_flat):
        self.model.eval()
        X_t   = self._to_tensor(X_flat)
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_t), self.batch_size):
                preds.append(
                    self.model(X_t[i : i + self.batch_size]).cpu().numpy()
                )
        return np.concatenate(preds)


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def get_base_models(lstm_epochs=40):
    """
    Returns a fresh dict of base learners.

    lstm_epochs controls how many training epochs the LSTM runs:
      • 20-30 epochs during OOF folds (called 5 times per horizon)
        — keeps total compute time reasonable while still producing
          meaningful OOF predictions for the meta-learner.
      • 50 epochs for the final full-training refit (Cell 7)
        — gives the model more capacity since it trains on ~78k samples
          instead of the OOF subset.

    Total LSTM trainings: 4 horizons × 5 folds × oof_epochs
                        + 4 horizons × 1 final × final_epochs
    """
    return {
        'RF': RandomForestRegressor(
            n_estimators=500, max_depth=10, random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(
            n_estimators=500, max_depth=10, learning_rate=0.05,
            random_state=42, n_jobs=-1, verbosity=0),
        'LSTM': LSTMWrapper(
            past_step=PAST_STEP, n_feat=N_FEAT,
            hidden_size=256, num_layers=2, dropout=0.2,
            epochs=lstm_epochs, batch_size=256, lr=3e-4),
    }


print('Utility functions and LSTM architecture defined.')
print(f'GHI_LSTM parameter count (per horizon): '
      f'{sum(p.numel() for p in GHI_LSTM(N_FEAT if "N_FEAT" in dir() else 5).parameters()):,}')


# %%
# ─────────────────────────────────────────────────────────────────────────
# CELL 4 — Build All Windows (Horizon Loop)
#
# We build windows from the COMPLETE scaled array once per horizon, then
# partition them by the absolute position of the y-value:
#   y_pos  < TRAIN_END  → training sample
#   y_pos in [TRAIN_END, CAL_END)  → calibration sample
#   y_pos >= CAL_END   → test sample
#
# This ensures that X-windows whose look-back spans the train/cal boundary
# are handled correctly — the model has never seen those context rows at
# training time (they are in the future), so no leakage occurs.
# ─────────────────────────────────────────────────────────────────────────

all_windows = {}   # all_windows[h] = dict with X/y split arrays

for h in HORIZONS:
    X_all, y_all, y_pos = build_windows_indexed(all_scaled, PAST_STEP, h)

    tr_mask  = y_pos < TRAIN_END
    cal_mask = (y_pos >= TRAIN_END) & (y_pos < CAL_END)
    te_mask  = y_pos >= CAL_END

    all_windows[h] = {
        'X_train' : X_all[tr_mask],
        'y_train' : y_all[tr_mask],
        'X_cal'   : X_all[cal_mask],
        'y_cal'   : y_all[cal_mask],
        'X_test'  : X_all[te_mask],
        'y_test'  : y_all[te_mask],
        'y_pos_te': y_pos[te_mask],    # absolute positions (for DateTime reindexing)
    }

    n_tr, n_cal, n_te = tr_mask.sum(), cal_mask.sum(), te_mask.sum()
    print(f'h={h:2d}h  train={n_tr:,}  cal={n_cal:,}  test={n_te:,}  '
          f'feature_dim={X_all.shape[1]}')


# %%
# ─────────────────────────────────────────────────────────────────────────
# CELL 5 — Out-of-Fold (OOF) Predictions for Stacking (Layer 1)
#
# Mathematical setup is identical to v3:
#   Z_{i,m} = prediction of model m on sample i without training on i.
#   Meta-feature matrix Z ∈ ℝ^{n_train × 3} feeds the Ridge meta-learner.
#
# Change vs v3: get_base_models(lstm_epochs=20) — the LSTM trains for
# 20 epochs per OOF fold rather than the full 50, which is a deliberate
# compute-accuracy trade-off. Even lightly trained OOF LSTM predictions
# give the Ridge meta-learner a useful signal about which samples the LSTM
# handles well vs poorly. The full 50-epoch training happens in Cell 7.
# ─────────────────────────────────────────────────────────────────────────

N_FOLDS   = 5
tscv      = TimeSeriesSplit(n_splits=N_FOLDS)
oof_store = {}

for h in HORIZONS:
    print(f'\n{"="*62}')
    print(f'  OOF generation — horizon h = {h}h')
    print(f'{"="*62}')

    X_tr = all_windows[h]['X_train']
    y_tr = all_windows[h]['y_train']
    n    = len(X_tr)

    Z_oof = np.full((n, 3), np.nan)

    for fold_k, (tr_idx, val_idx) in enumerate(tscv.split(X_tr)):
        print(f'  Fold {fold_k+1}/{N_FOLDS}  '
              f'train={len(tr_idx):,}  val={len(val_idx):,}')
        # 20 epochs for OOF folds — see docstring above
        for col, (name, model) in enumerate(get_base_models(lstm_epochs=20).items()):
            model.fit(X_tr[tr_idx], y_tr[tr_idx])
            Z_oof[val_idx, col] = model.predict(X_tr[val_idx])
        print(f'    Fold {fold_k+1} complete.')

    valid = ~np.isnan(Z_oof).any(axis=1)
    oof_store[h] = {
        'Z': Z_oof[valid],
        'y': y_tr[valid],
    }
    print(f'  OOF valid samples: {valid.sum():,}/{n:,}')


# %%
# ─────────────────────────────────────────────────────────────────────────
# CELL 6 — Train Ridge Meta-Learner (Stacking Layer 2)
#
# The meta-learner maps base model OOF predictions → target GHI.
# Ridge is chosen because:
#   (a) Closed-form solution: w* = (Z^T Z + λI)^{-1} Z^T y
#   (b) L2 regularisation prevents degenerate zero-weight solutions;
#       with λ=1 each model is softly included even if correlated.
#   (c) Coefficients are directly interpretable as blend weights.
#
# All operations remain in the [0,1] scaled domain for numerical consistency.
# ─────────────────────────────────────────────────────────────────────────

meta_learners = {}   # one per horizon

for h in HORIZONS:
    Z = oof_store[h]['Z']
    y = oof_store[h]['y']
    ridge = Ridge(alpha=1.0)
    ridge.fit(Z, y)
    meta_learners[h] = ridge

    w = ridge.coef_
    print(f'h={h:2d}h  Ridge weights → RF: {w[0]:+.4f}  '
          f'XGB: {w[1]:+.4f}  MLP: {w[2]:+.4f}  '
          f'intercept: {ridge.intercept_:+.4f}')


# %%
# ─────────────────────────────────────────────────────────────────────────
# CELL 7 — Retrain Base Models on Full Training Set
#
# Standard two-phase stacking protocol:
#   Phase 1 (done): OOF meta-features → Ridge meta-learner fitted.
#   Phase 2 (here): retrain all base models on full D_train.
#   Phase 3 (Cell 8): base predictions on cal/test → Ridge → ensemble.
#
# Change vs v3: lstm_epochs=50 gives the LSTM a thorough final training
# on the full 78k-sample training set.
# ─────────────────────────────────────────────────────────────────────────

final_base = {}

for h in HORIZONS:
    print(f'\nh={h}h — retraining on {len(all_windows[h]["X_train"]):,} samples')
    X_tr = all_windows[h]['X_train']
    y_tr = all_windows[h]['y_train']
    final_base[h] = {}
    # 50 epochs for final training (more thorough than OOF folds)
    for name, model in get_base_models(lstm_epochs=150).items():
        print(f'  Training {name} ...')
        model.fit(X_tr, y_tr)
        final_base[h][name] = model
    print(f'  h={h}h done.')


# %%
# ─────────────────────────────────────────────────────────────────────────
# CELL 8 — Generate Calibration & Test Predictions
#
# Identical logic to v3. Only change: key 'MLP' → 'LSTM' throughout.
# ─────────────────────────────────────────────────────────────────────────

cal_results  = {}
test_results = {}

for h in HORIZONS:
    w = all_windows[h]

    # ── Calibration set ──────────────────────────────────────────────────────
    Z_cal = np.column_stack([
        final_base[h]['RF'].predict(w['X_cal']),
        final_base[h]['XGBoost'].predict(w['X_cal']),
        final_base[h]['LSTM'].predict(w['X_cal']),
    ])
    y_hat_cal_s = meta_learners[h].predict(Z_cal)

    cal_results[h] = {
        'y_true'       : inverse_transform_target(scaler, w['y_cal']),
        'y_hat'        : inverse_transform_target(scaler, y_hat_cal_s),
        'y_hat_scaled' : y_hat_cal_s,
    }

    # ── Test set ─────────────────────────────────────────────────────────────
    Z_test = np.column_stack([
        final_base[h]['RF'].predict(w['X_test']),
        final_base[h]['XGBoost'].predict(w['X_test']),
        final_base[h]['LSTM'].predict(w['X_test']),
    ])
    y_hat_te_s = meta_learners[h].predict(Z_test)

    test_results[h] = {
        'y_true'  : inverse_transform_target(scaler, w['y_test']),
        'y_hat'   : inverse_transform_target(scaler, y_hat_te_s),
        'RF'      : inverse_transform_target(scaler, Z_test[:, 0]),
        'XGBoost' : inverse_transform_target(scaler, Z_test[:, 1]),
        'LSTM'    : inverse_transform_target(scaler, Z_test[:, 2]),
        'y_pos'   : w['y_pos_te'],
    }

print('Calibration and test predictions generated for all horizons.')


# %%
# ─────────────────────────────────────────────────────────────────────────
# CELL 9 — Adaptive Conformal Prediction (Gibbs & Candes, 2021)
#
# ── Why we need ACP ────────────────────────────────────────────────────────
# Split conformal prediction's coverage guarantee requires EXCHANGEABILITY:
# calibration and test samples must be i.i.d. from the same distribution.
# Solar irradiance time series violates this because:
#   (a) Inter-annual solar variability (ENSO, volcanic aerosols)
#   (b) Seasonal non-stationarity in GHI variance
#   (c) Slow climate drift over the 15-year dataset
# This caused the empirical coverage in v3 to fall below nominal
# (88.0% at h=1 instead of the claimed 90%) — a correctness violation.
#
# ── ACP theory (Gibbs & Candes, NeurIPS 2021) ──────────────────────────────
# Let  s_i = |y_i - ŷ_i|  be the nonconformity score on the calibration set.
# Sort and store cal_scores.  Initialize  α_0 = α.
#
# For each test step  t = 1, …, T:
#
#   1. Compute adaptive quantile:
#        level_t  = min( ceil((n+1)(1 - α_t)) / n , 1 )
#        q̂_t     = Quantile(cal_scores, level_t)
#
#   2. Predict interval:
#        Ĉ_t = [ ŷ_t - q̂_t ,  ŷ_t + q̂_t ]
#
#   3. Observe y_t, compute miscoverage indicator:
#        err_t = 1{ y_t ∉ Ĉ_t }
#
#   4. Update the running miscoverage target:
#        α_{t+1} = clip( α_t + γ·(α - err_t) , 0.001, 0.999 )
#
#      Intuition of the update:
#        • err_t = 1 (interval MISSED):  α - 1 < 0  ⟹  α_{t+1} < α_t
#          Smaller α_t  →  larger (1-α_t)  →  higher quantile  →  wider interval.
#        • err_t = 0 (interval COVERED): α - 0 = α > 0  ⟹  α_{t+1} > α_t
#          Larger α_t  →  smaller (1-α_t)  →  lower quantile  →  narrower interval.
#      The algorithm self-corrects to maintain the long-run coverage target.
#
# ── Coverage guarantee ─────────────────────────────────────────────────────
#   Time-average miscoverage is bounded regardless of distribution:
#     |  (1/T) Σ_{t=1}^T err_t  -  α  |  ≤  γ / (1 - γ)
#   With γ = 0.02:  bound ≈ 0.020  →  coverage lies in [88%, 92%] for α=0.10.
#   This holds WITHOUT any exchangeability or stationarity assumption.
#
# ── Winkler Score ───────────────────────────────────────────────────────────
#   The Winkler score jointly penalises interval width and coverage violations:
#
#     W_t(α) = (u_t - l_t)
#              + (2/α)·max(l_t - y_t, 0)    ← penalty if y below lower bound
#              + (2/α)·max(y_t - u_t, 0)    ← penalty if y above upper bound
#
#   Mean Winkler = (1/T) Σ W_t.  Lower is better.  The (2/α) factor means
#   a miss is weighted roughly 20× the interval width at α=0.10.
# ─────────────────────────────────────────────────────────────────────────

ALPHA_LEVELS = [0.10, 0.05]
GAMMA        = 0.02   # ACP learning rate; see bound above
conformal    = {}     # conformal[h][alpha] = results dict


def winkler_score(y_true, lower, upper, alpha):
    """
    Vectorised Winkler score.
    W_t = width + (2/alpha)*max(lower - y, 0) + (2/alpha)*max(y - upper, 0)
    """
    width   = upper - lower
    penalty = (2.0 / alpha) * (
        np.maximum(lower - y_true, 0) + np.maximum(y_true - upper, 0)
    )
    return float(np.mean(width + penalty))


def adaptive_conformal(cal_scores, y_hat_test, y_true_test,
                        alpha=0.10, gamma=GAMMA):
    """
    Runs the ACP online update loop over the test sequence.

    Returns
    -------
    lower, upper   : per-step prediction bounds  (W/m²)
    alpha_trace    : trajectory of α_t over test steps
    coverage       : empirical PICP
    avg_width      : mean interval width (MPIW)  (W/m²)
    winkler        : mean Winkler score
    """
    n_cal   = len(cal_scores)
    n_test  = len(y_hat_test)
    cal_sorted = np.sort(cal_scores)   # sort once for fast quantile lookup

    alpha_t     = alpha                # running miscoverage level
    lower       = np.empty(n_test)
    upper       = np.empty(n_test)
    alpha_trace = np.empty(n_test)

    for t in range(n_test):
        # Step 1: compute q̂_t at the current adaptive level
        level   = min(np.ceil((n_cal + 1) * (1 - alpha_t)) / n_cal, 1.0)
        q_hat_t = np.quantile(cal_sorted, level)

        # Step 2: form interval
        lower[t] = y_hat_test[t] - q_hat_t
        upper[t] = y_hat_test[t] + q_hat_t
        alpha_trace[t] = alpha_t

        # Step 3: observe y_t and compute miscoverage
        err_t = 0.0 if (lower[t] <= y_true_test[t] <= upper[t]) else 1.0

        # Step 4: update α_t  (clip to (0.001, 0.999) for numerical safety)
        alpha_t = float(np.clip(alpha_t + gamma * (alpha - err_t), 0.001, 0.999))

    coverage  = float(np.mean((y_true_test >= lower) & (y_true_test <= upper)))
    avg_width = float(np.mean(upper - lower))
    winkler   = winkler_score(y_true_test, lower, upper, alpha)

    return lower, upper, alpha_trace, coverage, avg_width, winkler


# ── Run ACP for all horizons and alpha levels ─────────────────────────────────
for h in HORIZONS:
    conformal[h] = {}
    # Calibration nonconformity scores in W/m²
    cal_scores = np.abs(
        cal_results[h]['y_true'] - cal_results[h]['y_hat']
    )
    y_hat_test  = test_results[h]['y_hat']
    y_true_test = test_results[h]['y_true']

    for alpha in ALPHA_LEVELS:
        lower, upper, alpha_trace, coverage, avg_width, winkler = adaptive_conformal(
            cal_scores, y_hat_test, y_true_test, alpha=alpha, gamma=GAMMA
        )
        conformal[h][alpha] = {
            'lower'      : lower,
            'upper'      : upper,
            'alpha_trace': alpha_trace,
            'coverage'   : coverage,
            'avg_width'  : avg_width,
            'winkler'    : winkler,
        }
        print(
            f'h={h:2d}h  alpha={alpha:.2f}  '
            f'PICP={coverage*100:5.1f}%  '
            f'(nominal {(1-alpha)*100:.0f}%)  '
            f'MPIW={avg_width:6.1f} W/m2  '
            f'Winkler={winkler:.2f}'
        )


# %%
# ─────────────────────────────────────────────────────────────────────────
# CELL 9b — ACP Alpha-Trace Visualisation  [NEW in v4]
#
# This figure shows how α_t evolves over the test sequence for h=1h.
# It is a key diagnostic plot that demonstrates:
#   (a) The model is adapting — α_t is not flat.
#   (b) Despite adaptation, α_t stays near the nominal α most of the time.
#   (c) Spikes in α_t correspond to periods where the model consistently
#       missed (e.g., sudden cloud events), after which it widens intervals.
# Publishing this plot alongside the PICP table gives reviewers a clear
# picture of how ACP behaves temporally, not just in aggregate.
# ─────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(len(HORIZONS), 1, figsize=(16, 3.5 * len(HORIZONS)),
                          sharex=False)

for ax, h in zip(axes, HORIZONS):
    for alpha, col, nom in [(0.10, 'steelblue', '90%'), (0.05, 'tomato', '95%')]:
        tr    = conformal[h][alpha]['alpha_trace']
        t_idx = np.arange(len(tr))
        ax.plot(t_idx, tr * 100, color=col, lw=0.8, alpha=0.8,
                label=f'alpha_t (nominal {nom})')
        ax.axhline(alpha * 100, color=col, lw=1.5, ls='--', alpha=0.5,
                   label=f'Nominal alpha={alpha*100:.0f}%')

    ax.set_ylabel('alpha_t (%)')
    ax.set_xlabel('Test step')
    ax.set_title(f'ACP Adaptive Miscoverage Level — h={h}h  (gamma={GAMMA})')
    ax.legend(fontsize=8, ncol=4)
    ax.grid(True, alpha=0.3)

plt.suptitle(
    'ACP alpha_t Trace: How the Adaptive Quantile Responds to Coverage Events',
    fontsize=12, y=1.01
)
plt.tight_layout()
plt.savefig('acp_alpha_trace.jpeg', dpi=300, bbox_inches='tight')
plt.show()
print('ACP bound guarantee: |time-avg miscoverage - alpha| <= gamma/(1-gamma) =',
      round(GAMMA / (1 - GAMMA), 4))


# %%
# ─────────────────────────────────────────────────────────────────────────
# CELL 10 — Error Metrics: All Models x All Horizons
#
# Changes vs v3:
#   • 'MLP' → 'LSTM' in the model name list
#   • Added: PICP, MPIW, Winkler from ACP results (90% level)
# ─────────────────────────────────────────────────────────────────────────

metrics_table = {}

for h in HORIZONS:
    tr     = test_results[h]
    y_true = tr['y_true']
    rows   = {}
    for mname in ['RF', 'XGBoost', 'LSTM', 'Ensemble']:
        y_pred = tr['y_hat'] if mname == 'Ensemble' else tr[mname]
        mae, mse, rmse, r2 = compute_metrics(y_true, y_pred)
        rows[mname] = [mae, mse, rmse, r2]
    metrics_table[h] = pd.DataFrame(rows, index=['MAE', 'MSE', 'RMSE', 'R2'])
    print(f'\n── Horizon h = {h}h ──')
    print(metrics_table[h].round(4))
    # ACP summary for 90% level
    acp = conformal[h][0.10]
    print(f'   ACP (90%): PICP={acp["coverage"]*100:.1f}%  '
          f'MPIW={acp["avg_width"]:.1f} W/m2  '
          f'Winkler={acp["winkler"]:.2f}')


# %%
# ─────────────────────────────────────────────────────────────────────────
# CELL 11 — SHAP Explainability (h=1h)
#
# SHAP Shapley value (unchanged theory):
#   phi_j(f,X) = Sum_{S in F\{j}} |S|!(|F|-|S|-1)!/|F|! * [f(S+j) - f(S)]
#
# Explainer selection:
#   RF, XGBoost : TreeExplainer     — exact, O(TLD^2) per tree  [unchanged]
#   LSTM        : GradientExplainer — integrated gradients via backprop
#                 (Sundararajan et al., 2017).  Requires a 2D output wrapper
#                 because SHAP internally calls outputs[:, idx], which needs
#                 shape (batch, n_outputs).  GHI_LSTM.forward() squeezes to
#                 (batch,) for MSELoss compatibility, so we wrap it.
#
# Output shape note:
#   GradientExplainer returns SHAP values with the SAME shape as the INPUT,
#   not the output.  So even through the wrapper returns (batch,1), the SHAP
#   values are (N_SHAP, 144, 5) — one value per (sample, lag, feature).
#   When the wrapper is used, GradientExplainer returns a list of one array;
#   we unwrap it below.
# ─────────────────────────────────────────────────────────────────────────

H_SHAP = 1
N_SHAP = 200   # GradientExplainer is fast; 200 samples give stable estimates

X_test_shap = all_windows[H_SHAP]['X_test'][:N_SHAP]   # (200, 720) flat
X_train_bg  = all_windows[H_SHAP]['X_train']           # background pool

print(f'Computing SHAP on {N_SHAP} test samples (h={H_SHAP}h) ...')

# ── 1. XGBoost : TreeExplainer (exact, unchanged) ────────────────────────────
print('  XGBoost TreeExplainer ...', end=' ', flush=True)
xgb_explainer = shap.TreeExplainer(final_base[H_SHAP]['XGBoost'])
shap_xgb      = xgb_explainer.shap_values(X_test_shap)   # (N_SHAP, 720)
print('done')

# ── 2. RF : TreeExplainer (exact, unchanged) ──────────────────────────────────
print('  RF TreeExplainer ...', end=' ', flush=True)
rf_explainer = shap.TreeExplainer(final_base[H_SHAP]['RF'])
shap_rf      = rf_explainer.shap_values(X_test_shap)     # (N_SHAP, 720)
print('done')

# ── 3. LSTM : GradientExplainer with output-shape wrapper ────────────────────
#
# ROOT CAUSE OF THE IndexError:
#   GHI_LSTM.forward() ends with .squeeze(-1), converting (batch,1) → (batch,).
#   SHAP's internal line  `selected = [val for val in outputs[:, idx]]`
#   performs a 2D column-slice.  A 1D tensor has no second axis, so
#   outputs[:, 0] raises:  IndexError: too many indices for tensor of dimension 1
#
# FIX: _SHAPOutputWrapper adds .unsqueeze(-1) after the forward pass,
#   restoring (batch, 1) so SHAP sees a proper 2D output matrix.
#   Training weights are untouched — only the output shape is changed
#   for the purpose of attribution computation.
#
class _SHAPOutputWrapper(nn.Module):
    """
    Thin wrapper around GHI_LSTM that returns shape (batch, 1) instead of
    (batch,).  Required because shap.GradientExplainer assumes 2D output:
        outputs[:, idx]  — fails if outputs.ndim == 1
    The unsqueeze does not affect gradient computation: d/dx unsqueeze(f(x))
    = unsqueeze(d/dx f(x)), so SHAP values are numerically identical to what
    you would get from a model that natively returned (batch, 1).
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).unsqueeze(-1)   # (batch,) → (batch, 1)


print('  LSTM GradientExplainer ...', end=' ', flush=True)

lstm_model = final_base[H_SHAP]['LSTM'].model   # raw GHI_LSTM (trained)
lstm_model.eval()

# Wrap for SHAP — training weights are completely unaffected
lstm_shap_model = _SHAPOutputWrapper(lstm_model).to(DEVICE)
lstm_shap_model.eval()

# Background: 100 training samples reshaped to (100, 144, 5) tensor
bg_seq = torch.tensor(
    X_train_bg[:100].reshape(-1, PAST_STEP, N_FEAT),
    dtype=torch.float32
).to(DEVICE)

# Explain: N_SHAP test samples reshaped to (N_SHAP, 144, 5) tensor
ex_seq = torch.tensor(
    X_test_shap.reshape(-1, PAST_STEP, N_FEAT),
    dtype=torch.float32
).to(DEVICE)

lstm_explainer = shap.GradientExplainer(lstm_shap_model, bg_seq)
shap_lstm_raw  = lstm_explainer.shap_values(ex_seq)

# GradientExplainer returns a list when the model has n_outputs outputs.
# _SHAPOutputWrapper returns (batch, 1), so SHAP treats the model as having
# 1 output neuron and appends the output dimension at the END of the
# attribution tensor.  The actual returned shape is (N_SHAP, 144, 5, 1),
# NOT (N_SHAP, 144, 5) as you might expect.
#
# Step 1 — unwrap the outer list to get the single ndarray.
if isinstance(shap_lstm_raw, list):
    shap_lstm_raw = shap_lstm_raw[0]
shap_lstm = np.array(shap_lstm_raw)

# Step 2 — squeeze the trailing size-1 output dimension.
# Without this squeeze, shap_lstm is (N_SHAP, 144, 5, 1) and
# aggregate_shap_by_variable would return (5, 1) instead of (5,),
# causing: ValueError: Data must be 1-dimensional, got ndarray of shape (5,1).
if shap_lstm.ndim == 4 and shap_lstm.shape[-1] == 1:
    shap_lstm = shap_lstm.squeeze(-1)   # (N_SHAP, 144, 5, 1) → (N_SHAP, 144, 5)
print('done')


def aggregate_shap_by_variable(shap_mat, past_step, n_feat):
    """
    Compute mean |SHAP| per physical input variable, averaged over
    all samples and all lag positions.

    Handles both shapes that arise in this notebook:
      2D (N, past_step*n_feat)  — RF and XGBoost (flat input SHAP)
      3D (N, past_step, n_feat) — LSTM GradientExplainer (sequence SHAP)

    Returns shape (n_feat,):
      Phi_d = mean_i mean_k |phi_{k*n_feat+d}(X_i)|  for d = 0..4
    """
    arr = np.asarray(shap_mat)
    if arr.ndim == 2:                           # flat RF/XGB input
        arr = arr.reshape(-1, past_step, n_feat)
    # Defensive squeeze: if a trailing size-1 output dimension survived
    # (e.g. from a SHAP version that behaves differently), remove it.
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    # arr is now (N, past_step, n_feat) for all models
    return np.abs(arr).mean(axis=(0, 1))        # → (n_feat,)


shap_per_var = pd.DataFrame({
    'XGBoost': aggregate_shap_by_variable(shap_xgb,  PAST_STEP, N_FEAT),
    'RF'     : aggregate_shap_by_variable(shap_rf,   PAST_STEP, N_FEAT),
    'LSTM'   : aggregate_shap_by_variable(shap_lstm, PAST_STEP, N_FEAT),
}, index=FEATURES)

print('\nAggregated |SHAP| per input variable (h=1h):')
print(shap_per_var.round(6))


# %%
# ─────────────────────────────────────────────────────────────────────────
# CELL 12 — Actual vs Predicted GHI (All Horizons, Inset Zoom)
# Change vs v3: 'MLP' key/label → 'LSTM'
# ─────────────────────────────────────────────────────────────────────────

def plot_actual_vs_predicted(h=1, x_range=(500, 1220), x_zm=(900, 924)):
    tr  = test_results[h]
    idx = np.arange(len(tr['y_true']))

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(idx, tr['y_true'],  color='orange', lw=2,   label='Actual GHI')
    ax.plot(idx, tr['RF'],      color='blue',   lw=1,   ls='--', label='RF')
    ax.plot(idx, tr['XGBoost'], color='green',  lw=1,   ls='--', label='XGBoost')
    ax.plot(idx, tr['LSTM'],    color='red',    lw=1,   ls='--', label='LSTM')
    ax.plot(idx, tr['y_hat'],   color='black',  lw=1.5, ls='-.', label='Stacking Ensemble')
    ax.set_xlabel('Time Instance (test set)')
    ax.set_ylabel('GHI (W/m2)')
    ax.set_title(f'Actual vs Predicted GHI  h={h}h Ahead')
    ax.legend(); ax.grid(True)

    s, e   = x_range
    sz, ez = x_zm
    y_lo   = min(tr['y_true'][s:e].min(), tr['XGBoost'][s:e].min(), tr['y_hat'][s:e].min())
    y_hi   = max(tr['y_true'][s:e].max(), tr['XGBoost'][s:e].max(), tr['y_hat'][s:e].max())
    margin = (y_hi - y_lo) * 0.1

    axins = ax.inset_axes([0.55, 0.2, 0.2, 0.7])
    for arr, col, ls in [
        (tr['y_true'],  'orange', '-'),
        (tr['RF'],      'blue',   '--'),
        (tr['XGBoost'], 'green',  '--'),
        (tr['LSTM'],    'red',    '--'),
        (tr['y_hat'],   'black',  '-.'),
    ]:
        axins.plot(idx, arr, color=col, lw=1, ls=ls)
    axins.set_xlim(sz, ez)
    axins.set_ylim(y_lo - margin, y_hi + margin)
    ax.indicate_inset_zoom(axins, edgecolor='black', lw=1)

    plt.tight_layout()
    plt.savefig(f'actual_vs_predicted_h{h}.jpeg', dpi=300)
    plt.show()

for h in HORIZONS:
    plot_actual_vs_predicted(h=h)


# %%
# ─────────────────────────────────────────────────────────────────────────
# CELL 13 — Conformal Prediction Interval Plots
# ─────────────────────────────────────────────────────────────────────────

def plot_conformal_intervals(h=1, alpha=0.10, x_range=(100, 400)):
    """Visualises the symmetric conformal band around the ensemble forecast."""
    s, e   = x_range
    y_true = test_results[h]['y_true'][s:e]
    y_hat  = test_results[h]['y_hat'][s:e]
    lower  = conformal[h][alpha]['lower'][s:e]
    upper  = conformal[h][alpha]['upper'][s:e]
    idx    = np.arange(len(y_true))

    cov_pct   = conformal[h][alpha]['coverage'] * 100
    nom_pct   = (1 - alpha) * 100
    width_val = conformal[h][alpha]['avg_width']

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(idx, lower, upper, alpha=0.25, color='steelblue',
                    label=f'{nom_pct:.0f}% PI  (avg width ≈ {width_val:.1f} W/m²)')
    ax.plot(idx, y_true, color='orange',    lw=2,   label='Actual GHI')
    ax.plot(idx, y_hat,  color='steelblue', lw=1.5, ls='--', label='Ensemble Forecast')

    ax.set_xlabel('Time Instance')
    ax.set_ylabel('GHI (W/m²)')
    ax.set_title(f'Conformal PI — h={h}h | {nom_pct:.0f}% nominal, {cov_pct:.1f}% empirical coverage')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f'conformal_h{h}_alpha{int(alpha*100)}.jpeg', dpi=300)
    plt.show()

for h in HORIZONS:
    plot_conformal_intervals(h=h, alpha=0.10)
    plot_conformal_intervals(h=h, alpha=0.05)


# %%
# ─────────────────────────────────────────────────────────────────────────
# CELL 14 — Horizon Degradation Curves
# Change vs v3: 'MLP' → 'LSTM' in model_names list
# ─────────────────────────────────────────────────────────────────────────

model_names = ['RF', 'XGBoost', 'LSTM', 'Ensemble']
colors_h    = {'RF':'blue', 'XGBoost':'green', 'LSTM':'red', 'Ensemble':'black'}
markers_h   = {'RF':'s',    'XGBoost':'^',     'LSTM':'d',   'Ensemble':'o'}

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for ax, metric in zip(axes, ['MAE', 'MSE', 'RMSE', 'R2']):
    for mname in model_names:
        vals = [metrics_table[h].loc[metric, mname] for h in HORIZONS]
        ax.plot(HORIZONS, vals, label=mname,
                color=colors_h[mname], marker=markers_h[mname], lw=2, markersize=8)
    ax.set_xlabel('Forecast Horizon (hours)')
    ax.set_ylabel(metric + (' (W/m2)' if metric in ['MAE','RMSE'] else ''))
    ax.set_title(f'{metric} vs Horizon')
    ax.set_xticks(HORIZONS); ax.legend(fontsize=9); ax.grid(True)

plt.suptitle('Forecast Skill Degradation as a Function of Horizon', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('horizon_degradation_curves.jpeg', dpi=300, bbox_inches='tight')
plt.show()


# %%
# ─────────────────────────────────────────────────────────────────────────
# CELL 15 — SHAP Feature Importance: Per Physical Variable (h=1h)
# Change vs v3: 'MLP' → 'LSTM' in palette and shap_per_var columns
# ─────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
palette   = {'XGBoost':'#2ca02c', 'RF':'#1f77b4', 'LSTM':'#d62728'}

for ax, name in zip(axes, ['XGBoost', 'RF', 'LSTM']):
    vals = shap_per_var[name].values
    bars = ax.barh(FEATURES, vals, color=palette[name], alpha=0.85)
    ax.set_xlabel('Mean |SHAP| Value')
    ax.set_title(f'SHAP  {name}  (h=1h)')
    ax.grid(True, axis='x', ls='--', alpha=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                f'{v:.5f}', va='center', fontsize=8)

plt.suptitle('Input Variable Importance  Mean |SHAP| Aggregated over All Lags', fontsize=12)
plt.tight_layout()
plt.savefig('shap_variable_importance.jpeg', dpi=300)
plt.show()


# %%
# ─────────────────────────────────────────────────────────────────────────
# CELL 16 — SHAP Lag × Feature Heatmap (XGBoost, h=1h)
#
# Reveals how the importance of each physical variable changes across the
# 144-step look-back window.  Answers: which lag depths matter most?
# ─────────────────────────────────────────────────────────────────────────

# Reshape XGBoost SHAP: (N_SHAP, 720) → (N_SHAP, 144, 5)
shap_3d     = np.abs(shap_xgb).reshape(N_SHAP, PAST_STEP, N_FEAT)
lag_mean    = shap_3d.mean(axis=0)     # (144, 5) mean over samples

# Downsample lag axis for readability (every 6 steps → 24 points)
STEP = 6
lag_ds   = lag_mean[::STEP, :]          # (24, 5)
lag_ticks = list(range(0, PAST_STEP, STEP))

fig, ax = plt.subplots(figsize=(14, 4))
im = ax.imshow(lag_ds.T, aspect='auto', cmap='YlOrRd', origin='lower')
ax.set_yticks(range(N_FEAT))
ax.set_yticklabels(FEATURES)
ax.set_xticks(range(len(lag_ticks)))
ax.set_xticklabels([f'{t}' for t in lag_ticks], rotation=45, ha='right')
ax.set_xlabel(f'Lag step (0 = most recent, step size = {STEP})')
ax.set_title('XGBoost SHAP: Importance vs Lag Depth per Input Variable (h=1h)')
plt.colorbar(im, ax=ax, label='Mean |SHAP|')
plt.tight_layout()
plt.savefig('shap_lag_heatmap.jpeg', dpi=300)
plt.show()


# %%
# ─────────────────────────────────────────────────────────────────────────
# CELL 17 — Monthly Average GHI Comparison (h=1h)
# Change vs v3: 'MLP' key and label → 'LSTM'
# ─────────────────────────────────────────────────────────────────────────

def plot_monthly_comparison(h=1, year=None):
    tr     = test_results[h]
    dt_idx = data.index[tr['y_pos']]

    def to_series(arr):
        return pd.Series(arr, index=dt_idx)

    s_act = to_series(tr['y_true'])
    s_rf  = to_series(tr['RF'])
    s_xgb = to_series(tr['XGBoost'])
    s_lstm= to_series(tr['LSTM'])
    s_ens = to_series(tr['y_hat'])

    if year:
        mask  = dt_idx.year == year
        s_act  = s_act[mask];  s_rf   = s_rf[mask]
        s_xgb  = s_xgb[mask]; s_lstm  = s_lstm[mask]; s_ens = s_ens[mask]

    fig, ax = plt.subplots(figsize=(13, 5))
    for s, lbl, col, mk in [
        (s_act,  'Actual',   'orange', 'o'),
        (s_rf,   'RF',       'blue',   's'),
        (s_xgb,  'XGBoost',  'green',  '^'),
        (s_lstm, 'LSTM',     'red',    'd'),
        (s_ens,  'Ensemble', 'black',  'x'),
    ]:
        m = s.resample('ME').mean()
        ax.plot(m.index, m, marker=mk, color=col, lw=1.5, label=lbl)

    yr_str = f'  {year}' if year else '  (Full Test Period)'
    ax.set_title(f'Monthly Average GHI  h={h}h{yr_str}')
    ax.set_xlabel('Month'); ax.set_ylabel('Average GHI (W/m2)')
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig(f'monthly_ghi_h{h}.jpeg', dpi=300)
    plt.show()

plot_monthly_comparison(h=1)


# %%
# ─────────────────────────────────────────────────────────────────────────
# CELL 18 — Error Distribution (h=1h)
# Change vs v3: 'MLP' key and label → 'LSTM'
# ─────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 5))
tr = test_results[1]

for mname, col in [('RF','blue'), ('XGBoost','green'),
                   ('LSTM','red'), ('Ensemble','black')]:
    y_pred = tr['y_hat'] if mname == 'Ensemble' else tr[mname]
    sns.histplot(tr['y_true'] - y_pred, label=mname, color=col,
                 alpha=0.35, bins=60, ax=ax)

ax.axvline(0, color='k', lw=1.5, ls='--')
ax.set_xlabel('Prediction Error  (Actual - Predicted, W/m2)')
ax.set_ylabel('Frequency')
ax.set_title('Error Distribution  h=1h (Test Set)')
ax.legend(); ax.grid(True)
plt.tight_layout()
plt.savefig('error_distribution_h1.jpeg', dpi=300)
plt.show()


# %%
# ─────────────────────────────────────────────────────────────────────────
# CELL 19 — Model Performance Bar Charts (All Horizons)
# Change vs v3: 'MLP' → 'LSTM' in model_names
# ─────────────────────────────────────────────────────────────────────────

def plot_performance_bars(h=1):
    m_tab  = metrics_table[h]
    models = list(m_tab.columns)   # ['RF', 'XGBoost', 'LSTM', 'Ensemble']
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#333333']
    x      = np.arange(len(models))

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes = axes.flatten()
    legend_bars = None

    for i, met in enumerate(['MAE', 'MSE', 'RMSE', 'R2']):
        vals = [m_tab.loc[met, m] for m in models]
        bars = axes[i].bar(x, vals, 0.5, color=colors)
        axes[i].set_title(met)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(models, rotation=25, ha='right')
        axes[i].grid(True, axis='y', ls='--', alpha=0.6)
        if i == 0:
            legend_bars = bars

    fig.legend(legend_bars, models, loc='upper center',
               bbox_to_anchor=(0.5, 0.97), ncol=4, title='Models')
    plt.suptitle(f'Model Performance  GHI  h={h}h', fontsize=11, y=1.01)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(f'performance_bars_h{h}.jpeg', dpi=300, bbox_inches='tight')
    plt.show()

for h in HORIZONS:
    plot_performance_bars(h=h)


# %%
# ─────────────────────────────────────────────────────────────────────────
# CELL 20 — ACP Coverage & Width Summary Table
# Change vs v3: ACP results; added Winkler Score column
# ─────────────────────────────────────────────────────────────────────────

rows = {}
for h in HORIZONS:
    row = {}
    for alpha in ALPHA_LEVELS:
        nom = f'{int((1-alpha)*100)}%'
        acp = conformal[h][alpha]
        row[f'{nom} PICP']         = f"{acp['coverage']*100:.1f}%"
        row[f'{nom} MPIW (W/m2)'] = f"{acp['avg_width']:.2f}"
        row[f'{nom} Winkler']      = f"{acp['winkler']:.2f}"
    rows[f'h={h}h'] = row

cov_df = pd.DataFrame(rows).T
print('ACP Prediction Interval Summary (v4):')
print(cov_df.to_string())
print(f'\nACP bound: |avg_miscoverage - alpha| <= gamma/(1-gamma) = {GAMMA/(1-GAMMA):.4f}')
print('All PICP values should lie within this bound of the nominal level.')


# %%
# ═══════════════════════════════════════════════════════════════════════════════
# CELL — VMD + Per-Mode LSTM  (Direct Multi-Horizon Forecasting)
#
# Requires:  vmdpy  (auto-installed below if missing)
# Depends on: all_scaled, scaler, data, DEVICE, PAST_STEP, N_FEAT, TARGET_IDX,
#             TARGET, FEATURES, HORIZONS, TRAIN_END, CAL_END,
#             compute_metrics, inverse_transform_target
#             — all defined in earlier cells of forecasting_v4.ipynb
#
# ── Architecture ────────────────────────────────────────────────────────────
#
#   Step 1  DECOMPOSE
#     VMD splits the raw GHI series into K modes (IMFs).
#     Each mode captures a distinct frequency band:
#       Mode 1 → low-frequency seasonal trend
#       Mode 2 → daily cycle
#       Mode 3 → sub-daily variation
#       Mode 4 → cloud transients (hours)
#       Mode 5 → rapid fluctuations
#     Property: Σ_k mode_k(t) ≈ GHI(t)  (near-exact reconstruction)
#
#   Step 2  TRAIN
#     One MultiHorizonLSTM per mode.  Each LSTM receives the same
#     X = scaled_features[i:i+144] and directly predicts the mode's
#     value at ALL four horizons simultaneously:
#       output = [mode_k(+1h), mode_k(+3h), mode_k(+6h), mode_k(+24h)]
#     This is the DIRECT multi-horizon strategy — one forward pass,
#     four predictions, no iterated error compounding.
#
#   Step 3  RECONSTRUCT
#     For each horizon h:
#       GHI_pred_h = Σ_k  LSTM_k_pred_h(X)
#     Modes are z-score normalised before training (per training-set stats)
#     and denormalised before summing — ensures equal gradient scale
#     across modes of very different energy.
#
# ── Why VMD over raw LSTM ────────────────────────────────────────────────────
#   • Each LSTM specialises on ONE frequency band, reducing the learning
#     complexity per model.
#   • VMD decomposition removes the non-stationarity that causes the
#     LSTM's large h=1 error observed in the horizon degradation curves.
#   • Modal specialisation allows the direct head to focus on the correct
#     temporal scale for each horizon (low-freq modes dominate h=24,
#     high-freq modes dominate h=1).
#
# ── Comparison ───────────────────────────────────────────────────────────────
#   This cell prints a MAE comparison table against the v4 stacking ensemble
#   if metrics_table is already defined; otherwise prints standalone metrics.
# ═══════════════════════════════════════════════════════════════════════════════

import subprocess, sys, time

# ── Auto-install vmdpy if missing ─────────────────────────────────────────────
try:
    from vmdpy import VMD
    print("vmdpy already installed.")
except ImportError:
    print("Installing vmdpy ...")
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', 'vmdpy', '-q',
         '--break-system-packages'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    from vmdpy import VMD
    print("vmdpy installed successfully.")


# ══════════════════════════════════════════════════════════════════════════════
# § 1   HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

# ── VMD ───────────────────────────────────────────────────────────────────────
K_MODES   = 5       # number of modes; 5 is standard for solar GHI
                    # (trend / daily / sub-daily / cloud / rapid)
VMD_ALPHA = 2000    # bandwidth constraint — higher → narrower spectral bands
VMD_TAU   = 0.0     # noise tolerance  (0 = exact, use 0.1–0.5 for noisy data)
VMD_DC    = 0       # 0 = no explicit DC component (low-freq mode captures it)
VMD_INIT  = 1       # centre-frequency initialisation: 1 = uniformly spread
VMD_TOL   = 1e-7    # ADMM convergence tolerance

# ── Per-mode LSTM ─────────────────────────────────────────────────────────────
VMD_HIDDEN  = 128   # hidden units per LSTM; K×hidden = total capacity
VMD_LAYERS  = 2     # stacked LSTM layers
VMD_DROPOUT = 0.2   # inter-layer dropout (only applied when LAYERS > 1)
VMD_EPOCHS  = 50    # training epochs per modal LSTM
VMD_BATCH   = 512   # mini-batch size (chronological, shuffle=False)
VMD_LR      = 3e-4  # Adam learning rate; slower than default → more stable

N_HOR = len(HORIZONS)   # 4: [1, 3, 6, 24]


# ══════════════════════════════════════════════════════════════════════════════
# § 2   VMD DECOMPOSITION
#
#   Applied to the RAW (unscaled) GHI signal.
#   The feature windows X remain the same scaled arrays as in all_windows.
# ══════════════════════════════════════════════════════════════════════════════
ghi_raw = data[TARGET].values.astype(np.float64)   # (N,)  original W/m²

print("─"*70)
print(f"[1/6] VMD decomposition  K={K_MODES}  alpha={VMD_ALPHA}")
print(f"      signal length = {len(ghi_raw):,}  (may take 1–3 min)")
print("─"*70)

t0 = time.time()
u, u_hat, omega_vmd = VMD(ghi_raw, VMD_ALPHA, VMD_TAU,
                           K_MODES, VMD_DC, VMD_INIT, VMD_TOL)
# u       : (K_MODES, N)  — each row is one IMF in W/m²
# omega_vmd: (K_MODES, iterations) — centre-frequency trajectory
modes = u.astype(np.float32)   # (K, N)
elapsed = time.time() - t0
print(f"      VMD complete in {elapsed:.1f} s\n")

# Reconstruction check: Σ modes ≈ original GHI
recon     = modes.sum(axis=0)                                       # (N,)
recon_mae = float(np.abs(recon - ghi_raw).mean())
recon_r2  = float(1 - ((recon - ghi_raw)**2).sum() /
                  ((ghi_raw - ghi_raw.mean())**2).sum())
print(f"      Reconstruction  MAE={recon_mae:.4f} W/m²  R²={recon_r2:.7f}")
print(f"      (R² ≥ 0.9999 confirms near-exact decomposition)\n")

mode_colors = plt.cm.tab10(np.linspace(0, 0.9, K_MODES))
omega_final = omega_vmd[:, -1]   # converged centre frequencies

print("      Mode summary:")
for k in range(K_MODES):
    energy = float((modes[k]**2).sum() / (ghi_raw**2).sum()) * 100
    print(f"        Mode {k+1}: ω={omega_final[k]:.5f} cycles/sample  "
          f"energy={energy:.2f}%")


# ── VMD mode visualisation ────────────────────────────────────────────────────
# Show 2 representative weeks from the start of the test set
_plot_start = TRAIN_END
_plot_end   = min(_plot_start + 24 * 14, len(ghi_raw))
_t          = np.arange(_plot_end - _plot_start)

fig, axes = plt.subplots(K_MODES + 1, 1,
                          figsize=(18, 2.4 * (K_MODES + 1)),
                          sharex=True)
axes[0].plot(_t, ghi_raw[_plot_start:_plot_end], color='orange', lw=1.5)
axes[0].set_ylabel('W/m²')
axes[0].set_title('Original GHI Signal (2-week excerpt from test set)')
axes[0].grid(True, alpha=0.3)

for k in range(K_MODES):
    energy_k = float((modes[k]**2).sum() / (ghi_raw**2).sum()) * 100
    axes[k + 1].plot(_t, modes[k, _plot_start:_plot_end],
                     color=mode_colors[k], lw=1.0)
    axes[k + 1].set_ylabel(f'Mode {k+1}')
    axes[k + 1].set_title(
        f'IMF {k+1}  |  ω = {omega_final[k]:.5f} cycles/sample  '
        f'|  energy = {energy_k:.2f}%', fontsize=9)
    axes[k + 1].grid(True, alpha=0.3)

axes[-1].set_xlabel('Hours (relative to test-set start)')
plt.suptitle(f'VMD Decomposition  K={K_MODES} modes  '
             f'(Σ modes → GHI,  R²={recon_r2:.6f})', fontsize=12)
plt.tight_layout()
plt.savefig('vmd_modes.jpeg', dpi=200, bbox_inches='tight')
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# § 3   MODE NORMALISATION  (z-score, fit on training data only)
#
#   Modes span very different amplitude ranges.
#   Mode 1 (trend) may have amplitude ~400 W/m²;
#   Mode 5 (noise) may have amplitude ~10 W/m².
#   Without normalisation, the MSELoss is dominated by high-energy modes
#   and low-energy modes are undertrained.
#
#   normalised_k(t) = (mode_k(t) − μ_k) / σ_k
#   μ_k, σ_k fit on training rows ONLY — no leakage.
#   After prediction: raw_k = pred_norm_k × σ_k + μ_k
# ══════════════════════════════════════════════════════════════════════════════
mode_mu  = np.array([modes[k, :TRAIN_END].mean() for k in range(K_MODES)],
                    dtype=np.float32)
mode_sig = np.array([modes[k, :TRAIN_END].std()  for k in range(K_MODES)],
                    dtype=np.float32) + 1e-8  # +ε avoids division by zero

modes_norm = ((modes - mode_mu[:, None]) / mode_sig[:, None])   # (K, N)


# ══════════════════════════════════════════════════════════════════════════════
# § 4   WINDOW BUILDING  (direct multi-horizon)
#
#   For a window starting at row i:
#     X_i  = all_scaled[i : i+PAST_STEP,  :N_FEAT].flatten()   — shape (720,)
#             ← same scaled feature vector as in all_windows
#     y_i  = [mode_norm_k[i+PAST_STEP+h−1] for h in HORIZONS]  — shape (4,)
#             ← normalised mode value at EACH of the 4 target horizons
#
#   Split key: position of the FURTHEST target (h=24).
#   A window is assigned to train / cal / test based on where its h=24
#   target falls.  This guarantees ALL four horizon targets for a given
#   window belong to the same split — strictly leak-free.
#
#   Consequence: slightly fewer windows than per-horizon all_windows
#   because h=24 is the binding constraint everywhere.
# ══════════════════════════════════════════════════════════════════════════════
print("─"*70)
print(f"[2/6] Building multi-horizon windows  "
      f"PAST_STEP={PAST_STEP}  HORIZONS={HORIZONS}")
print("─"*70)

MAX_H = max(HORIZONS)   # 24 — the binding horizon for split assignment

def _build_windows_vmd(scaled_full, mode_norm_k, past_step, horizons):
    """
    Build supervised windows for ONE normalised VMD mode.

    Parameters
    ----------
    scaled_full  : ndarray (N, N_FEAT+1)  full scaled array from Cell 2
    mode_norm_k  : ndarray (N,)            z-scored mode values
    past_step    : int                     look-back length (144)
    horizons     : list[int]               [1, 3, 6, 24]

    Returns
    -------
    X      (n, past_step*N_FEAT) float32 — flattened feature windows
    y      (n, len(horizons))    float32 — normalised mode at each horizon
    y_pos  (n,)                  int64   — abs. row index of furthest horizon
    """
    max_h = max(horizons)
    max_i = len(scaled_full) - past_step - max_h + 1
    X_list, y_list, pos_list = [], [], []
    for i in range(max_i):
        X_list.append(scaled_full[i : i + past_step, :N_FEAT].flatten())
        y_list.append([mode_norm_k[i + past_step + h - 1] for h in horizons])
        pos_list.append(i + past_step + max_h - 1)
    return (np.array(X_list,   dtype=np.float32),
            np.array(y_list,   dtype=np.float32),
            np.array(pos_list, dtype=np.int64))


# Build windows for mode 0 first — X and y_pos are identical for all modes
X_all_vmd, _, ypos_all = _build_windows_vmd(
    all_scaled, modes_norm[0], PAST_STEP, HORIZONS
)

# Train / cal / test masks based on furthest horizon position
tr_vmd  =  ypos_all < TRAIN_END
cal_vmd = (ypos_all >= TRAIN_END) & (ypos_all < CAL_END)
te_vmd  =  ypos_all >= CAL_END

X_tr_vmd  = X_all_vmd[tr_vmd]
X_cal_vmd = X_all_vmd[cal_vmd]
X_te_vmd  = X_all_vmd[te_vmd]

print(f"  train={tr_vmd.sum():,}  cal={cal_vmd.sum():,}  test={te_vmd.sum():,}")
print(f"  feature dim = {X_tr_vmd.shape[1]}  |  targets per window = {N_HOR}")

# Build y targets for every mode
y_tr_modes  = []
y_cal_modes = []
y_te_modes  = []

for k in range(K_MODES):
    _, yk_all, _ = _build_windows_vmd(
        all_scaled, modes_norm[k], PAST_STEP, HORIZONS
    )
    y_tr_modes.append( yk_all[tr_vmd])   # (n_train, 4)
    y_cal_modes.append(yk_all[cal_vmd])  # (n_cal,   4)
    y_te_modes.append( yk_all[te_vmd])   # (n_test,  4)

# True GHI W/m² at each horizon for the test set  (for metric computation)
# We cannot reuse all_windows directly because the split boundary shifted.
y_true_ghi_te = {}
for h in HORIZONS:
    max_i    = len(all_scaled) - PAST_STEP - MAX_H + 1
    ghi_cols = np.array([
        ghi_raw[i + PAST_STEP + h - 1] for i in range(max_i)
    ], dtype=np.float32)
    y_true_ghi_te[h] = ghi_cols[te_vmd]

print(f"\n  y targets:  train={y_tr_modes[0].shape}  "
      f"cal={y_cal_modes[0].shape}  test={y_te_modes[0].shape}")


# ══════════════════════════════════════════════════════════════════════════════
# § 5   ARCHITECTURE
#
#   MultiHorizonLSTM — direct multi-step head
#   ─────────────────────────────────────────
#   Input  : (batch, 144, 5)
#   LSTM   : 2 layers, hidden=128, dropout=0.2 between layers
#   Readout: last hidden state h_{T-1}  → (batch, 128)
#   Head   : Linear(128→64) → ReLU → Dropout(0.1) → Linear(64→4)
#   Output : (batch, 4)  — one prediction per horizon [+1h, +3h, +6h, +24h]
#
#   Why 4 outputs instead of 1?
#   The standard LSTM in all_windows trains a separate model for each
#   horizon (4 models × same architecture).  Here, the multi-horizon
#   head shares the LSTM encoder and uses separate linear weights per
#   horizon — more parameter-efficient and ensures that the encoder
#   learns representations useful for ALL horizons simultaneously.
# ══════════════════════════════════════════════════════════════════════════════

class MultiHorizonLSTM(nn.Module):
    """
    LSTM encoder with a direct multi-step output head.

    Predicts N_HOR horizon values in a single forward pass — no iterated
    feedback, no error compounding.  The four output neurons share the
    LSTM encoder but learn independent linear mappings from h_{T-1}.
    """
    def __init__(self, input_size, n_horizons,
                 hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_horizons),   # one output per horizon
        )

    def forward(self, x):
        out, _ = self.lstm(x)              # (batch, 144, hidden)
        return self.head(out[:, -1, :])    # (batch, n_horizons)


class _ModalWrapper:
    """
    Thin training / inference wrapper for one modal MultiHorizonLSTM.

    .fit(X_flat, y_multi)   X_flat: (n, 720)  y_multi: (n, 4)
    .predict(X_flat)        → (n, 4) normalised mode predictions
    """
    def __init__(self, past_step, n_feat, n_horizons,
                 hidden_size, num_layers, dropout,
                 epochs, batch_size, lr):
        self.past_step   = past_step
        self.n_feat      = n_feat
        self.n_horizons  = n_horizons
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.dropout     = dropout
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.lr          = lr
        self.model       = None

    def _seq(self, X_flat):
        """Reshape flat (n, 720) → 3D tensor (n, 144, 5) on DEVICE."""
        return torch.tensor(
            X_flat.reshape(-1, self.past_step, self.n_feat),
            dtype=torch.float32,
        ).to(DEVICE)

    def fit(self, X_flat, y_multi):
        X_t = self._seq(X_flat)
        y_t = torch.tensor(y_multi, dtype=torch.float32).to(DEVICE)

        self.model = MultiHorizonLSTM(
            input_size  = self.n_feat,
            n_horizons  = self.n_horizons,
            hidden_size = self.hidden_size,
            num_layers  = self.num_layers,
            dropout     = self.dropout,
        ).to(DEVICE)

        opt    = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        crit   = nn.MSELoss()
        # patience=8: long enough to survive early-training noise,
        # short enough to exploit the plateau before FINAL_EPOCHS ends
        sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=8, factor=0.5
        )
        loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size = self.batch_size,
            shuffle    = False,   # preserve chronological order
        )

        self.model.train()
        for ep in range(self.epochs):
            total_loss = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                pred = self.model(xb)       # (batch, 4)
                loss = crit(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                total_loss += loss.item()
            sched.step(total_loss)
            if (ep + 1) % 10 == 0:
                avg    = total_loss / len(loader)
                lr_now = opt.param_groups[0]['lr']
                print(f"      Ep {ep+1:3d}/{self.epochs}  "
                      f"MSE(norm)={avg:.6f}  lr={lr_now:.2e}")
        return self

    def predict(self, X_flat):
        """Returns (n, n_horizons) normalised predictions."""
        self.model.eval()
        X_t   = self._seq(X_flat)
        parts = []
        with torch.no_grad():
            for i in range(0, len(X_t), self.batch_size):
                parts.append(
                    self.model(X_t[i : i + self.batch_size]).cpu().numpy()
                )
        return np.concatenate(parts, axis=0)   # (n, 4)


n_modal_params = sum(
    p.numel() for p in MultiHorizonLSTM(N_FEAT, N_HOR,
                                        VMD_HIDDEN, VMD_LAYERS,
                                        VMD_DROPOUT).parameters()
)
print(f"\n  MultiHorizonLSTM parameters : {n_modal_params:,}  "
      f"(× {K_MODES} modes = {K_MODES * n_modal_params:,} total)")


# ══════════════════════════════════════════════════════════════════════════════
# § 6   TRAIN ONE LSTM PER MODE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*70)
print(f"[3/6] Training {K_MODES} modal LSTMs  "
      f"hidden={VMD_HIDDEN}  epochs={VMD_EPOCHS}  "
      f"batch={VMD_BATCH}  lr={VMD_LR}")
print("─"*70)

modal_models = []
for k in range(K_MODES):
    energy_k = float((modes[k]**2).sum() / (ghi_raw**2).sum()) * 100
    print(f"\n  ── Mode {k+1}/{K_MODES}  "
          f"ω={omega_final[k]:.5f}  energy={energy_k:.2f}% ──")
    w = _ModalWrapper(
        past_step   = PAST_STEP,
        n_feat      = N_FEAT,
        n_horizons  = N_HOR,
        hidden_size = VMD_HIDDEN,
        num_layers  = VMD_LAYERS,
        dropout     = VMD_DROPOUT,
        epochs      = VMD_EPOCHS,
        batch_size  = VMD_BATCH,
        lr          = VMD_LR,
    )
    w.fit(X_tr_vmd, y_tr_modes[k])
    modal_models.append(w)
    print(f"  Mode {k+1} done.")


# ══════════════════════════════════════════════════════════════════════════════
# § 7   PREDICT + RECONSTRUCT
#
#   For each split and each modal LSTM:
#     pred_norm_k  : (n, 4)   normalised predictions
#     pred_raw_k   : (n, 4)   denormalised  = pred_norm_k × σ_k + μ_k
#
#   Reconstruction per horizon:
#     GHI_pred[:, j] = Σ_k  pred_raw_k[:, j]
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*70)
print("[4/6] Predicting all modes and reconstructing GHI ...")
print("─"*70)

def _reconstruct(X_flat, modal_models, mode_mu, mode_sig):
    """
    Run all K modal LSTMs and reconstruct GHI at every horizon.

    Returns
    -------
    ghi_pred   (n, N_HOR)   reconstructed GHI in W/m²
    raw_per_mode (K, n, N_HOR)  per-mode contribution before summing
    """
    K = len(modal_models)
    n = len(X_flat)
    raw_per_mode = np.zeros((K, n, N_HOR), dtype=np.float32)
    for k, model in enumerate(modal_models):
        pred_norm        = model.predict(X_flat)               # (n, 4) normalised
        raw_per_mode[k]  = pred_norm * mode_sig[k] + mode_mu[k]  # denormalise
    ghi_pred = raw_per_mode.sum(axis=0)                        # (n, 4) W/m²
    return ghi_pred, raw_per_mode


# Calibration
ghi_cal_pred, _cal_modes = _reconstruct(X_cal_vmd,  modal_models, mode_mu, mode_sig)

# Test
ghi_te_pred, te_modes    = _reconstruct(X_te_vmd,   modal_models, mode_mu, mode_sig)

print("  Reconstruction complete.")
print(f"  Test prediction shape: {ghi_te_pred.shape}  "
      f"(samples × horizons)")


# ══════════════════════════════════════════════════════════════════════════════
# § 8   METRICS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*70)
print("[5/6] VMD-LSTM Test Metrics  (GHI W/m²)")
print("─"*70)

vmd_metrics = {}
for j, h in enumerate(HORIZONS):
    y_true = y_true_ghi_te[h]
    y_pred = ghi_te_pred[:, j]
    mae, mse, rmse, r2 = compute_metrics(y_true, y_pred)
    vmd_metrics[h] = dict(MAE=mae, MSE=mse, RMSE=rmse, R2=r2)

vmd_df = pd.DataFrame(vmd_metrics, index=['MAE', 'MSE', 'RMSE', 'R2']).T
print(vmd_df.round(4).to_string())

# ── Comparison vs v4 stacking ensemble ────────────────────────────────────────
try:
    _ens = metrics_table   # defined in Cell 12 of forecasting_v4.ipynb
    print(f"\n{'═'*64}")
    print("  Comparison: VMD-LSTM  vs  v4 Stacking Ensemble  (MAE, W/m²)")
    print(f"{'═'*64}")
    print(f"  {'h':>5}  {'Ensemble':>12}  {'VMD-LSTM':>12}  "
          f"{'ΔMAE':>9}  {'Winner':>12}")
    print("  " + "─"*58)
    for h in HORIZONS:
        ens_mae = float(_ens[h].loc['MAE', 'Ensemble'])
        vmd_mae = vmd_metrics[h]['MAE']
        delta   = vmd_mae - ens_mae
        winner  = "VMD-LSTM ✓" if delta < 0 else "Ensemble ✓"
        print(f"  h={h:2d}h  {ens_mae:>12.2f}  {vmd_mae:>12.2f}  "
              f"{delta:>+9.2f}  {winner:>12}")
    print(f"{'═'*64}")
except NameError:
    # metrics_table not yet defined — cell run before Cell 12
    print("\n  (metrics_table not found — run Cell 12 first for comparison)")


# ══════════════════════════════════════════════════════════════════════════════
# § 9   PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*70)
print("[6/6] Generating plots ...")
print("─"*70)
N_SHOW = 500   # test steps to display in time-series plots

# ── Plot A: Forecast vs Actual — one panel per horizon ────────────────────────
fig, axes = plt.subplots(N_HOR, 1,
                          figsize=(18, 4 * N_HOR),
                          sharex=False)
for ax, (j, h) in zip(axes, enumerate(HORIZONS)):
    idx    = np.arange(N_SHOW)
    y_true = y_true_ghi_te[h][:N_SHOW]
    y_pred = ghi_te_pred[:N_SHOW, j]
    mae_h  = float(np.abs(y_true - y_pred).mean())

    ax.plot(idx, y_true, color='orange', lw=2,   label='Actual GHI')
    ax.plot(idx, y_pred, color='black',  lw=1.2, ls='--',
            label=f'VMD-LSTM  MAE={mae_h:.1f} W/m²')

    # Shade the per-mode stacked contribution
    cum = np.zeros(N_SHOW)
    for k in range(K_MODES):
        contrib = te_modes[k, :N_SHOW, j]
        ax.fill_between(idx, cum, cum + contrib,
                        alpha=0.20, color=mode_colors[k], lw=0)
        cum += contrib

    ax.set_ylabel('GHI (W/m²)')
    ax.set_title(f'VMD-LSTM  h={h}h  —  '
                 f'Σ_{{{K_MODES}}} modal LSTMs  '
                 f'(shaded = per-mode contribution)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.35)

axes[-1].set_xlabel(f'Test instance  (first {N_SHOW} steps)')
plt.suptitle(
    f'VMD-LSTM Direct Multi-Horizon Forecast  |  K={K_MODES} modes',
    fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig('vmd_lstm_forecast_all_horizons.jpeg', dpi=200,
            bbox_inches='tight')
plt.show()
print("  Saved: vmd_lstm_forecast_all_horizons.jpeg")

# ── Plot B: Modal contribution breakdown — h=1h ───────────────────────────────
j_h1    = HORIZONS.index(1)
N_CONT  = 200

fig, axes_b = plt.subplots(2, 1, figsize=(16, 10))

# Top: stacked contribution chart
cum = np.zeros(N_CONT)
for k in range(K_MODES):
    contrib = te_modes[k, :N_CONT, j_h1]
    energy_k = float((modes[k]**2).sum() / (ghi_raw**2).sum()) * 100
    axes_b[0].fill_between(
        np.arange(N_CONT), cum, cum + contrib,
        alpha=0.65, color=mode_colors[k],
        label=f'Mode {k+1}  ω={omega_final[k]:.4f}  ({energy_k:.1f}%)'
    )
    cum += contrib

axes_b[0].plot(np.arange(N_CONT), y_true_ghi_te[1][:N_CONT],
               color='orange', lw=2.5, label='Actual GHI', zorder=5)
axes_b[0].plot(np.arange(N_CONT), ghi_te_pred[:N_CONT, j_h1],
               color='black',  lw=1.5, ls='--',
               label='Σ modes (VMD-LSTM h=1h)', zorder=5)
axes_b[0].set_ylabel('GHI (W/m²)')
axes_b[0].set_title(f'Stacked Modal Contributions to h=1h Forecast  '
                    f'(first {N_CONT} test steps)')
axes_b[0].legend(fontsize=8, loc='upper right', ncol=2)
axes_b[0].grid(True, alpha=0.3)

# Bottom: horizon MAE bar chart
maes  = [vmd_metrics[h]['MAE'] for h in HORIZONS]
xpos  = np.arange(N_HOR)
bars  = axes_b[1].bar(xpos, maes, color=[mode_colors[i % K_MODES]
                                          for i in range(N_HOR)],
                       alpha=0.85, width=0.5)
axes_b[1].set_xticks(xpos)
axes_b[1].set_xticklabels([f'h={h}h' for h in HORIZONS])
axes_b[1].set_ylabel('MAE (W/m²)')
axes_b[1].set_title('VMD-LSTM MAE per Forecast Horizon')
for bar, mae_v in zip(bars, maes):
    axes_b[1].text(bar.get_x() + bar.get_width() / 2,
                   bar.get_height() + 0.3,
                   f'{mae_v:.2f}', ha='center', va='bottom', fontsize=10)
axes_b[1].grid(True, axis='y', alpha=0.4)

plt.suptitle(f'VMD-LSTM: K={K_MODES} Modes × 4 Horizons  '
             f'(Direct Multi-Horizon Strategy)', fontsize=12)
plt.tight_layout()
plt.savefig('vmd_modal_contributions_h1.jpeg', dpi=200, bbox_inches='tight')
plt.show()
print("  Saved: vmd_modal_contributions_h1.jpeg")


# ══════════════════════════════════════════════════════════════════════════════
# § 10  SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*70}")
print(f"  VMD-LSTM complete")
print(f"  K={K_MODES} modes  ×  {K_MODES} LSTMs  ×  {N_HOR} horizons ({HORIZONS})")
print(f"  Parameters per model : {n_modal_params:,}  "
      f"(total : {K_MODES * n_modal_params:,})")
print(f"  Outputs saved:")
print(f"    ghi_te_pred      — shape {ghi_te_pred.shape}  (W/m², test set)")
print(f"    te_modes         — shape {te_modes.shape}  (per-mode contributions)")
print(f"    vmd_metrics      — dict[horizon → MAE/MSE/RMSE/R²]")
print(f"    modal_models     — list of {K_MODES} _ModalWrapper objects")
print(f"{'═'*70}")




