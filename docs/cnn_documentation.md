# CNN documentation — 1D temporal CNN on raw step windows

The tree models (XGBoost/RF) use 48 hand-engineered per-step features. The **1D-CNN does not** —
it is fed the **raw multichannel waveform of each step** and learns its own shape/texture
detectors. So there is no feature list to document; instead this covers (1) the input channels,
(2) how each step window is prepared, (3) the network architecture and what it learns, and (4) how
it's trained and evaluated. Applies to both notebooks — only the number of output classes differs
(3 for grass/loose/paved, 2 for soft/hard).

---

## 1. Inputs — 9 channels × 64 time-steps

Each step is represented as a `[64 × 9]` array: 9 signal channels sampled at 64 time-points across
the gait cycle. The 9 channels (`SEQ_CHANNELS`) are:

| # | channel | what it is | how derived |
|---|---------|-----------|-------------|
| 1 | `heel_sum` | total heel pressure | sum of the 2 heel pads (`pressure_08`, `pressure_11`) |
| 2 | `fore_sum` | total forefoot pressure | sum of the 5 forefoot pads (`pressure_02/04/05/06/09`) |
| 3 | `total_p` | total plantar pressure | sum of all 12 pressure pads |
| 4–6 | `accel_x/y/z` | tri-axial acceleration | raw accelerometer, per axis |
| 7–9 | `gyro_x/y/z` | tri-axial angular rate | raw gyroscope, per axis |

The magnetometer is dead (all-zeros) and is not used. Note the CNN is given the **three pressure
aggregates** (heel / forefoot / total) rather than all 12 pads individually — this keeps the input
compact and focuses the network on the front-to-back loading pattern that carries the surface
signal, while the accel/gyro axes give it the raw vibration and rotation to find texture in.

## 2. Window preparation

- **Step segmentation** — the same `find_step_windows` used by the tree pipeline detects gait
  cycles (heel-strike to heel-strike); each window is one step. Windows shorter than `MIN_DP`
  samples are dropped. A window's label is the majority surface tag within it (single-surface
  steps only; `transition`/`wait` regions are excluded upstream).
- **Resample to a fixed length** — steps vary in duration (~0.5–1.2 s), so each channel is linearly
  resampled to exactly **T_LEN = 64** time-points (`_resample_zscore`). This time-normalises the
  gait cycle so a fast and a slow step are comparable.
- **Per-channel z-scoring** — each channel of each step is standardised to mean 0, std 1
  *individually*. This removes magnitude and keeps **shape** only — the same session/person
  invariance principle behind the tree model's normalised features. The CNN therefore sees *how*
  pressure and motion evolve through the step, not how large they are, so it doesn't key on body
  weight or how tightly the insole was worn.

## 3. Architecture — what the network learns

Input `[batch, 64, 9]` is transposed to `[batch, 9 channels, 64 time]` and passed through:

**Convolutional body** (learns temporal pattern detectors):
- `Conv1d(9→32, kernel 7, pad 3)` → BatchNorm → ReLU → `MaxPool1d(2)` — 32 filters, each a
  learned 7-sample temporal pattern across all 9 channels (e.g. a sharp impact spike, a loading
  ramp); pooling halves the time axis (64→32) and adds small shift-tolerance.
- `Conv1d(32→64, kernel 5, pad 2)` → BatchNorm → ReLU → `MaxPool1d(2)` — 64 mid-level filters
  combining the low-level patterns (32→16 time-steps).
- `Conv1d(64→64, kernel 3, pad 1)` → BatchNorm → ReLU — 64 higher-level filters.
- `AdaptiveAvgPool1d(1)` → `Flatten` — **global average pooling** over time collapses each of the
  64 filter responses to a single number: a 64-dim learned summary of the step.

**Classifier head**:
- `Dropout(0.3)` → `Linear(64→64)` → ReLU → `Dropout(0.3)` → `Linear(64→n_classes)`.

So the CNN's "features" are the **64 learned convolutional filters** — the analog of the
hand-crafted features, but discovered from data. Conceptually they can capture the same physical
things the engineered features target: impact sharpness (like `accel_jerk_ratio`), loading-curve
shape (like `gu_dip_depth`/`loading_rate`), and rotational texture (like `gyro_x_std`) — just
learned as temporal filters rather than computed formulas. BatchNorm stabilises training; the two
dropout layers (0.3) fight overfitting given the modest dataset.

## 4. Training & evaluation

- **Loss/optimiser** — cross-entropy with **softened class weights** (the same
  `CLASS_WEIGHT_STRENGTH` dial as the trees, so the minority class is up-weighted consistently);
  Adam, lr 1e-3, weight decay 1e-4.
- **Early stopping** — trains up to 80 epochs, keeping the weights with the best *macro-F1 on a
  grouped internal validation split* (whole held-out recordings), patience 12. `drop_last` avoids a
  size-1 final batch that would break BatchNorm.
- **Cross-validation** — `StratifiedGroupKFold` grouped by recording/user, **identical to the tree
  models**, so the CNN's out-of-fold scores are directly comparable. Always compare against the
  **grouped** (not shuffled) XGBoost focus-class F1.

## 5. Two honest limitations (why the CNN often ties or loses to the merged trees)

1. **It cannot see cross-step features.** Each window is an independent, z-scored step, so the
   CNN structurally has **no access to step-to-step variability** — yet the tree model's single
   strongest feature is exactly that (`orient_rest_a_roll3_std`, the rolling std of foot tilt
   across consecutive steps). A CNN over single steps can't reconstruct a quantity defined across
   steps, so it is missing the most discriminating signal the trees rely on.
2. **Data size.** With a few thousand steps and the minority class present in only a handful of
   recordings, a CNN overfits easily. If its grouped focus-class F1 doesn't beat XGBoost, that's
   the honest result, not a tuning failure — and given limitation (1), a tie or slight loss is the
   expected outcome, not a surprise.

## 6. One-line reading

The CNN learns temporal filters over the raw heel/forefoot/total-pressure and accel/gyro
waveforms of each step (time-normalised and z-scored, so it reads *shape*), and classifies from a
global-average summary of those filters. It captures the same compliance/texture information the
engineered features do, but — seeing each step in isolation — it misses the cross-step
variability that is the merged tree model's strongest cue, which is why the trees remain the
model to beat.
