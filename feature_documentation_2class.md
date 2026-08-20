# Feature documentation — merged 2-class model (v16, soft vs hard)

Same feature set as the 3-class model — the only difference is the **target**: `soft` (grass /
compliant ground) vs `hard` (dirt, compact ground, concrete, asphalt, exposed aggregate, brick).
Every feature is computed **per step** (one gait cycle, ~0.5–1.2 s at 62.5 Hz) from the 12 plantar
pressure pads, the 3-axis accelerometer, and the 3-axis gyroscope (the magnetometer is dead and
unused). Steps spanning a surface change or a `transition`/`wait` tag are excluded.

Conventions:
- **"norm" / session-normalised** — divided by a within-recording scale (e.g. that step's mean
  total pressure) so values don't scale with body weight or insole tightness.
- **amplitude-invariant** — a ratio/shape number, unaffected by overall magnitude.

The model uses **48 features**; the person-confound features it computes but drops are listed at
the end.

**Why the soft/hard split is the reliable product:** these features primarily encode **surface
compliance** (how much the ground gives under load). On the soft-vs-hard target the top features
separate the classes about twice as sharply as on the 3-class target — e.g. the strongest,
`orient_rest_a_roll3_std`, scores F≈3109 here vs ≈1612 in the 3-class model. The features that
would be needed to split the *hard* class further (firm natural ground vs pavement) are weak,
which is why the 3-class `loose` label doesn't generalise but soft-vs-hard does.

Below, "F" is the ANOVA soft-vs-hard separation score on the full data (higher = better).

---

## 1. Foot-tilt variability (the strongest features)

From a ZUPT-gated orientation estimate: within each step, the instant of minimum gyro magnitude
(foot flat and momentarily still) is found, and the foot's tilt there is read from the
accelerometer (pointing along gravity). Two orthogonal tilt axes, A and B.

- **orient_rest_a_roll3_std** (F≈3109) and **orient_rest_b_roll3_std** (F≈1710) — **step-to-step
  variability** of resting foot tilt (rolling 3-step std of flat-foot tilt within each
  recording+foot sequence). High = the foot lands at a different orientation each step, the
  signature of soft/compliant ground. The single strongest discriminators, and they generalise
  across people. (The absolute resting tilt `orient_rest_a/b` is a per-person mount offset and is
  **not** used as a feature — only its step-to-step variability.)
- **orient_range_a / orient_range_b** — range (max − min) of foot tilt *during a single stance*:
  how much the foot rocks while planted (within-step complement to the above).

## 2. Accelerometer texture / impact (vibration & hardness)

Computed on the accelerometer within the step; spectral features use an FFT of the detrended
signal, jerk/zcr use the vertical (accel-Y) axis.

- **accel_jerk_ratio** (F≈2311) — RMS of the first difference of vertical acceleration ÷ RMS of
  the signal (amplitude-invariant). Sharp/jerky strikes → hard; damped → soft.
- **accel_hf_frac** (F≈819) — fraction of accelerometer spectral power above 5 Hz. Higher on firm
  surfaces.
- **accel_spec_centroid** (F≈693) — centre-of-mass frequency of the accelerometer power spectrum;
  shifts up with harder impacts.
- **accel_spec_entropy** — normalised spectral entropy (tonal vs broadband) of the acceleration.
- **accel_zcr** — zero-crossing rate of vertical acceleration (roughness/oscillation).
- **accel_mag_strike** — peak accelerometer magnitude in the impact region (first third of step).

## 3. Gyroscope motion (foot rotation & stability)

- **gyro_x_std** (F≈1507) — std of X-axis angular rate through the step: side-to-side foot
  rotation/wobble. Strong discriminator.
- **gyro_x_strike** (F≈946) / **gyro_y_strike / gyro_z_strike** — angular rate at the strike
  instant (rotational "kick" at contact).
- **gyro_y_std / gyro_z_std** — angular-rate variability on the other axes.
- **gyro_hf_frac** — fraction of gyroscope spectral power above 5 Hz.
- **gyro_mag_strike** — gyro magnitude at strike.

## 4. Stance timing & pressure-curve shape

From the total-pressure curve, mostly session-normalised (shape, not size).

- **stance_duration** (F≈1352) — loaded time per step; compliant ground softens/lengthens contact.
- **loading_rate / loading_rate_norm** — how fast pressure rises to peak (raw and normalised);
  hard loads faster.
- **rise_slope / fall_slope** — pressure rise and fall slopes.
- **slope_asym** — asymmetry between loading and unloading slopes.
- **time_to_peak_frac** — fraction of the step elapsed at peak loading.
- **gu_dip_depth** — depth of the mid-stance dip in the normalised total-pressure curve
  (`1 − min(midstance)/max`); deeper on soft ground.
- **gu_load_first_frac** — share of loading in the first half of the step.
- **gu_push_peak** — height of the push-off peak (normalised).
- **gu_load_slope / gu_unload_slope** (unload F≈668) — max rising / min falling slope of the
  normalised total curve.

## 5. Weight distribution (heel vs forefoot)

- **heel_fore_ratio** (F≈665) — mean heel ÷ mean forefoot pressure (front-to-back load).
- **heel_impulse_norm** (F≈695) — heel pressure integrated over the step, session-normalised.
- **heel_skew / heel_kurt** — skewness/kurtosis of the heel-pressure time series.
- **heel_ripple_frac** — high-frequency "ripple" fraction in the heel signal.
- **midstance_diff** — heel-minus-forefoot balance at the flat-foot instant.
- **gu_fore_early_frac** — forefoot loading share in early stance.
- **gu_fore_heel_toff** — timing offset between forefoot and heel pressure peaks.

## 6. Pressure-map distribution & dynamics (across the 12 pads)

- **gu_spatial_entropy_stance** (F≈976) and **gu_spatial_entropy_ff** — normalised Shannon entropy
  of the pressure distribution across the 12 pads (whole stance, and at flat-foot). High = load
  spread evenly (compliant ground conforms); low = concentrated.
- **gu_participation_ff** — effective fraction of pads loaded at flat-foot (inverse participation
  ratio).
- **gu_active_sensors** — count of pads above 10% of the peak pad at flat-foot (contact breadth).
- **gu_cop_wander** — mean frame-to-frame change of the normalised pressure map through stance
  (dynamic load shifting).
- **gu_press_jitter** — mean per-pad temporal jitter (std of 2nd difference), session-normalised.
- **stance_duration_roll3_std / loading_rate_roll3_std / impact_mag_roll3_std** — step-to-step
  variability (rolling 3-step std) of stance duration, loading rate, and impact.

---

## Dropped (computed but excluded from the model)

Removed because cross-user permutation testing showed they act as a **person fingerprint** and
hurt generalisation to a new person:

- **Per-sensor pressure ratios** — `pressure_01_mean_ratio` … `pressure_12_mean_ratio`,
  `heel_mean_ratio`, `fore_mean_ratio`, `heel_pmax_ratio`, `fore_pmax_ratio`.
- **Raw magnitudes** — `total_p_mean`, `total_pmax`, `impact_mag`, `impact_mag_norm`,
  `heel_impulse` (scale with body weight / donning).

---

## One-line reading

Soft-vs-hard is driven by **step-to-step foot-tilt variability**, **accelerometer jerk/texture**,
**gyro rotation variability**, **stance timing**, and **pressure-map spread** — all direct
measures of surface *compliance*. Because compliance is exactly what a pressure/IMU insole senses
well, this two-class model is the reliable, deployable product.
