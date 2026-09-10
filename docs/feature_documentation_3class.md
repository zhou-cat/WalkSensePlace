# Feature documentation — merged 3-class model (v16)

Every feature is computed **per step** (one gait cycle, from one total-pressure peak to the
next, ~0.5–1.2 s at 62.5 Hz). Sensors used: 12 plantar pressure pads, a 3-axis accelerometer,
and a 3-axis gyroscope (the magnetometer is dead — all zeros — so it is unused). Steps that
span a surface change or a `transition`/`wait` tag are excluded, so each step lies on a single
surface.

Two conventions appear throughout:
- **"norm" / session-normalised** — divided by a within-recording scale (e.g. that step's mean
  total pressure) so the value doesn't scale with body weight or how tightly the insole was worn.
- **amplitude-invariant** — the metric is a *ratio* or *shape* number, unaffected by overall size.

The model uses **48 features**. The person-confound features the pipeline computes but
**drops** from the model (per-sensor pressure ratios and raw magnitudes) are listed at the end.

Below, the "F" values are the ANOVA class-separation scores on the full data — higher means the
feature separates grass/loose/paved better. The features are grouped by what they measure.

---

## 1. Foot-tilt variability (the strongest features)

These come from a ZUPT-gated orientation estimate: within each step, the instant of minimum
gyro magnitude (foot flat and momentarily still) is found, and the foot's tilt at that instant
is read from the accelerometer (which then points along gravity). Two orthogonal tilt axes, A
and B, are tracked.

- **orient_rest_a_roll3_std** (F≈1612) and **orient_rest_b_roll3_std** (F≈889) — the
  **step-to-step variability** of resting foot tilt, computed as the rolling 3-step standard
  deviation of the flat-foot tilt within each recording+foot sequence. High values mean the foot
  lands at a *different* orientation each step — the signature of an uneven/compliant surface.
  These are the single most class-differentiable features in the model and generalise across
  people. (The absolute resting tilt `orient_rest_a/b` is computed but **not** used as a feature —
  it's mostly a per-person sensor-mount offset; only its step-to-step *variability* is kept.)
- **orient_range_a / orient_range_b** — the range (max − min) of foot tilt *during a single
  stance* on each axis: how much the foot rocks while planted. A within-step complement to the
  step-to-step variability above.

## 2. Accelerometer texture / impact (vibration and hardness)

Computed on the accelerometer signal within the step. The spectral features run an FFT on the
detrended signal; the "jerk"/"zcr" features run on the vertical (accel-Y) axis.

- **accel_jerk_ratio** (F≈1205) — RMS of the first difference of vertical acceleration divided by
  the RMS of the signal itself (amplitude-invariant). Captures how "jerky"/high-frequency the
  strike is — hard surfaces transmit sharper vibration than compliant ones.
- **accel_hf_frac** (F≈549) — fraction of accelerometer spectral power above 5 Hz. Higher on
  firm/rigid surfaces.
- **accel_spec_centroid** (F≈517) — the spectral centroid (centre-of-mass frequency) of the
  accelerometer power spectrum. Shifts up with harder, higher-frequency impacts.
- **accel_spec_entropy** (F≈492) — normalised Shannon entropy of the accelerometer power
  spectrum: tonal/peaky spectra (low entropy) vs broadband (high). A texture descriptor.
- **accel_zcr** — zero-crossing rate of the vertical acceleration (how often it changes sign): a
  simple roughness/oscillation measure.
- **accel_mag_strike** — peak accelerometer magnitude in the impact region (first third of the
  step): the strike intensity.

## 3. Gyroscope motion (foot rotation & stability)

- **gyro_x_std** (F≈1046) — standard deviation of the X-axis angular rate during the step:
  side-to-side foot rotation/wobble through stance. A strong discriminator.
- **gyro_x_strike / gyro_y_strike / gyro_z_strike** — angular rate on each axis at the strike
  instant: the rotational "kick" at foot contact.
- **gyro_y_std / gyro_z_std** — angular-rate variability on the other two axes.
- **gyro_hf_frac** (F≈397) — fraction of gyroscope spectral power above 5 Hz (rotational
  high-frequency content, analogous to accel_hf_frac).
- **gyro_mag_strike** — gyro magnitude at the strike instant.

## 4. Stance timing & pressure-curve shape

Derived from the total-pressure curve over the step, mostly session-normalised so they describe
*shape* not size.

- **stance_duration** (F≈838) — time (or samples) the foot is loaded during the step. Compliant
  ground tends to lengthen/soften contact.
- **loading_rate / loading_rate_norm** — how fast pressure rises to peak (raw and
  session-normalised). Hard surfaces load faster.
- **rise_slope / fall_slope** — slopes of the pressure rise and fall.
- **slope_asym** — asymmetry between the loading (rise) and unloading (fall) slopes.
- **time_to_peak_frac** — fraction of the step elapsed when total pressure peaks (timing of peak
  loading within the cycle).
- **gu_dip_depth** — depth of the mid-stance dip in the (amplitude-normalised) total-pressure
  curve: `1 − min(midstance)/max`. Grass shows a deeper mid-stance dip.
- **gu_load_first_frac** — fraction of total loading that occurs in the first half of the step.
- **gu_push_peak** — height of the push-off (second) peak in the normalised curve.
- **gu_load_slope / gu_unload_slope** — max rising / min falling slope of the normalised total
  curve (ground-up complements to the accumulated slope features).

## 5. Weight distribution (heel vs forefoot)

- **heel_fore_ratio** (F≈372) — mean heel pressure ÷ mean forefoot pressure over the step: where
  load sits front-to-back.
- **heel_impulse_norm** (F≈394) — heel pressure integrated over the step, session-normalised
  (time-under-load at the heel).
- **heel_skew / heel_kurt** — skewness and kurtosis of the heel-pressure time series (shape of the
  heel loading profile).
- **heel_ripple_frac** — fraction of high-frequency "ripple" in the heel signal.
- **midstance_diff** — heel-minus-forefoot balance at the mid-stance (flat-foot) instant.
- **gu_fore_early_frac** — share of forefoot loading in the early stance phase.
- **gu_fore_heel_toff** — timing offset between the forefoot and heel pressure peaks.

## 6. Pressure-map distribution & dynamics (spatial, across the 12 pads)

Treats the 12 sensors as a spatial pressure map.

- **gu_spatial_entropy_stance** (F≈510) and **gu_spatial_entropy_ff** — normalised Shannon
  entropy of the pressure distribution across the 12 pads (over the whole stance, and at the
  flat-foot instant). High = load spread evenly across the sole; low = concentrated on a few
  pads. A compliant/conforming surface spreads load more.
- **gu_participation_ff** — effective fraction of pads bearing load at flat-foot (inverse
  participation ratio), a complementary spread measure.
- **gu_active_sensors** — count of pads loaded above 10% of the peak pad at flat-foot: contact
  breadth.
- **gu_cop_wander** — mean frame-to-frame change of the normalised pressure map through stance:
  how much the load *shifts around* the sole (a dynamic unevenness proxy).
- **gu_press_jitter** — mean per-pad temporal jitter (std of the 2nd difference) of pressure,
  session-normalised: small pressure fluctuations from surface irregularity.
- **stance_duration_roll3_std / loading_rate_roll3_std / impact_mag_roll3_std** — step-to-step
  variability (rolling 3-step std) of stance duration, loading rate, and impact: gait-consistency
  measures that rise on irregular ground.

---

## Dropped (computed but excluded from the model)

These were removed because cross-user permutation testing showed they act as a **person
fingerprint** (they encode who is walking or how the insole is seated, not the surface), and they
*hurt* generalisation to a new person:

- **Per-sensor pressure ratios** — `pressure_01_mean_ratio` … `pressure_12_mean_ratio`, plus
  `heel_mean_ratio`, `fore_mean_ratio`, `heel_pmax_ratio`, `fore_pmax_ratio`.
- **Raw magnitudes** — `total_p_mean`, `total_pmax`, `impact_mag`, `impact_mag_norm`,
  `heel_impulse` (these scale with body weight / donning tightness).

---

## How to read the model, in one line

The features that carry the surface signal — and survive to a new person — are **step-to-step
foot-tilt variability**, **accelerometer texture/jerk**, **gyro rotation variability**, **stance
timing**, and **pressure-map spread**. Together these describe *compliance* (soft vs firm) very
well, which is why grass-vs-hard is reliable; they describe *fine firm-surface texture* weakly,
which is why loose-vs-paved remains hard.
