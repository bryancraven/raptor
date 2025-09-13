#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Hazard-Based Model for Estimating the Likelihood of Seeing Birds of Prey
on the Boise Greenbelt (≥1 raptor observed during a walk)

Author: You (with assist)
Date: 2025-09-13

--------------------------------------------------------------------
WHAT'S NEW (beyond the earlier "improved" version you saw):
--------------------------------------------------------------------
A) Core modeling upgrades
   1) Hazard / rate-scale modeling (Poisson) retained, but now enriched with:
      - Species-specific *time-of-day sensitivity* (diurnal vs. nocturnal vs. soaring).
      - Species-specific *weather sensitivity* (thermals matter more for soaring hawks).
      - *Habitat-route weighting*: factor in how much of your walk hugs the river vs. parkland,
        woodland, or urban segments (with species habitat preferences).
      - *Seasonality by month*: smooth, circular (month-on-a-wheel) Gaussian peaks per species,
        normalized so that August ~ baseline (because priors are August-centric).
      - *Party-size effect*: detection improves with more observers, but saturates.

   2) Shared-day correlation modeled via LogNormal multiplier on the rate scale with E[Q]=1,
      avoiding truncation bias and maintaining positivity.

   3) Duration scaling with *diminishing returns* using an exposure exponent gamma (default 0.95),
      i.e., effective exposure E = (duration/base_hours)^gamma.

B) Bayesian updating from your own checklists (Beta-Binomial)
   - Same mechanism as before: update per-species Beta priors with your "seen/total" inputs.

C) Explainability & outputs
   - Overall probability and credible interval (configurable `--ci`).
   - Per-species *marginal probabilities* (P(≥1 of species i)).
   - *Contribution shares* to P(any), using Poisson superposition: share_i ≈ E[P_any * λ_i / Σλ] / E[P_any].
   - Optional plots: distribution histogram and a contribution bar chart.
   - Optional JSON or CSV report files.
   - `--list_species` quickly lists the included species.
   - `--explain_per_species` prints a per-species multiplier breakdown (time/weather/habitat/season).

D) Reproducibility, ergonomics, and performance
   - Vectorized NumPy Monte Carlo; `--seed` for reproducibility.
   - Clean CLI with validation and helpful error messages.

--------------------------------------------------------------------
ASSUMPTIONS & SCOPE
--------------------------------------------------------------------
- The per-species prior means represent the chance of ≥1 observation during a ~1-hour walk
  in August on typical Boise River Greenbelt routes.
- We convert those August probabilities to *base hourly rates* via λ = -ln(1 - p).
- The observed process is “at least one sighting,” not counts. Poisson superposition gives
  P(any) = 1 - exp(-Σ λ_i'), where λ_i' include all scenario multipliers.
- Weather/time/habitat/season/party-size effects act on the *rate scale* (multipliers on λ).
- Seasonality is expressed relatively: every species’ seasonal factor is normalized to equal 1.0 in August.
  This keeps your August priors as the on-average baseline and scales other months up/down.

DISCLAIMER
- Priors are approximate and intended for educational, exploratory modeling—real frequencies vary by
  exact location, observer skill, and conditions. Integrating eBird and live weather APIs would
  be the next production step.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# CONFIGURABLE CONSTANTS
# ----------------------------

BASE_DURATION_HRS: float = 1.0      # Priors reflect about a 1-hour walk in August.
DEFAULT_K: float = 50.0             # Beta prior strength (higher = narrower Beta around mean).
DEFAULT_CI: float = 0.90            # Credible interval level for summaries.
AUGUST_MONTH: int = 8               # Month used to normalize seasonality factors.

# Default Greenbelt-like route composition (rough guess; sums to 1.0).
BASELINE_ROUTE: Dict[str, float] = {
    "river": 0.6,
    "park": 0.3,
    "woodland": 0.1,
    "urban": 0.0,
}

# Supported habitat keys order (for printing and robust parsing).
HABITATS: Tuple[str, ...] = ("river", "park", "woodland", "urban")


# ----------------------------
# SPECIES PRIORS (AUGUST)
# ----------------------------
# Means are the chance of ≥1 observation in about 1 hour during August.
# Names kept ASCII-safe for portability.
SPECIES_MEANS_AUGUST: Dict[str, float] = {
    "Osprey": 0.30,
    "American Kestrel": 0.18,
    "Swainson's Hawk": 0.12,
    "Red-tailed Hawk": 0.08,
    "Cooper's Hawk": 0.05,
    "Peregrine Falcon": 0.03,
    "Bald Eagle": 0.02,
    "Great Horned Owl": 0.02,
    "Western Screech-Owl": 0.01,
}

# Group tags for species-specific behavior.
# - "soarer": thermals matter; midday penalties are minimal; river/habitat can matter (Osprey).
# - "falcon": active much of the day; thermals less central than for soaring hawks.
# - "accipiter": woodland/edges hunters; time-of-day boost at dawn/dusk.
# - "owl": nocturnal; day penalties are strong; night boosts are strong.
SPECIES_GROUP: Dict[str, str] = {
    "Osprey": "soarer",
    "American Kestrel": "falcon",
    "Swainson's Hawk": "soarer",
    "Red-tailed Hawk": "soarer",
    "Cooper's Hawk": "accipiter",
    "Peregrine Falcon": "falcon",
    "Bald Eagle": "soarer",
    "Great Horned Owl": "owl",
    "Western Screech-Owl": "owl",
}

# Species habitat preferences (weights sum to 1.0). These are approximate.
SPECIES_HABITAT_WEIGHTS: Dict[str, Dict[str, float]] = {
    "Osprey":               {"river": 0.85, "park": 0.05, "woodland": 0.05, "urban": 0.05},
    "American Kestrel":    {"river": 0.15, "park": 0.50, "woodland": 0.15, "urban": 0.20},
    "Swainson's Hawk":     {"river": 0.20, "park": 0.50, "woodland": 0.20, "urban": 0.10},
    "Red-tailed Hawk":     {"river": 0.20, "park": 0.40, "woodland": 0.30, "urban": 0.10},
    "Cooper's Hawk":       {"river": 0.15, "park": 0.15, "woodland": 0.50, "urban": 0.20},
    "Peregrine Falcon":    {"river": 0.25, "park": 0.20, "woodland": 0.15, "urban": 0.40},
    "Bald Eagle":          {"river": 0.70, "park": 0.10, "woodland": 0.15, "urban": 0.05},
    "Great Horned Owl":    {"river": 0.05, "park": 0.20, "woodland": 0.60, "urban": 0.15},
    "Western Screech-Owl": {"river": 0.10, "park": 0.15, "woodland": 0.70, "urban": 0.05},
}

# Species seasonal parameters (circular Gaussian).
# Each entry defines the *raw* peak shape; it will be normalized to equal 1.0 in August.
# Params: peak_month (1..12), sigma_months (>0), min_factor (>0), max_factor (>0).
SPECIES_SEASONALITY: Dict[str, Dict[str, float]] = {
    "Osprey":               {"peak_month": 8, "sigma": 1.5, "min_factor": 0.3, "max_factor": 1.3},
    "American Kestrel":     {"peak_month": 7, "sigma": 6.0, "min_factor": 0.9, "max_factor": 1.1},
    "Swainson's Hawk":      {"peak_month": 8, "sigma": 1.5, "min_factor": 0.2, "max_factor": 1.4},
    "Red-tailed Hawk":      {"peak_month": 6, "sigma": 6.0, "min_factor": 0.9, "max_factor": 1.1},
    "Cooper's Hawk":        {"peak_month": 6, "sigma": 6.0, "min_factor": 0.9, "max_factor": 1.1},
    "Peregrine Falcon":     {"peak_month": 9, "sigma": 6.0, "min_factor": 0.9, "max_factor": 1.1},
    "Bald Eagle":           {"peak_month": 1, "sigma": 2.0, "min_factor": 0.4, "max_factor": 1.5},
    "Great Horned Owl":     {"peak_month": 5, "sigma": 6.0, "min_factor": 0.95, "max_factor": 1.05},
    "Western Screech-Owl":  {"peak_month": 5, "sigma": 6.0, "min_factor": 0.95, "max_factor": 1.05},
}

# Weather elasticity (species group sensitivity). Factor W_base ** elasticity.
WEATHER_ELASTICITY_BY_GROUP: Dict[str, float] = {
    "soarer": 1.15,
    "falcon": 1.00,
    "accipiter": 0.90,
    "owl": 0.80,
}

# Time-of-day species group modifiers applied on top of a baseline time factor.
# These are gentle multipliers (not exponents); they are designed to keep overall scales reasonable.
TIME_GROUP_MODIFIER: Dict[str, Dict[str, float]] = {
    "soarer":   {"dawn": 1.10, "dusk": 1.10, "midday": 1.10, "night": 0.25, "average": 1.00},
    "falcon":   {"dawn": 1.10, "dusk": 1.05, "midday": 1.00, "night": 0.25, "average": 1.00},
    "accipiter":{"dawn": 1.15, "dusk": 1.10, "midday": 0.90, "night": 0.25, "average": 1.00},
    "owl":      {"dawn": 0.60, "dusk": 1.20, "midday": 0.30, "night": 2.00, "average": 0.80},
}


# ----------------------------
# UTILS
# ----------------------------

def beta_params_from_mean_k(mean: float, k: float) -> Tuple[float, float]:
    """Compute Beta(alpha, beta) parameters from a target mean and pseudo-count.

    Args:
        mean: Target mean in (0, 1); clipped safely away from 0/1.
        k: Pseudo-count (concentration). Larger k narrows the prior around mean.
    Returns:
        (alpha, beta) parameters for Beta distribution.
    """
    m = float(np.clip(mean, 1e-6, 1 - 1e-6))
    return m * k, (1 - m) * k


def parse_keyvals(spec: Optional[str],
                  allowed_keys: Optional[Tuple[str, ...]] = None) -> Dict[str, float]:
    """Parse comma/semicolon-separated `key=value` pairs into a float dict.

    Example: "river=0.7,park=0.2,woodland=0.1".
    Unknown keys are rejected if `allowed_keys` is provided.
    """
    if not spec:
        return {}
    parts = re.split(r"[;,]\s*", spec.strip())
    out: Dict[str, float] = {}
    for p in parts:
        if not p:
            continue
        if "=" not in p:
            raise ValueError(f"Expected 'key=value' pairs, got '{p}'.")
        k, v = p.split("=", 1)
        key = k.strip().lower()
        if allowed_keys and key not in allowed_keys:
            raise ValueError(f"Unknown key '{key}'. Allowed: {allowed_keys}")
        try:
            val = float(v.strip())
        except ValueError:
            raise ValueError(f"Value for '{key}' must be numeric, got '{v}'.")
        out[key] = val
    return out


def parse_user_obs(spec: Optional[str]) -> Dict[str, Tuple[int, int]]:
    """Parse personal checklist summaries like "Osprey=4/12,Kestrel=2/10".

    Returns a mapping of species -> (seen, total). Input is validated: total>0, 0<=seen<=total.
    """
    if not spec:
        return {}
    items = re.split(r"[;,]\s*", spec.strip())
    out: Dict[str, Tuple[int, int]] = {}
    for it in items:
        if not it:
            continue
        m = re.match(r"(.+?)\s*=\s*(\d+)\s*/\s*(\d+)", it)
        if not m:
            raise ValueError(f"Could not parse user_obs item: '{it}'. Use 'Species=seen/total'.")
        name = m.group(1).strip()
        seen = int(m.group(2))
        total = int(m.group(3))
        if total <= 0 or seen < 0 or seen > total:
            raise ValueError(f"Invalid counts for '{name}': seen={seen}, total={total}.")
        out[name] = (seen, total)
    return out


def clamp(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))


def safe_log1p_neg(x: np.ndarray) -> np.ndarray:
    """Compute log(1-x) safely for x in [0,1)."""
    x = np.clip(x, 0.0, 1.0 - 1e-12)
    return np.log1p(-x)


def prob_to_rate(p: np.ndarray) -> np.ndarray:
    """Convert probability of ≥1 event in base duration to Poisson rate.

    Uses λ = -ln(1 - p). Shapes are preserved element-wise.
    """
    return -safe_log1p_neg(p)


def rate_to_prob(lmbda: np.ndarray) -> np.ndarray:
    """Inverse conversion from rate to probability: p = 1 - exp(-λ)."""
    return 1.0 - np.exp(-lmbda)


def month_circular_distance(m1: int, m2: int) -> int:
    """Distance on a 12-month circle (1..12)."""
    a = ((m1 - 1) % 12) + 1
    b = ((m2 - 1) % 12) + 1
    d = abs(a - b)
    return min(d, 12 - d)


def seasonal_raw_factor(month: int, peak_month: int, sigma: float,
                        min_factor: float, max_factor: float) -> float:
    """Circular Gaussian-like seasonal factor in [min_factor, max_factor].

    At distance d months from peak, factor = min + (max-min)*exp(-0.5*(d/sigma)^2).
    """
    d = month_circular_distance(month, peak_month)
    amp = max_factor - min_factor
    return min_factor + amp * math.exp(-0.5 * (d / max(sigma, 1e-6)) ** 2)


def seasonal_relative_factor(species: str, month: int) -> float:
    """Species seasonality normalized to equal 1.0 in August.

    Keeps August priors as the baseline and scales other months up/down.
    """
    params = SPECIES_SEASONALITY[species]
    raw_now = seasonal_raw_factor(month, params["peak_month"], params["sigma"],
                                  params["min_factor"], params["max_factor"])
    raw_aug = seasonal_raw_factor(AUGUST_MONTH, params["peak_month"], params["sigma"],
                                  params["min_factor"], params["max_factor"])
    return float(raw_now / max(raw_aug, 1e-9))


def time_of_day_phase(s: str) -> str:
    """
    Normalize a user-supplied time-of-day string to one of:
    {'dawn','dusk','midday','night','average'}
    """
    t = (s or "").strip().lower()
    synonyms = {
        "sunrise": "dawn", "am": "dawn", "morning": "dawn",
        "sunset": "dusk", "pm": "dusk", "evening": "dusk",
        "noon": "midday", "mid-day": "midday",
        "late night": "night"
    }
    t = synonyms.get(t, t)
    if t not in {"dawn", "dusk", "midday", "night", "average"}:
        return "average"
    return t


def time_factor_baseline(phase: str) -> float:
    """Baseline time-of-day factor on the rate scale.

    Calibrated gently around 1.0; species group modifiers apply on top.
    """
    if phase == "dawn":
        return 1.10
    if phase == "dusk":
        return 1.10
    if phase == "midday":
        return 0.95  # slight reduction overall; soaring hawks will get a group boost.
    if phase == "night":
        return 0.60  # overall large reduction (diurnal species mostly).
    return 1.00     # average/unspecified


def weather_factor_rate_smooth(high_temp: float = 91.0,
                               low_temp: float = 62.0,
                               precip_chance: float = 5.0,
                               wind_speed: float = 8.0,
                               cloud_cover: float = 20.0) -> float:
    """Smooth, continuous weather factor on the rate scale.

    Combines temperature, wind, clouds, and precipitation into a multiplicative factor
    typically in ~[0.75, 1.10] (then clipped). Group-specific elasticities apply later.
    """
    # Temperature effect (use high_temp as proxy for daytime soaring environment)
    temp_sigma = 12.0
    f_temp = 0.80 + 0.30 * math.exp(-0.5 * ((high_temp - 88.0) / temp_sigma) ** 2)  # 0.80..1.10

    # Wind effect: modest boost near ~8 mph, penalties at very high winds
    wind_sigma = 6.0
    f_wind = 0.85 + 0.25 * math.exp(-0.5 * ((wind_speed - 8.0) / wind_sigma) ** 2)  # 0.85..1.10

    # Cloud cover: clearer skies help a bit
    cloud_sigma = 20.0
    f_cloud = 0.85 + 0.25 * math.exp(-0.5 * ((cloud_cover - 25.0) / cloud_sigma) ** 2)  # 0.85..1.10

    # Precipitation: monotone penalty; 0% -> ~1.0; 100% -> ~0.5
    pc = max(0.0, min(100.0, precip_chance)) / 100.0
    f_precip = 1.0 - 0.5 * (pc ** 1.2)  # 0.5..1.0

    w = f_temp * f_wind * f_cloud * f_precip
    return clamp(w, 0.60, 1.60)


def party_size_factor(party_size: int) -> float:
    """Saturating detection boost for more observers.

    Examples: 1→1.00, 2→~1.22, 3→~1.30, 4+→asymptote ~1.40.
    """
    n = max(1, int(party_size))
    if n == 1:
        return 1.0
    x = float(n - 1)
    return 1.0 + 0.40 * (x / (1.0 + 0.80 * x))


def route_vector(route: Optional[Dict[str, float]]) -> np.ndarray:
    """
    Normalize and return route composition vector in the canonical habitat order.
    If route is None, return normalized BASELINE_ROUTE.
    """
    if route is None or len(route) == 0:
        route = BASELINE_ROUTE
    vec = np.array([float(route.get(h, 0.0)) for h in HABITATS], dtype=float)
    s = float(np.sum(vec))
    if s <= 0:
        # default back to baseline to avoid divide-by-zero
        vec = np.array([BASELINE_ROUTE[h] for h in HABITATS], dtype=float)
        s = float(np.sum(vec))
    vec /= s
    return vec


def habitat_factor_for_species(species: str,
                               user_route_vec: np.ndarray,
                               baseline_route_vec: np.ndarray) -> float:
    """Habitat suitability ratio for a species.

    Computes dot(species_habitat_weights, route_vec) for user vs. baseline and returns the ratio.
    """
    w = SPECIES_HABITAT_WEIGHTS[species]
    sp_vec = np.array([w.get(h, 0.0) for h in HABITATS], dtype=float)
    # normalize species weights if needed
    sw = float(np.sum(sp_vec))
    if sw <= 0:
        sp_vec = np.ones_like(sp_vec) / len(sp_vec)
    else:
        sp_vec /= sw

    eps = 1e-6
    suit_user = float(np.dot(sp_vec, user_route_vec))
    suit_base = float(np.dot(sp_vec, baseline_route_vec))
    factor = (suit_user + eps) / (suit_base + eps)
    return clamp(factor, 0.5, 1.8)


def day_quality_log_normal(n_sims: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Shared day-quality multiplier Q ~ LogNormal with E[Q]=1.

    Uses mu = -0.5*sigma^2 so the lognormal has mean 1.0. Applies to all species per draw.
    """
    sigma = max(0.0, float(sigma))
    mu = -0.5 * sigma * sigma
    return rng.lognormal(mean=mu, sigma=sigma, size=n_sims)


# ----------------------------
# DATA CLASSES
# ----------------------------

@dataclass
class ModelInputs:
    """Configuration for a single run of the simulator.

    Most fields map directly to CLI flags. See README for guidance and ranges.
    """
    species_means_august: Dict[str, float]
    k: float = DEFAULT_K
    month: int = AUGUST_MONTH
    n_sims: int = 20_000
    duration_hours: float = 1.5
    exposure_gamma: float = 0.95
    time_of_day: str = "average"
    high_temp: float = 91.0
    low_temp: float = 62.0
    precip_chance: float = 5.0
    wind_speed: float = 8.0
    cloud_cover: float = 20.0
    sigma_day: float = 0.15            # shared-day lognormal sigma
    party_size: int = 1
    route: Optional[Dict[str, float]] = None
    seed: Optional[int] = None
    user_obs: Optional[Dict[str, Tuple[int, int]]] = None


@dataclass
class SimulationResults:
    """Outputs from a single run of the simulator.

    Arrays are sized as: n_sims draws × S species where applicable.
    """
    p_overall: np.ndarray                     # (n_sims,)
    species_names: List[str]
    species_p_marginal: Dict[str, np.ndarray]# P(≥1 of species i), per sim
    species_lambda_adj: np.ndarray            # adjusted rates, (n_sims, S)
    weights_first_sighting: np.ndarray       # λ_i / Σλ, (n_sims, S)
    exposure_effect: float
    weather_base: float
    time_base: float
    time_phase: str
    party_factor: float
    route_used: Dict[str, float]


# ----------------------------
# CORE SIMULATION
# ----------------------------

def run_simulation(inputs: ModelInputs) -> SimulationResults:
    """Vectorized Monte Carlo with species sensitivities, route, and seasonality.

    Algorithm per draw:
      1) Sample p_i ~ Beta(alpha_i, beta_i) (optionally updated by `--user_obs`).
      2) Convert to base hourly rates λ_i = -ln(1 - p_i).
      3) Build multipliers on the rate scale (weather, time-of-day, habitat, season, party, exposure, shared day Q).
      4) Compose adjusted rates λ'_i and compute: P(any)=1-exp(-Σ λ'_i), P_i=1-exp(-λ'_i), weights=λ'_i/Σλ'_i.
    Returns arrays of size n_sims and n_sims×S for downstream summaries.
    """
    rng = np.random.default_rng(inputs.seed)

    species_names = list(inputs.species_means_august.keys())
    S = len(species_names)
    n = int(inputs.n_sims)

    # Build Beta parameters (with optional user Beta-Binomial updates)
    alphas = np.empty(S)
    betas = np.empty(S)
    for j, sp in enumerate(species_names):
        mean = float(inputs.species_means_august[sp])
        a0, b0 = beta_params_from_mean_k(mean, inputs.k)
        if inputs.user_obs and sp in inputs.user_obs:
            seen, total = inputs.user_obs[sp]
            a = a0 + seen
            b = b0 + (total - seen)
        else:
            a, b = a0, b0
        alphas[j] = a
        betas[j] = b

    # Sample Beta draws per species -> probabilities matrix (n, S)
    p_samples = np.column_stack([rng.beta(alphas[j], betas[j], size=n) for j in range(S)])
    lambda_base = prob_to_rate(p_samples)  # (n, S)

    # Scenario factors common and species-specific
    phase = time_of_day_phase(inputs.time_of_day)
    T_base = time_factor_baseline(phase)  # baseline time factor (species-agnostic)
    W_base = weather_factor_rate_smooth(inputs.high_temp, inputs.low_temp,
                                        inputs.precip_chance, inputs.wind_speed, inputs.cloud_cover)

    # Route vectors
    user_route_vec = route_vector(inputs.route)
    baseline_route_vec = route_vector(BASELINE_ROUTE)

    # Exposure (duration with diminishing returns)
    exposure = (max(0.0, float(inputs.duration_hours)) / BASE_DURATION_HRS) ** float(inputs.exposure_gamma)

    # Party factor
    party = party_size_factor(inputs.party_size)

    # Day-quality shared multiplier
    Q = day_quality_log_normal(n, sigma=float(inputs.sigma_day), rng=rng)

    # Prepare per-species multipliers
    # Each species gets: W_species = W_base ** elasticity[group]
    #                    T_species = T_base * TIME_GROUP_MODIFIER[group][phase]
    #                    H_species = habitat factor(route vs baseline)
    #                    S_species = seasonal factor normalized to August
    W_elast = np.zeros(S, dtype=float)
    T_group = np.zeros(S, dtype=float)
    H_fac = np.zeros(S, dtype=float)
    S_season = np.zeros(S, dtype=float)

    for j, sp in enumerate(species_names):
        group = SPECIES_GROUP.get(sp, "falcon")  # default modest behavior
        W_elast[j] = WEATHER_ELASTICITY_BY_GROUP.get(group, 1.0)
        T_group[j] = TIME_GROUP_MODIFIER.get(group, {}).get(phase, 1.0)
        H_fac[j] = habitat_factor_for_species(sp, user_route_vec, baseline_route_vec)
        S_season[j] = seasonal_relative_factor(sp, int(inputs.month))

    # Broadcast multipliers (species scalars -> (n, S))
    # Species-specific scalars -> (n, S) by outer product with ones(n)
    W_species = (W_base ** W_elast)[None, :]              # (1, S)
    T_species = (T_base * T_group)[None, :]               # (1, S)
    H_species = H_fac[None, :]                            # (1, S)
    S_species = S_season[None, :]                         # (1, S)
    common_scale = (Q * (exposure * party))[:, None]      # (n, 1)

    # Adjusted rates per draw and species.
    # Formula reference: λ' = λ × Q × E × Party × Weather_species × Time_species × Habitat × Season
    lambda_adj = lambda_base * common_scale * W_species * T_species * H_species * S_species  # (n, S)

    # Overall and per-species probabilities
    sum_lambda = np.sum(lambda_adj, axis=1)                      # (n,)
    p_overall = 1.0 - np.exp(-sum_lambda)                        # (n,)
    species_p_marginal = {sp: 1.0 - np.exp(-lambda_adj[:, j]) for j, sp in enumerate(species_names)}

    # Contribution weights (safe divide): interpretable as the probability
    # the first sighting in a draw would be of species j under Poisson superposition.
    denom = np.where(sum_lambda > 0, sum_lambda, 1.0)
    weights = lambda_adj / denom[:, None]                        # (n, S)

    return SimulationResults(
        p_overall=p_overall,
        species_names=species_names,
        species_p_marginal=species_p_marginal,
        species_lambda_adj=lambda_adj,
        weights_first_sighting=weights,
        exposure_effect=exposure,
        weather_base=W_base,
        time_base=T_base,
        time_phase=phase,
        party_factor=party,
        route_used={h: float(user_route_vec[i]) for i, h in enumerate(HABITATS)},
    )


# ----------------------------
# REPORTING & VISUALIZATION
# ----------------------------

def summarize(results: SimulationResults, ci: float = DEFAULT_CI) -> Dict:
    """Summarize overall probability, per-species marginals, and contribution shares.

    Returns a JSON-serializable dict suitable for `--report_json` or downstream analysis.
    """
    p = results.p_overall
    mean_p = float(np.mean(p))
    std_p = float(np.std(p))
    alpha = (1.0 - float(ci)) / 2.0
    p_lo = float(np.quantile(p, alpha))
    p_hi = float(np.quantile(p, 1.0 - alpha))

    weights = results.weights_first_sighting  # (n, S)
    contr = np.mean(p[:, None] * weights, axis=0)  # (S,)
    shares = contr / mean_p if mean_p > 1e-12 else np.zeros_like(contr)

    species_stats = []
    for j, sp in enumerate(results.species_names):
        pi = results.species_p_marginal[sp]
        species_stats.append({
            "species": sp,
            "p_mean": float(np.mean(pi)),
            "p_lo": float(np.quantile(pi, alpha)),
            "p_hi": float(np.quantile(pi, 1.0 - alpha)),
            "contribution_share": float(shares[j]),
        })

    species_stats.sort(key=lambda d: d["contribution_share"], reverse=True)

    return {
        "overall": {
            "mean_p": mean_p,
            "std_p": std_p,
            "ci_level": float(ci),
            "p_lo": p_lo,
            "p_hi": p_hi,
            "odds": (mean_p / (1.0 - mean_p)) if mean_p < 1.0 else float("inf"),
        },
        "factors": {
            "duration_effect_hours_equiv": float(results.exposure_effect) * BASE_DURATION_HRS,
            "time_phase": results.time_phase,
            "time_factor_base": float(results.time_base),
            "weather_factor_base": float(results.weather_base),
            "party_size_factor": float(results.party_factor),
            "route_used": results.route_used,
        },
        "species": species_stats,
    }


def interpret(mean_p: float) -> str:
    """Short qualitative interpretation of mean P(any)."""
    if mean_p > 0.70:
        return "High likelihood: a raptor sighting is very likely on this walk."
    if mean_p > 0.50:
        return "Moderate likelihood: better than even odds; dawn/dusk helps."
    if mean_p > 0.30:
        return "Fair chance: possible, but not guaranteed; longer walks help."
    return "Low likelihood: conditions or duration may be limiting today."


def plot_distribution(p_overall: np.ndarray, mean_p: float, path: str = "raptor_prob_dist.png") -> None:
    """Histogram of simulated overall probabilities (saved to file)."""
    plt.figure(figsize=(8, 5))
    plt.hist(p_overall, bins=30, density=True, alpha=0.7)
    plt.axvline(mean_p, linestyle="--", label=f"Mean = {mean_p:.1%}")
    plt.title("Distribution of Simulated Probability of Seeing ≥1 Raptor")
    plt.xlabel("Probability")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=144)
    plt.close()


def plot_contributions(species_stats: List[Dict], path: str = "raptor_contributions.png") -> None:
    """Bar chart of species contribution shares (saved to file)."""
    names = [d["species"] for d in species_stats]
    shares = [d["contribution_share"] for d in species_stats]
    plt.figure(figsize=(10, 5))
    plt.bar(names, shares)
    plt.ylabel("Contribution share to P(any)")
    plt.xticks(rotation=30, ha="right")
    plt.title("Species Contribution Shares (sum ≈ 100%)")
    plt.tight_layout()
    plt.savefig(path, dpi=144)
    plt.close()


def print_explain_per_species(results: SimulationResults,
                              inputs: ModelInputs) -> None:
    """Print per-species multiplier components (approximate, using medians).

    Uses deterministic recomputation of scalar factors to avoid Monte Carlo noise.
    """
    # We'll reconstruct approximate per-species multiplier medians from known scalars:
    route_vec = route_vector(inputs.route)
    base_vec = route_vector(BASELINE_ROUTE)
    W_base = weather_factor_rate_smooth(inputs.high_temp, inputs.low_temp,
                                        inputs.precip_chance, inputs.wind_speed, inputs.cloud_cover)
    phase = time_of_day_phase(inputs.time_of_day)
    T_base = time_factor_baseline(phase)

    print("\nPer-species factor breakdown (approximate medians):")
    for sp in results.species_names:
        group = SPECIES_GROUP.get(sp, "falcon")
        W_elast = WEATHER_ELASTICITY_BY_GROUP.get(group, 1.0)
        T_mod = TIME_GROUP_MODIFIER.get(group, {}).get(phase, 1.0)
        H_fac = habitat_factor_for_species(sp, route_vec, base_vec)
        S_fac = seasonal_relative_factor(sp, inputs.month)

        W_sp = W_base ** W_elast
        T_sp = T_base * T_mod
        print(f"  - {sp:22s}  Weather≈{W_sp:5.2f}  Time≈{T_sp:5.2f}  Habitat≈{H_fac:5.2f}  Season≈{S_fac:5.2f}")


# ----------------------------
# SENSITIVITY ANALYSIS (optional)
# ----------------------------

def quick_sensitivity(inputs: ModelInputs, base_summary: Dict) -> List[Tuple[str, Dict]]:
    """
    Small "what-if" analysis to see directional sensitivity.
    We hold seed fixed (if provided) so results are comparable.
    Returns list of (label, summary_dict).
    """
    scenarios: List[Tuple[str, ModelInputs]] = []

    # Duration +/- 1 hour (bounded at 0)
    scenarios.append(("duration -1h", dataclasses_replace(inputs, duration_hours=max(0.0, inputs.duration_hours - 1.0))))
    scenarios.append(("duration +1h", dataclasses_replace(inputs, duration_hours=inputs.duration_hours + 1.0)))

    # Time of day swaps
    for t in ["dawn", "midday", "dusk", "night"]:
        if time_of_day_phase(inputs.time_of_day) != t:
            scenarios.append((f"time={t}", dataclasses_replace(inputs, time_of_day=t)))

    # Temperature and wind tweaks
    scenarios.append(("+10°F", dataclasses_replace(inputs, high_temp=inputs.high_temp + 10.0)))
    scenarios.append(("-10°F", dataclasses_replace(inputs, high_temp=inputs.high_temp - 10.0)))
    scenarios.append(("+6 mph wind", dataclasses_replace(inputs, wind_speed=inputs.wind_speed + 6.0)))
    scenarios.append(("-6 mph wind", dataclasses_replace(inputs, wind_speed=max(0.0, inputs.wind_speed - 6.0))))

    # Compute summaries (holding seed to reduce MC noise)
    out: List[Tuple[str, Dict]] = []
    for label, inp in scenarios:
        res = run_simulation(inp)
        summ = summarize(res, ci=base_summary["overall"]["ci_level"])
        out.append((label, summ))
    return out


def dataclasses_replace(inputs: ModelInputs, **kwargs) -> ModelInputs:
    """
    Lightweight replacement for dataclasses.replace to avoid import.
    Returns a shallow-copied ModelInputs with provided fields updated.
    """
    d = inputs.__dict__.copy()
    d.update(kwargs)
    return ModelInputs(**d)


# ----------------------------
# CLI
# ----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Estimate P(≥1 raptor) on a Boise Greenbelt walk via a hazard-based Monte Carlo "
            "model with species priors, time/weather/habitat/season multipliers, party size, "
            "and diminishing returns with duration."
        )
    )
    # Season & time
    p.add_argument(
        "--month", type=int, default=AUGUST_MONTH,
        help="Month number (1-12). Seasonality is relative to August baselines."
    )
    p.add_argument(
        "--time_of_day", type=str, default="average",
        help=(
            "Time of day: dawn, dusk, midday, night, average. Synonyms like 'morning', 'sunset' are accepted."
        ),
    )
    # Weather
    p.add_argument("--high_temp", type=float, default=91.0, help="High temperature (°F). Used in thermals proxy.")
    p.add_argument("--low_temp", type=float, default=62.0, help="Low temperature (°F).")
    p.add_argument("--precip_chance", type=float, default=5.0, help="Precipitation chance (%), 0-100.")
    p.add_argument("--wind_speed", type=float, default=8.0, help="Wind speed (mph).")
    p.add_argument("--cloud_cover", type=float, default=20.0, help="Cloud cover (%), 0-100.")
    # Route & party
    p.add_argument(
        "--route", type=str, default=None,
        help=(
            'Route composition, e.g. "river=0.7,park=0.2,woodland=0.1". Keys: river, park, woodland, urban. '
            "Weights are normalized; omit to use a baseline Greenbelt mix."
        ),
    )
    p.add_argument("--party_size", type=int, default=1, help="Number of observers (>=1).")
    # Duration & exposure
    p.add_argument(
        "--duration_hours", type=float, default=1.5,
        help="Walk duration in hours (>=0). Diminishing returns via exposure gamma."
    )
    p.add_argument(
        "--exposure_gamma", type=float, default=0.95,
        help="Diminishing returns exponent in (0,1]. 1.0 = linear exposure."
    )
    # Priors / evidence
    p.add_argument("--k", type=float, default=DEFAULT_K, help="Beta prior pseudo-count strength.")
    p.add_argument(
        "--user_obs", type=str, default=None,
        help=(
            'Personal checklists to update priors, e.g. "Osprey=4/12,American Kestrel=2/10". '
            "Interpreted as seen/total checklists (Beta–Binomial)."
        ),
    )
    # Monte Carlo & random
    p.add_argument("--n_sims", type=int, default=20_000, help="Number of Monte Carlo simulations.")
    p.add_argument(
        "--sigma_day", type=float, default=0.15,
        help="Std dev of shared day-quality LogNormal on rates (0 = no shared effect)."
    )
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    # Output
    p.add_argument("--ci", type=float, default=DEFAULT_CI, help="Credible interval level (e.g., 0.90 or 0.95).")
    p.add_argument(
        "--plot", action="store_true",
        help='Save plots: "raptor_prob_dist.png" (distribution) and "raptor_contributions.png" (species shares).'
    )
    p.add_argument("--report_json", type=str, default=None, help="Path to save summary JSON (see README for schema).")
    p.add_argument("--report_csv", type=str, default=None, help="Path to save per-species CSV (probabilities and shares).")
    p.add_argument("--list_species", action="store_true", help="Print known species and exit.")
    p.add_argument(
        "--explain_per_species", action="store_true",
        help="Print per-species multiplier breakdown (time/weather/habitat/season)."
    )
    # Sensitivity
    p.add_argument("--sensitivity", action="store_true", help="Run a small what-if sensitivity analysis.")
    return p


def main():
    args = build_arg_parser().parse_args()

    # Quick list and exit
    if args.list_species:
        print("Species in model:")
        for sp in SPECIES_MEANS_AUGUST:
            print(f"- {sp}")
        return

    # Validate inputs
    if not (1 <= int(args.month) <= 12):
        raise ValueError("month must be in 1..12.")
    if float(args.duration_hours) < 0:
        raise ValueError("duration_hours must be ≥ 0.")
    if not (0 < float(args.exposure_gamma) <= 1.0):
        raise ValueError("exposure_gamma must be in (0, 1].")
    for name, val in [("precip_chance", args.precip_chance), ("cloud_cover", args.cloud_cover)]:
        if not (0.0 <= float(val) <= 100.0):
            raise ValueError(f"{name} must be between 0 and 100.")

    route_dict = parse_keyvals(args.route, allowed_keys=HABITATS) if args.route else None
    user_obs = parse_user_obs(args.user_obs) if args.user_obs else None

    inputs = ModelInputs(
        species_means_august=SPECIES_MEANS_AUGUST,
        k=float(args.k),
        month=int(args.month),
        n_sims=int(args.n_sims),
        duration_hours=float(args.duration_hours),
        exposure_gamma=float(args.exposure_gamma),
        time_of_day=args.time_of_day,
        high_temp=float(args.high_temp),
        low_temp=float(args.low_temp),
        precip_chance=float(args.precip_chance),
        wind_speed=float(args.wind_speed),
        cloud_cover=float(args.cloud_cover),
        sigma_day=float(args.sigma_day),
        party_size=int(args.party_size),
        route=route_dict,
        seed=args.seed,
        user_obs=user_obs,
    )

    # Run simulation
    results = run_simulation(inputs)
    summary = summarize(results, ci=float(args.ci))

    # Human-readable summary
    overall = summary["overall"]
    factors = summary["factors"]
    print(f"Mean probability of seeing ≥1 raptor: {overall['mean_p']:.2%}")
    print(f"Standard deviation: {overall['std_p']:.2%}")
    print(f"{int(args.ci*100)}% credible interval: {overall['p_lo']:.2%} – {overall['p_hi']:.2%}")
    print(f"Odds of success: {overall['odds']:.2f}:1")
    print(f"Factors: duration_effect≈{factors['duration_effect_hours_equiv']:.2f}h, "
          f"time_phase={factors['time_phase']}, time_base≈{factors['time_factor_base']:.2f}, "
          f"weather_base≈{factors['weather_factor_base']:.2f}, party≈{factors['party_size_factor']:.2f}, "
          f"route={factors['route_used']}")
    print(interpret(overall["mean_p"]))

    # Top contributors
    species_stats = summary["species"]
    print("\nTop contributors (share of overall P(any)):")
    for row in species_stats[:5]:
        print(f"  - {row['species']:<22s} share={row['contribution_share']*100:5.1f}%  "
              f"P(≥1) mean={row['p_mean']:.1%}  CI[{row['p_lo']:.1%},{row['p_hi']:.1%}]")

    # Optional per-species explanation
    if args.explain_per_species:
        print_explain_per_species(results, inputs)

    # Optional plots
    if args.plot:
        plot_distribution(results.p_overall, overall["mean_p"], path="raptor_prob_dist.png")
        plot_contributions(species_stats, path="raptor_contributions.png")
        print("\nSaved plots: raptor_prob_dist.png, raptor_contributions.png")

    # Optional JSON
    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved JSON report to: {args.report_json}")

    # Optional CSV (per-species)
    if args.report_csv:
        with open(args.report_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["species", "p_mean", "p_lo", "p_hi", "contribution_share"])
            for row in species_stats:
                writer.writerow([row["species"], f"{row['p_mean']:.6f}", f"{row['p_lo']:.6f}",
                                 f"{row['p_hi']:.6f}", f"{row['contribution_share']:.6f}"])
        print(f"Saved per-species CSV to: {args.report_csv}")

    # Optional sensitivity
    if args.sensitivity:
        print("\n--- Quick sensitivity analysis ---")
        sens = quick_sensitivity(inputs, summary)
        for label, summ in sens:
            mp = summ["overall"]["mean_p"]
            print(f"{label:>14s}: mean P(any) = {mp:.2%}")


if __name__ == "__main__":
    main()
