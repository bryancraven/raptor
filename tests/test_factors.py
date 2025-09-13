import numpy as np
import pytest

from raptor_model_advanced import (
    prob_to_rate,
    rate_to_prob,
    party_size_factor,
    weather_factor_rate_smooth,
    route_vector,
    habitat_factor_for_species,
    seasonal_relative_factor,
    month_circular_distance,
    BASELINE_ROUTE,
    HABITATS,
    SPECIES_MEANS_AUGUST,
)


def test_prob_rate_roundtrip():
    p = np.array([0.01, 0.2, 0.5, 0.8])
    lam = prob_to_rate(p)
    p_back = rate_to_prob(lam)
    assert np.allclose(p, p_back, rtol=0, atol=1e-12)


def test_party_size_monotonic_and_bounds():
    vals = [party_size_factor(n) for n in range(1, 8)]
    assert all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
    assert vals[0] == pytest.approx(1.0, abs=1e-12)
    assert vals[-1] <= 1.41  # asymptote ~1.40


def test_route_vector_normalizes():
    v = route_vector({"river": 2.0, "park": 2.0})
    assert np.isclose(v.sum(), 1.0)
    # Only river+park specified -> woodland+urban 0
    assert v[list(HABITATS).index("woodland")] == 0.0


def test_habitat_factor_baseline_is_oneish():
    # Using baseline vs baseline should be ~1 for any species
    user_vec = route_vector(BASELINE_ROUTE)
    base_vec = route_vector(BASELINE_ROUTE)
    for sp in SPECIES_MEANS_AUGUST.keys():
        f = habitat_factor_for_species(sp, user_vec, base_vec)
        assert f == pytest.approx(1.0, rel=0, abs=1e-6)


def test_month_circular_distance_wrap():
    assert month_circular_distance(12, 1) == 1
    assert month_circular_distance(1, 12) == 1
    assert month_circular_distance(3, 9) == 6


def test_seasonal_relative_august_is_one():
    for sp in SPECIES_MEANS_AUGUST.keys():
        f = seasonal_relative_factor(sp, 8)
        assert f == pytest.approx(1.0, rel=0, abs=1e-12)


def test_weather_factor_bounds_and_precip_penalty():
    w_clear = weather_factor_rate_smooth(high_temp=88, precip_chance=0, wind_speed=8, cloud_cover=25)
    w_rain = weather_factor_rate_smooth(high_temp=88, precip_chance=100, wind_speed=8, cloud_cover=25)
    assert 0.60 <= w_clear <= 1.60
    assert 0.60 <= w_rain <= 1.60
    assert w_clear > w_rain
