import numpy as np

from raptor_model_advanced import (
    run_simulation,
    ModelInputs,
    SPECIES_MEANS_AUGUST,
    BASELINE_ROUTE,
)


def base_inputs(**overrides):
    params = dict(
        species_means_august=SPECIES_MEANS_AUGUST,
        month=8,
        n_sims=2000,
        duration_hours=1.5,
        exposure_gamma=0.95,
        time_of_day="dawn",
        high_temp=90.0,
        low_temp=60.0,
        precip_chance=5.0,
        wind_speed=8.0,
        cloud_cover=20.0,
        sigma_day=0.15,
        party_size=1,
        route=BASELINE_ROUTE,
        seed=123,
        user_obs=None,
    )
    params.update(overrides)
    return ModelInputs(**params)


def test_reproducibility_same_seed():
    inp = base_inputs(seed=123)
    r1 = run_simulation(inp)
    r2 = run_simulation(inp)
    assert np.array_equal(r1.p_overall, r2.p_overall)
    assert np.array_equal(r1.species_lambda_adj, r2.species_lambda_adj)


def test_duration_zero_gives_zero_prob():
    inp = base_inputs(duration_hours=0.0)
    r = run_simulation(inp)
    assert np.all(r.p_overall == 0.0)


def test_user_obs_increases_species_probability():
    # Strong positive personal evidence for Osprey should raise its marginal mean
    base = run_simulation(base_inputs(user_obs=None))
    with_obs = run_simulation(base_inputs(user_obs={"Osprey": (10, 10)}))
    m0 = float(np.mean(base.species_p_marginal["Osprey"]))
    m1 = float(np.mean(with_obs.species_p_marginal["Osprey"]))
    assert m1 > m0
