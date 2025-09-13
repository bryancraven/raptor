import json
import csv

import numpy as np

from raptor_model_advanced import (
    run_simulation,
    summarize,
    print_explain_per_species,
    ModelInputs,
    SPECIES_MEANS_AUGUST,
    BASELINE_ROUTE,
)


def make_results(seed=123):
    inputs = ModelInputs(
        species_means_august=SPECIES_MEANS_AUGUST,
        month=8,
        n_sims=1500,
        duration_hours=1.0,
        time_of_day="dawn",
        party_size=1,
        route=BASELINE_ROUTE,
        seed=seed,
    )
    return run_simulation(inputs)


def test_summarize_schema():
    res = make_results()
    summ = summarize(res, ci=0.9)
    assert set(summ.keys()) == {"overall", "factors", "species"}
    assert 0.0 <= summ["overall"]["mean_p"] <= 1.0
    assert 0.0 <= summ["overall"]["p_lo"] <= summ["overall"]["p_hi"] <= 1.0
    assert isinstance(summ["species"], list) and len(summ["species"]) > 0
    # Shares sum to ~1
    total_share = sum(s["contribution_share"] for s in summ["species"])
    assert np.isclose(total_share, 1.0, atol=1e-2)


def test_write_reports(tmp_path):
    res = make_results()
    summ = summarize(res, ci=0.9)

    # JSON
    jpath = tmp_path / "summary.json"
    jpath.write_text(json.dumps(summ), encoding="utf-8")
    loaded = json.loads(jpath.read_text(encoding="utf-8"))
    assert loaded["overall"]["ci_level"] == 0.9

    # CSV with headers
    cpath = tmp_path / "species.csv"
    with cpath.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["species", "p_mean", "p_lo", "p_hi", "contribution_share"])
        for row in summ["species"]:
            w.writerow([row["species"], row["p_mean"], row["p_lo"], row["p_hi"], row["contribution_share"]])
    with cpath.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        assert headers == ["species", "p_mean", "p_lo", "p_hi", "contribution_share"]
        rows = list(reader)
        assert len(rows) == len(summ["species"])


def test_explain_per_species_prints(capsys):
    res = make_results()
    # Rebuild minimal inputs for factor recomputation
    inputs = ModelInputs(
        species_means_august=SPECIES_MEANS_AUGUST,
        month=8,
        n_sims=100,
        duration_hours=1.0,
        time_of_day="dawn",
        party_size=1,
        route=BASELINE_ROUTE,
    )
    print_explain_per_species(res, inputs)
    out = capsys.readouterr().out
    assert "Per-species factor breakdown" in out
    # At least one known species appears
    assert any(sp in out for sp in SPECIES_MEANS_AUGUST.keys())

