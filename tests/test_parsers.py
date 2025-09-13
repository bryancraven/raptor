import pytest

from raptor_model_advanced import parse_keyvals, parse_user_obs, HABITATS


def test_parse_keyvals_basic():
    out = parse_keyvals("river=0.7,park=0.2;woodland=0.1", allowed_keys=HABITATS)
    assert set(out.keys()) == {"river", "park", "woodland"}
    assert pytest.approx(out["river"], rel=0, abs=1e-9) == 0.7


def test_parse_keyvals_rejects_unknown():
    with pytest.raises(ValueError):
        parse_keyvals("desert=1.0", allowed_keys=HABITATS)


def test_parse_user_obs_basic():
    out = parse_user_obs("Osprey=4/12,American Kestrel=2/10")
    assert out["Osprey"] == (4, 12)
    assert out["American Kestrel"] == (2, 10)


def test_parse_user_obs_invalid_counts():
    with pytest.raises(ValueError):
        parse_user_obs("Osprey=5/3")
