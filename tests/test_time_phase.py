from raptor_model_advanced import time_of_day_phase


def test_time_of_day_synonyms():
    assert time_of_day_phase("sunrise") == "dawn"
    assert time_of_day_phase("morning") == "dawn"
    assert time_of_day_phase("sunset") == "dusk"
    assert time_of_day_phase("pm") == "dusk"
    assert time_of_day_phase("noon") == "midday"
    assert time_of_day_phase("late night") == "night"
    assert time_of_day_phase("unknown") == "average"

