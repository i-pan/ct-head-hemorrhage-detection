import pytest

from skp.search import generate_search, generate_sequence


def test_generate_sequence_shape_and_bounds():
    sequence = generate_sequence(
        num_samples=5,
        num_dims=3,
        skip=0,
        per_dim_shift=False,
        shuffle_sequence=False,
        primes=[3, 5, 7],
    )

    assert len(sequence) == 5
    assert all(len(point) == 3 for point in sequence)
    assert all(0.0 <= value < 1.0 for point in sequence for value in point)


def test_generate_search_supports_continuous_and_discrete_spaces():
    trials = generate_search(
        {
            "optimizer_params__lr": {"min": 1e-5, "max": 1e-3, "scaling": "log"},
            "batch_size": {"feasible_points": [8, 16, 32]},
        },
        num_trials=4,
    )

    assert len(trials) == 4
    assert all(1e-5 <= trial.optimizer_params__lr <= 1e-3 for trial in trials)
    assert {trial.batch_size for trial in trials}.issubset({8, 16, 32})


def test_generate_search_rejects_invalid_trial_count():
    with pytest.raises(ValueError, match="num_trials"):
        generate_search({"x": {"min": 0, "max": 1}}, num_trials=0)
