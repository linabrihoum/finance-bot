from __future__ import annotations

import numpy as np
import pytest

from strategy_evaluation.q_learner import QLearner


def _make_learner(**overrides) -> QLearner:
    defaults = dict(
        num_states=10,
        num_actions=3,
        alpha=0.2,
        gamma=0.9,
        rar=0.0,
        radr=0.99,
        dyna=0,
        random_state=42,
    )
    defaults.update(overrides)
    return QLearner(**defaults)


class TestInit:
    def test_q_table_initialized_to_zeros(self):
        lrn = _make_learner()
        assert np.all(lrn.Q == 0.0)

    def test_q_table_shape(self):
        lrn = _make_learner(num_states=5, num_actions=2)
        assert lrn.Q.shape == (5, 2)


class TestQuerySetState:
    def test_returns_valid_action(self):
        lrn = _make_learner()
        for s in range(10):
            action = lrn.querysetstate(s)
            assert 0 <= action < 3

    def test_greedy_picks_highest_q(self):
        lrn = _make_learner(rar=0.0)
        lrn.Q[3, 2] = 5.0
        assert lrn.querysetstate(3) == 2

    def test_sets_internal_state(self):
        lrn = _make_learner()
        lrn.querysetstate(7)
        assert lrn.s == 7


class TestQuery:
    def test_q_table_updated(self):
        lrn = _make_learner(rar=0.0)
        lrn.querysetstate(0)
        lrn.query(1, 1.0)
        assert np.any(lrn.Q != 0.0)

    def test_bellman_update_exact_value(self):
        # alpha=1, gamma=0, Q=0 everywhere → Q[s,a] = r after one update
        lrn = QLearner(num_states=5, num_actions=2, alpha=1.0, gamma=0.0, rar=0.0, radr=1.0, random_state=0)
        lrn.querysetstate(0)
        lrn.query(1, 3.0)
        assert lrn.Q[0, 0] == pytest.approx(3.0)

    def test_rar_decays_after_each_call(self):
        lrn = _make_learner(rar=0.5, radr=0.9)
        lrn.querysetstate(0)
        lrn.query(1, 0.0)
        assert lrn.rar == pytest.approx(0.5 * 0.9)
        lrn.query(2, 0.0)
        assert lrn.rar == pytest.approx(0.5 * 0.9 * 0.9)

    def test_returns_valid_action(self):
        lrn = _make_learner()
        lrn.querysetstate(0)
        action = lrn.query(1, 0.0)
        assert 0 <= action < 3

    def test_updates_internal_state_to_s_prime(self):
        lrn = _make_learner(rar=0.0)
        lrn.querysetstate(0)
        lrn.query(4, 0.0)
        assert lrn.s == 4


class TestDyna:
    def test_dyna_runs_without_error(self):
        lrn = _make_learner(dyna=10, rar=0.0)
        lrn.querysetstate(0)
        lrn.query(1, 1.0)

    def test_dyna_modifies_q_table(self):
        lrn_no_dyna = _make_learner(dyna=0, rar=0.0, random_state=0)
        lrn_dyna = _make_learner(dyna=20, rar=0.0, random_state=0)

        for lrn in (lrn_no_dyna, lrn_dyna):
            lrn.querysetstate(0)
            lrn.query(1, 1.0)

        # Dyna performs extra updates, so Q-tables should differ
        assert not np.allclose(lrn_no_dyna.Q, lrn_dyna.Q)


class TestReproducibility:
    def test_same_seed_same_actions(self):
        def run(seed: int) -> list:
            lrn = QLearner(num_states=20, num_actions=3, rar=0.8, random_state=seed)
            lrn.querysetstate(0)
            return [lrn.query(i % 20, float(i)) for i in range(10)]

        assert run(7) == run(7)

    def test_different_seeds_differ(self):
        def run(seed: int) -> list:
            lrn = QLearner(num_states=20, num_actions=3, rar=0.8, random_state=seed)
            lrn.querysetstate(0)
            return [lrn.query(i % 20, float(i)) for i in range(10)]

        assert run(1) != run(2)
