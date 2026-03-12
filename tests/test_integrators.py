import hypothesis.strategies as st
import pytest
from coremaker.materials.mixture import Mixture
from hypothesis import given
from isotopes import I135, U235
from reactions.reaction import Reaction
from reactions.reaction_category import Fission
from reactions.reaction_rate import ReactionRate

from batman.graphs.decay import DecayGraph
from batman.graphs.filters import GraphFilter
from batman.solver import Configuration, InputData, timestep_const_flux_initial_power, timestep_constant_power
from batman.solver import SerialEasyData as EasyData
from batman.solver.time_est import max_step_initial_correct_predictor as tguess

U235FISS_ENER = 200e6  # A joule per fission event to make it simpler.
U235nf = Reaction.from_reaction(U235, Fission, I135, energy=U235FISS_ENER)
reaction = ReactionRate("moo", U235nf, 1, 0.0)
decay = DecayGraph()
u0 = 1.0
t = 1e-1
mix = Mixture({U235: u0}, 20.0)
_filter = GraphFilter()

powers = st.floats(min_value=1e-1, max_value=100)


def _uranium_den(U0, rateF, t) -> float:
    return U0 - rateF * t


@pytest.fixture(scope="module")
def simple_data():
    data = InputData([decay], [[reaction]], [_filter], [mix], [1.0])
    return EasyData.from_input(data)


@given(p=powers)
def test_predictor_with_power_is_undershooting_power(p, simple_data):
    config = Configuration(p_rtol=1, p_atol=1, time_guesser=tguess, threshold=1e-18)
    norm = simple_data.normalize(p, decay_power_allowed=False)
    newden, result = timestep_constant_power(
        simple_data,
        p,
        t,
        step_strategy=timestep_const_flux_initial_power,
        config=config,
    )
    res_mix = newden[0]
    expected_u = _uranium_den(u0, norm * reaction.mean, t)
    assert res_mix[U235] > (1 - 1e-10) * expected_u


@given(p=powers)
def test_energy_preserving_predictor_is_getting_power_right(p, simple_data):
    norm = simple_data.normalize(p, decay_power_allowed=False)
    config = Configuration(
        p_rtol=1,
        p_atol=1,
        time_guesser=tguess,
        threshold=1e-18,
    )
    newden, result = timestep_constant_power(
        simple_data,
        p,
        t,
        step_strategy=timestep_const_flux_initial_power,
        config=config,
    )
    res_mix = newden[0]
    expected_u = _uranium_den(u0, norm * reaction.mean, t)
    assert res_mix[U235] == pytest.approx(expected_u, rel=1e-8, abs=1e-10)
