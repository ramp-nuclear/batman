import hypothesis.strategies as st
import pytest
from coremaker.materials.mixture import Mixture
from hypothesis import given
from isotopes import I135, U235
from reactions.reaction import Reaction
from reactions.reaction_category import Fission
from reactions.reaction_rate import ReactionRate

from batman.exponentiators import CF3Magnus2IPFCRAM as Magnus
from batman.exponentiators import Exponentiator
from batman.exponentiators import cram48_coefficients as CRAM48
from batman.graphs.decay import DecayGraph
from batman.graphs.filters import GraphFilter
from batman.solver import (
    Configuration,
    InputData,
    timestep_const_flux_initial_power,
    timestep_const_power_linear_flux,
    timestep_constant_flux_energy_conserve,
    timestep_constant_power,
)
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

powers = st.floats(min_value=1e-1, max_value=1e7)


def _uranium_den(U0, rateF, t) -> float:
    return U0 - rateF * t


@pytest.fixture(scope="module")
def simple_data():
    data = InputData([decay], [[reaction]], [_filter], [mix], [1.0])
    return EasyData.from_input(data)


@pytest.fixture(scope="module")
def magnus48() -> Exponentiator:
    return Magnus(*CRAM48())


@given(p=powers)
def test_constant_flux_at_initial_power_is_undershooting_energy_loss(p, simple_data):
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
def test_energy_preserving_constant_flux_is_getting_energy_right_for_single_isotope(p, simple_data):
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
        step_strategy=timestep_constant_flux_energy_conserve,
        config=config,
    )
    res_mix = newden[0]
    expected_u = _uranium_den(u0, norm * reaction.mean, t)
    assert 1 - res_mix[U235] == pytest.approx(1 - expected_u, rel=1e-2, abs=1e-10)


@given(p=powers)
def test_magnus_exponential_with_constant_power_constant_flux_is_same_as_cram(p, simple_data, magnus48):
    config_magnus = Configuration(
        p_rtol=1,
        p_atol=1,
        time_guesser=tguess,
        expo=magnus48,
        threshold=1e-18,
    )
    config_cram = Configuration(
        p_rtol=1,
        p_atol=1,
        time_guesser=tguess,
        threshold=1e-18,
    )

    cramden, result = timestep_constant_power(
        simple_data,
        p,
        t,
        step_strategy=timestep_const_flux_initial_power,
        config=config_cram,
    )
    magden, result = timestep_constant_power(
        simple_data,
        p,
        t,
        step_strategy=timestep_const_flux_initial_power,
        config=config_magnus,
    )

    crammix, magmix = cramden[0], magden[0]
    assert 1 - magmix[U235] == pytest.approx(1 - crammix[U235], rel=1e-8, abs=1e-10)


@given(p=powers)
def test_magnus_exponential_with_constant_power_linear_flux_right_for_single_isotope(p, simple_data, magnus48):
    norm = simple_data.normalize(p, decay_power_allowed=False)
    config = Configuration(
        p_rtol=1,
        p_atol=1,
        time_guesser=tguess,
        expo=magnus48,
        threshold=1e-18,
    )
    newden, result = timestep_constant_power(
        simple_data,
        p,
        t,
        step_strategy=timestep_const_power_linear_flux,
        config=config,
    )
    res_mix = newden[0]
    expected_u = _uranium_den(u0, norm * reaction.mean, t)
    assert 1 - res_mix[U235] == pytest.approx(1 - expected_u, rel=1e-2, abs=1e-10)
