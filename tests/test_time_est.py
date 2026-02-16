"""Tests for the timestep estimation functionality.

"""
from datetime import timedelta
from math import inf

import hypothesis.strategies as st
import pytest
from coremaker.materials.mixture import Mixture
from hypothesis import given, settings
from isotopes import I135, U235
from reactions import Fission, Reaction, ReactionRate

from batman import DecayGraph, GraphFilter, InputData, SerialEasyData
from batman.solver.time_est import _allowed_predictor_step, deriv_p, \
    max_step_initial_correct_predictor as maximal_step_at_power
from batman.units import EV_TO_MJ


U235FISS_ENER = 200e6  # eV
u0 = 1e-3
fiss_rate = 2.6


# noinspection PyPep8Naming
def undying_u235() -> SerialEasyData:
    """Single region U235 based SerialEasyData where the U235 never dies away.
    Its fission creates energy and I135, which does not itself decay.

    """
    U235nf = Reaction.from_reaction(U235, Fission, energy=U235FISS_ENER,
                                    target=I135)
    reacrate = ReactionRate('moo', U235nf, 1., 0)
    mixture = Mixture({U235: u0}, 20.)
    data = SerialEasyData.from_input(InputData(
        [DecayGraph()], [[reacrate]], [GraphFilter()], [mixture], [1.],
        accumulates=[frozenset({U235})]
        ))
    return data


# noinspection PyPep8Naming
def dying_u235() -> SerialEasyData:
    """Single region U235 based SerialEasyData where the U235 never dies away.
    Its fission creates energy and I135, which does not itself decay.

    """
    U235nf = Reaction.from_reaction(U235, Fission, energy=U235FISS_ENER,
                                    target=I135, nu=2.44)
    reacrate = ReactionRate('moo', U235nf, fiss_rate, 0)
    mixture = Mixture({U235: u0}, 20.)
    data = SerialEasyData.from_input(InputData(
        [DecayGraph()], [[reacrate]], [GraphFilter()], [mixture], [1.]
        ))
    return data


def test_zero_change_makes_maximum_infinite():
    assert inf == maximal_step_at_power(undying_u235(), inf, 1.,
                                        atol=1e-3, rtol=1e-2)


@pytest.mark.parametrize(("data", "result"),
                         [(undying_u235(), 0.),
                          (dying_u235(),
                           -EV_TO_MJ * U235FISS_ENER * u0 * fiss_rate**2)]
                         )
def test_time_derivative_by_example(data, result):
    dpdt = deriv_p(data, norm=1.)
    assert dpdt == pytest.approx(result)


def test_max_step_smaller_when_more_accurate():
    t1 = maximal_step_at_power(dying_u235(), t=inf, p0=1, atol=0., rtol=1e-2)
    t2 = maximal_step_at_power(dying_u235(), t=inf, p0=1, atol=0., rtol=1e-3)
    assert t2 <= t1


def test_max_step_smaller_when_higher_power():
    t1 = maximal_step_at_power(dying_u235(), t=inf, p0=1, atol=0., rtol=1e-2)
    t2 = maximal_step_at_power(dying_u235(), t=inf, p0=2, atol=0., rtol=1e-2)
    assert t2 <= t1


timesteps = st.timedeltas(min_value=timedelta(days=0.5),
                          max_value=timedelta(days=30))


@settings(max_examples=700)
@given(t=timesteps)
def test_last_step_is_well_divided(t: timedelta):
    p0 = 0.001
    rtol, atol = 1e-2, 1e-16
    nudge = 1. - 1e-2
    data = dying_u235()
    dpdt = deriv_p(data, norm=p0/data.power(norm=1.))
    step = maximal_step_at_power(data, t.total_seconds(), p0,
                                 atol=atol, rtol=rtol, nudge=nudge,
                                 too_small=0.)
    timestep = timedelta(seconds=step)
    power_allowed_step = timedelta(seconds=_allowed_predictor_step(
        p0, dpdt, rtol=rtol, atol=atol, nudge=nudge))
    if t < power_allowed_step:
        assert t == timestep
    elif power_allowed_step < t < 2*power_allowed_step:
        assert timestep.total_seconds() == pytest.approx(t.total_seconds()/2)
    elif t > 2*power_allowed_step:
        assert timestep < t
        assert timestep.total_seconds() == pytest.approx(
            power_allowed_step.total_seconds())


thresholds = st.shared(st.timedeltas(min_value=timedelta(hours=1),
                                     max_value=timedelta(days=1)),
                       key="threshold")
warn_timesteps = thresholds.flatmap(
    lambda x: st.timedeltas(min_value=timedelta(seconds=1),
                            max_value=x-timedelta(minutes=1))
    )


@settings(max_examples=700)
@given(t=warn_timesteps, small=thresholds)
def test_small_step_warns(t: timedelta, small: timedelta):
    assert t < small
    data = dying_u235()
    p0 = 0.001
    nudge = 1. - 1e-2
    with pytest.warns(UserWarning):
        maximal_step_at_power(data, t.total_seconds(), p0,
                              atol=p0, rtol=1., nudge=nudge,
                              too_small=small.total_seconds())
