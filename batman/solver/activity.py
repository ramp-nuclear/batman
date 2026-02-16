"""module to calculate activities"""
from typing import Dict

from toolz import merge_with

from batman.solver.inputs import EasyData
from batman.solver.utils import RunData
from batman.units import Bq
from isotopes import ZAID


def _get_single_activities(data: RunData) -> Dict[ZAID, Bq]:
    """Calculate activity in Becquerel per unstable isotope"""
    dep, mixture, vol = data
    zaids, decay_model, _ = dep
    return {zaid: -dec * nd * vol * 1e24 for i, zaid in enumerate(zaids) if
            (dec := decay_model.mat[i, i]) and (nd := mixture.get(zaid))}


def activities(data: EasyData) -> Dict[ZAID, Bq]:
    """Calculate activity in Becquerel per unstable isotope"""
    return merge_with(sum, *data.map(_get_single_activities))
