"""Tools for solving the bateman's equation. This is most of the outer API."""

from .activity import activities as activities
from .inputs import (
    make_run_data as make_run_data,
    EasyData as EasyData,
    InputData as InputData,
    SerialEasyData as SerialEasyData,
)
from .inputs_dist import DistEasyData as DistEasyData
from .k_est import calculate_loss_factor as calculate_loss_factor
from .reach_k import step_desired_k_at_power as step_desired_k_at_power
from .solve import *
from .time_est import max_step_initial_correct_predictor

predictor_time_guess = max_step_initial_correct_predictor
