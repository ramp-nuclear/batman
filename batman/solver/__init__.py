"""Tools for solving the bateman's equation. This is most of the outer API.

"""
from .activity import activities
from .inputs import make_run_data, EasyData, InputData, SerialEasyData
from .inputs_dist import DistEasyData
from .k_est import calculate_loss_factor
from .reach_k import step_desired_k_at_power
from .solve import *
from .time_est import max_step_initial_correct_predictor as predictor_time_guess
