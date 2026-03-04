"""Objects for how the reactions, decays and depletion in general are modeled.

These objects are mostly used to generate matrices for the mathematical problems
in question, such as the Bateman matrices, the energy generation model and so
on.

"""

from .reactionmodel import ReactionModel as ReactionModel, fiss_arr_gen as fiss_arr_gen
from .decaymodel import DecayModel as DecayModel
from .depletionmodel import DepletionModel as DepletionModel, depletion_model as depletion_model
