"""Objects for how the reactions, decays and depletion in general are modeled.

These objects are mostly used to generate matrices for the mathematical problems
in question, such as the Bateman matrices, the energy generation model and so
on.

"""

from .reactionmodel import ReactionModel, fiss_arr_gen
from .decaymodel import DecayModel
from .depletionmodel import DepletionModel, depletion_model
