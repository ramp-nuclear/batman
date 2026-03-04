"""Module for defining type variables and the like

"""
from typing import TypeVar

import networkx as nx

BatmanGraph = TypeVar('BatmanGraph', bound=nx.MultiDiGraph)
