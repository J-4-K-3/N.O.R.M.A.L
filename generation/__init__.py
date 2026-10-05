"""Generation package for N.O.R.M.A.L.

Provides simplified generator, indexer, node orchestration and external connectors.
This is a lightweight scaffold intended to be extended with real embedding
and generation backends later.
"""

from .generator import generate_text
from .indexer import SimpleIndexer
from .nodes import NodeManager
from .external import ExternalConnector

__all__ = ["generate_text", "SimpleIndexer", "NodeManager", "ExternalConnector"]
