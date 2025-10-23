"""
T-Rep (Time-series Representation Learning) module for ChromaDB embeddings.

This module provides a ChromaDB-compatible embedding function using T-Rep,
a self-supervised representation learning method for time series.
"""

from .TRepEmbedding import TRepEmbedding
from .trep import TRep

__all__ = ['TRepEmbedding', 'TRep']
