"""Adapters for the DuoRec implementation and checkpoints supplied by ReAd."""
from .adapter import load_duorec
from .recbole import RecBoleBackboneAdapter

__all__ = ['load_duorec', 'RecBoleBackboneAdapter']
