"""
NeuroBulla Pipeline Bodies (NBPB)
A Biologically-Inspired Supervisory Framework for Machine-Learning Pipelines
"""
__version__ = "0.1.0"
__author__ = "Francesco Bulla & Stephanie Ewelu" 
from .nucleus import NBPBNucleus
from .hooks.pytorch_hook import PyTorchHook

__all__ = ["NBPBNucleus", "PyTorchHook"]