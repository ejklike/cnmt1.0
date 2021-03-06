""" Main entry point of the ONMT library """
from __future__ import division, print_function

import onmt.inputters
import onmt.models
import onmt.utils
import onmt.modules
from onmt.trainer import Trainer
import sys
import onmt.utils.optimizers
onmt.utils.optimizers.Optim = onmt.utils.optimizers.Optimizer
sys.modules["onmt.Optim"] = onmt.utils.optimizers

# For Flake
__all__ = [onmt.inputters, onmt.models, onmt.utils, onmt.modules, "Trainer"]

__version__ = "1.0.0"
