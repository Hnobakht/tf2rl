"""Compatibility imports; implementation lives in ``tf2rl.networks.layers``."""
from tf2rl.networks.layers import (
    L1L2_EQL as L1L2_EQL,
    DenseLayer as DenseLayer,
)
from tf2rl.networks.layers import __all__ as __all__
from tf2rl.networks.layers import EqlLayerv2 as EqlLayerv2, DenseLayerv2 as DenseLayerv2
from tf2rl.networks.layers import NoisyDense as NoisyDense, SNDense as SNDense
