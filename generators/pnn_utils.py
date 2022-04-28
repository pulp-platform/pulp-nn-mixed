from typing import Optional, Literal
from dataclasses import dataclass
from collections import OrderedDict

from mako.template import Template
import torch

# ISAs supported by PULP-NN-mixed kernels
_PULPNN_ISAS = ['XpulpV2',
                'XpulpNN',
                'XpulpNN-mixed']

# dataclass describing the quantization of a tensor
@dataclass
class PNNDataFormat:
    n_bits : int
    signed : bool

    @property
    def clip_hi(self):
        if signed:
            return 2**(self.n_bits-1) - 1
        return 2**self.n_bits - 1

    @property
    def clip_lo(self):
        if signed:
            return -(self.clip_hi+1)
        return 0


# TODO add this https://pypi.org/project/enforce-typing/

@dataclass
class PNNVersion:
    isa : Literal[*_PULPNN_ISAS]
    act_bw : Literal[32, 64]

# dataclass to describe a specific tensor's shape and data format
@dataclass
class PNNTensorSpec:
    shape : torch.Size
    data_format : PNNDataFormat

# class describing a function's C signature with utilities to pretty-print the
# signature and calls to the function. parameters and datatypes are stored in
# an OrderedDict
# rt is the return type of the function.
# attribute is an arbitrary string that will be added to the signature (e.g., "__attribute__((noinline))")
# the "print_call" method takes a "binding" dict, where each variable name in the
# signature is bound to a string which is assumed to be a variable name or
# literal value
class PNNSignature:
    def __init__(self, name : str, rt : str, attribute : str = "", params : Optional[OrderedDict] = None):
        assert len(name), "PNNSignature: name can not be empty!"
        assert len(rt), "PNNSignature: return type can not be empty!"
        self.name = name
        self.rt = rt
        self.attribute = attribute
        if params is None:
            params = OrderedDict()
        self.params = params
        self.t = Template(filename='./templates/signature.t', strict_undefined=True)

    # add a parameter to the params ODict
    def add_param(self, n : str, t : str):
        self.params[n] = t

    def print_signature(self, indent : int = 0):
        return self.t.render(sig=self, binding=None, indent=indent)

    def print_call(self, binding : dict, indent : int = 0):
        return self.t.render(sig=self, binding=binding, indent=indent)

    @property
    def n_params(self):
        return len(self.params)


