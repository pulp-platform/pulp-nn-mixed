from pathlib import Path
from copy import copy, deepcopy
from pnn_utils import PNNDataFormat

#parent class for all kernels.
# contains utilities to:
# - generate kernel code 
# 
class PNNKernel:
    def __init__(self, version : PNNVersion, out_folder : str, in_format : Union[list, PNNDataFormat], out_format : Optional[PNNDataFormat] = None, wt_format : Optional[PNNDataFormat] = None, **kwargs):
        self.version = version
        if not isinstance(in_format, list):
            in_format = [in_format]
        self.in_format = in_format
        self.out_format = out_format
        self.wt_format = wt_format
        self.out_root = Path(f"../{version.isa}/{version.act_bw}bit")
        self.out_folder = out_folder
        self.__dict__.update(kwargs)

    @property
    def template_file(self) -> Path:
        raise NotImplementedError

    @property
    def out_filename(self) -> str:
        raise NotImplementedError

    @property
    def out_path(self) -> str:
        return self.out_root.joinpath(self.out_folder).joinpath(self.out_filename)

    @property
    def signature(self) -> PNNSignature:
        raise NotImplementedError

    def print_signature(self, indent : int = 0) -> str:
        return self.signature.print_signature(indent=indent)

    def print_call(self, binding : dict, indent : int = 0) -> str:
        return self.signature.print_call(binding=binding, indent=indent)




