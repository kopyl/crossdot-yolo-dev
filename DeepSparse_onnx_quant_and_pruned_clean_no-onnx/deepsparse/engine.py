from typing import Union, Any
import os
import subprocess
import sys
import json
import importlib.util


def get_neuralmagic_binaries_dir():
    nm_package_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(nm_package_dir, AVX_TYPE)



def init_deepsparse_lib():
    try:
        onnxruntime_neuralmagic_so_path = os.path.join(
            get_neuralmagic_binaries_dir(), "deepsparse_engine.so"
        )
        spec = importlib.util.spec_from_file_location(
            "deepsparse.{}.deepsparse_engine".format(AVX_TYPE),
            onnxruntime_neuralmagic_so_path,
        )
        engine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(engine)

        return engine
    except ImportError:
        raise ImportError(
            "Unable to import deepsparse engine binaries. "
            "Please contact support@neuralmagic.com"
        )



__all__ = [
    "Engine",
]


class _Memoize:
    def __init__(self, f):
        self.f = f
        self.memo = {}

    def __call__(self, *args):
        if args not in self.memo:
            self.memo[args] = self.f(*args)
        return self.memo[args]

class architecture(dict):

    def __init__(self, *args, **kwargs):
        super(architecture, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def __setattr__(self, name: str, value: Any):
        if name != "__dict__":
            raise AttributeError(
                "Neural Magic: Architecture: can't modify {} to {}".format(name, value)
            )
        else:
            super(architecture, self).__setattr__(name, value)

    def override_isa(self, value: str):
        object.__setattr__(self, "isa", value)

    @property
    def threads_per_socket(self):
        return self.threads_per_core * self.cores_per_socket

    @property
    def num_threads(self):
        return self.threads_per_socket * self.num_sockets

    @property
    def num_physical_cores(self):
        return self.cores_per_socket * self.num_sockets

    @property
    def num_available_physical_cores(self):
        return self.available_cores_per_socket * self.available_sockets
    


VALID_VECTOR_EXTENSIONS = {"avx2", "avx512", "neon", "sve"}



@_Memoize
def _parse_arch_bin():
    package_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(package_path, "arch.bin")

    error_msg = "Neural Magic: Encountered exception while trying to read arch.bin: {}"

    try:
        info_str = subprocess.check_output(file_path).decode("utf-8")
        return architecture(json.loads(info_str))

    except subprocess.CalledProcessError as ex:
        error = json.loads(ex.stdout)
        raise OSError(error_msg.format(error["error"]))

    except Exception as ex:
        raise OSError(error_msg.format(ex))


def cpu_architecture():
    if not sys.platform.startswith("linux"):
        raise OSError(
            "Neural Magic: Only Linux is supported, not '{}'.".format(sys.platform)
        )

    arch = _parse_arch_bin()
    isa_type_override = os.getenv("NM_ARCH", None)

    if isa_type_override and isa_type_override != arch.isa:
        print(
            "Neural Magic: Using env variable NM_ARCH={} for isa_type".format(
                isa_type_override
            )
        )
        if isa_type_override not in VALID_VECTOR_EXTENSIONS:
            raise OSError(
                (
                    "Neural Magic: Invalid instruction set '{}' must be " "one of {}."
                ).format(isa_type_override, ",".join(VALID_VECTOR_EXTENSIONS))
            )
        arch.override_isa(isa_type_override)

    if arch.isa not in VALID_VECTOR_EXTENSIONS:
        raise OSError(
            (
                "Neural Magic: Unable to determine instruction set '{}'. This system "
                "may be unsupported but to try, set NM_ARCH to one of {} to continue."
            ).format(arch.isa, ",".join(VALID_VECTOR_EXTENSIONS))
        )

    return arch



ARCH = cpu_architecture()
NUM_CORES = ARCH.num_available_physical_cores
NUM_STREAMS = 0
AVX_TYPE = ARCH.isa
VNNI = ARCH.vnni

LIB = init_deepsparse_lib()


def _validate_num_cores(num_cores: Union[None, int]) -> int:
    if not num_cores:
        num_cores = NUM_CORES

    if num_cores < 1:
        raise ValueError("num_cores must be greater than 0")

    return num_cores


class Engine(object):

    def __init__(
        self,
        model,
        batch_size: int = 1,
        num_cores: int = None,
        num_streams: int = 0,
    ):
        self._num_cores = _validate_num_cores(num_cores)
        self._eng_net = LIB.deepsparse_engine(
            model,
            batch_size,
            self._num_cores,
            num_streams,
            "single_stream",
            None,
        )

    def __call__(
        self,
        inp,
    ):
        return self._eng_net.execute_list_out(inp)