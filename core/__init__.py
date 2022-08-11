from enum import Enum, auto


class GpuMode(Enum):
    AUTO = auto()
    FULL = auto()
    NONE = auto()
    KERNEL = auto()


def get_best_gpu_mode(model):
    """ Returns the best gpu mode with respect to the model that is to be run, gpu resources, ...
    TODO
    :return: GpuMode
    """
    from ..core.models.abstract_statistical_model import AbstractStatisticalModel
    from ..core.models.longitudinal_atlas import LongitudinalAtlas
    assert isinstance(model, AbstractStatisticalModel)

    gpu_mode = GpuMode.FULL
    if model.name == "LongitudinalAtlas":
        gpu_mode = GpuMode.KERNEL

    return gpu_mode
