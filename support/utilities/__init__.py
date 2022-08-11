import torch
import torch.multiprocessing as mp
import numpy as np

from ...core import GpuMode

import logging
logger = logging.getLogger(__name__)


def get_torch_scalar_type(dtype):
    return {'float16': torch.HalfTensor,
            'torch.float16': torch.HalfTensor,
            'float32': torch.FloatTensor,
            'torch.float32': torch.FloatTensor,
            'float64': torch.DoubleTensor,
            'torch.float64': torch.DoubleTensor}[dtype]


def get_torch_integer_type(dtype):
    """
    Note:
    'float32' is forced to torch LongTensors because of the following error: "RuntimeError: tensors used as indices must be long or byte tensors"
    """

    return {'uint8': torch.ByteTensor,
            'torch.uint8': torch.ByteTensor,
            'int8': torch.CharTensor,
            'torch.int8': torch.CharTensor,
            'float16': torch.ShortTensor,
            'torch.float16': torch.ShortTensor,
            'float32': torch.LongTensor,        # IntTensor
            'torch.float32': torch.LongTensor,  # IntTensor
            'float64': torch.LongTensor,
            'torch.float64': torch.LongTensor}[dtype]


def get_torch_dtype(t):
    """
    32-bit floating point	torch.float32 or torch.float	torch.FloatTensor	torch.cuda.FloatTensor
    64-bit floating point	torch.float64 or torch.double	torch.DoubleTensor	torch.cuda.DoubleTensor
    16-bit floating point	torch.float16 or torch.half	torch.HalfTensor	torch.cuda.HalfTensor
    8-bit integer (unsigned)	torch.uint8	torch.ByteTensor	torch.cuda.ByteTensor
    8-bit integer (signed)	torch.int8	torch.CharTensor	torch.cuda.CharTensor
    16-bit integer (signed)	torch.int16 or torch.short	torch.ShortTensor	torch.cuda.ShortTensor
    32-bit integer (signed)	torch.int32 or torch.int	torch.IntTensor	torch.cuda.IntTensor
    64-bit integer (signed)	torch.int64 or torch.long	torch.LongTensor	torch.cuda.LongTensor

    cf: https://pytorch.org/docs/stable/tensors.html
    :param t:
    :return:
    """
    if t in ['float32', np.float32]:
        return torch.float32
    if t in ['float64', np.float64]:
        return torch.float64
    if t in ['float16', np.float16]:
        return torch.float16
    if t in ['uint8', np.uint8]:
        return torch.uint8
    if t in ['int8', np.int8]:
        return torch.int8
    if t in ['int16', np.int16]:
        return torch.int16
    if t in ['int32', np.int32]:
        return torch.int32
    if t in ['int64', np.int64]:
        return torch.int64

    assert t in [torch.float32, torch.float64, torch.float16, torch.uint8,
                 torch.int8, torch.int16, torch.int32, torch.int64], 'dtype=' + t + ' was not recognized'
    return t


def move_data(data, device='cpu', dtype=None, requires_grad=None):
    """
    Move given data to target Torch Tensor to the given device
    :param data:    TODO
    :param device:  TODO
    :param requires_grad: None: do nothing, True: set requires_grad flag, False: unset requires_grad flag (detach)
    :param dtype:  dtype that the returned tensor should be formatted as.
                   If not defined, the same dtype as the input data will be used.
    :return: Torch.Tensor
    """
    assert data is not None, 'given input data cannot be None !'
    assert device is not None, 'given input device cannot be None !'

    if dtype is None:
        dtype = get_torch_dtype(data.dtype)

    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).type(dtype)
    elif isinstance(data, list):
        data = torch.tensor(data, dtype=dtype, device=device)

    assert isinstance(data, torch.Tensor), 'Expecting Torch.Tensor instance not ' + str(type(data))

    # move data to device. Note: tensor.to() does not move if data is already on target device
    data = data.type(dtype).to(device=device)

    # handle requires_grad flag
    if requires_grad is not None and requires_grad:
        # user wants grad
        if data.requires_grad is False:
            data.requires_grad_()
        # else data already has requires_grad flag to True
    elif requires_grad is not None:
        # user does not want grad
        if data.requires_grad is True:
            data.detach_()
        # else data already has requires_grad flag to False

    return data


def convert_deformable_object_to_torch(deformable_object, device='cpu'):
    from core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
    # bounding_box
    assert isinstance(deformable_object, DeformableMultiObject)
    assert deformable_object.bounding_box is not None
    if not isinstance(deformable_object.bounding_box, torch.Tensor):
        deformable_object.bounding_box = move_data(deformable_object.bounding_box, device=device)
    deformable_object.bounding_box = deformable_object.bounding_box.to(device)

    # object_list
    for i, _ in enumerate(deformable_object.object_list):
        if hasattr(deformable_object.object_list[i], 'bounding_box'):
            deformable_object.object_list[i].bounding_box = move_data(deformable_object.object_list[i].bounding_box, device=device)

        if hasattr(deformable_object.object_list[i], 'connectivity'):
            deformable_object.object_list[i].connectivity = move_data(deformable_object.object_list[i].connectivity, device=device)

        if hasattr(deformable_object.object_list[i], 'points'):
            deformable_object.object_list[i].points = move_data(deformable_object.object_list[i].points, device=device)

        if hasattr(deformable_object.object_list[i], 'affine'):
            deformable_object.object_list[i].affine = move_data(deformable_object.object_list[i].affine, device=device)

        if hasattr(deformable_object.object_list[i], 'corner_points'):
            deformable_object.object_list[i].corner_points = move_data(deformable_object.object_list[i].corner_points, device=device)

        if hasattr(deformable_object.object_list[i], 'intensities'):
            deformable_object.object_list[i].intensities = move_data(deformable_object.object_list[i].intensities, device=device)

    return deformable_object


def get_device_from_string(device):

    if isinstance(device, str) and device.startswith('cuda') and ':' in device:
        device_type, device_id = device.split(':')
        assert device_type == 'cuda'
        torch_device, device_id = torch.device(type=device_type, index=int(device_id)), int(device_id)
    elif isinstance(device, str) and device == 'cuda':
        torch_device, device_id = torch.device(type='cuda', index=0), 0
    elif isinstance(device, torch.device):
        torch_device, device_id = device, device.index if device.index is not None else -1
    else:
        torch_device, device_id = torch.device(type='cpu', index=0), -1

    return torch_device, device_id


def get_best_device(gpu_mode=GpuMode.AUTO):
    """
    :return:    Best device. can be: 'cpu', 'cuda:0', 'cuda:1' ...
    """
    assert gpu_mode is not None
    assert isinstance(gpu_mode, GpuMode)
    use_cuda = False
    if gpu_mode in [GpuMode.AUTO]:
        # TODO this should be more clever
        use_cuda = torch.cuda.is_available()
    elif gpu_mode in [GpuMode.FULL]:
        use_cuda = True
        if not torch.cuda.is_available():
            use_cuda = False
    assert isinstance(use_cuda, bool)

    device_id = 0 if use_cuda else -1
    device = 'cuda:' + str(device_id) if use_cuda and torch.cuda.is_available() else 'cpu'

    if use_cuda and mp.current_process().name != 'MainProcess':
        '''
        PoolWorker-1 will use cuda:0
        PoolWorker-2 will use cuda:1
        PoolWorker-3 will use cuda:2
        etc...
        '''
        # TODO: Use GPUtil to check if GPU memory is full
        # TODO: only use CPU if mp queue is still quite full (eg > 50%), else leave work for GPU

        process_id = int(mp.current_process().name.split('-')[1])   # eg: PoolWorker-0
        device_id = process_id % torch.cuda.device_count()
        device = 'cuda:' + str(device_id)

        # for device_id in range(torch.cuda.device_count()):
        #     # pool_worker_id = device_id
        #     if process_id < process_per_gpu:
        #         device = 'cuda:' + str(device_id)
        #         break
        #         # if mp.current_process().name == 'PoolWorker-' + str(pool_worker_id):
        #         #     device = 'cuda:' + str(device_id)
        #         #     break

        # try:
        #     device_id = GPUtil.getFirstAvailable(order='first', maxMemory=0.5, attempts=1, verbose=False)[0]
        #     device = 'cuda:' + str(device_id)
        #
        #     # for device_id in range(torch.cuda.device_count()):
        #     #     pool_worker_id = device_id
        #     #     if mp.current_process().name == 'PoolWorker-' + str(pool_worker_id):
        #     #         device = 'cuda:' + str(device_id)
        #     #         break
        #
        # except RuntimeError as e:
        #     # if no device is available
        #     logger.info(e)
        #     pass
    # elif torch.cuda.is_available() and mp.current_process().name == 'MainProcess':
    #     device_id = 0
    #     device = 'cuda:' + str(device_id)
    #     device_id = -1
    #     device = 'cpu'

    # logger.info("get_best_device is " + device)
    return device, device_id
    # return 'cuda:1', 1


def adni_extract_from_file_name(file_name):
    import re
    # file_name = 'sub-ADNI002S0729_ses-M06.vtk'
    m = re.search('\Asub-ADNI(.+?)_ses-M(.+?).vtk', file_name)
    if m:
        assert len(m.groups()) == 2
        subject_id = m.group(1)
        visit_age = m.group(2)
        return subject_id, visit_age
    else:
        raise LookupError('could not extract id and age from ' + file_name)


def longitudinal_extract_from_file_name(file_name):
    import re
    # file_name = 's0041_7110_0.nii'
    m = re.search('\As(.+?)_(.+?)_(.+?).nii', file_name)
    if m:
        assert len(m.groups()) == 3
        subject_id = m.group(1)
        visit_age = float(m.group(2))/100.0
        visit_id = int(m.group(3))
        return subject_id, visit_age, visit_id
    else:
        raise LookupError('could not extract id and age from ' + file_name)


def has_hyperthreading():
    import psutil

    try:
        c1 = psutil.cpu_count(logical=False)
        c2 = psutil.cpu_count(logical=True)
        if c1 is None or c2 is None:
            raise RuntimeError('ERROR')
        return c1 != c2
    except:
        pass

    return False
