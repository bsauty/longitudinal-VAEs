import logging
import os
import time
import torch
from abc import abstractmethod

import torch.multiprocessing as mp

from ...core import default

logger = logging.getLogger(__name__)

# used as a global variable when processes are initially started.
process_initial_data = None


def _initializer(*args):
    """
    Process initializer function that is called when mp.Pool is started.
    :param args:    arguments that are to be copied to the target process. This can be a tuple for convenience.
    """
    global process_initial_data
    process_id, process_initial_data = args

    assert 'OMP_NUM_THREADS' in os.environ
    torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))

    # manually set process name
    with process_id.get_lock():
        mp.current_process().name = 'PoolWorker-' + str(process_id.value)
        logger.info('pid=' + str(os.getpid()) + ' : ' + mp.current_process().name)

        process_id.value += 1


class AbstractStatisticalModel:
    """
    AbstractStatisticalModel object class.
    A statistical model is a generative function, which tries to explain an observed stochastic process.
    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, name='undefined', number_of_processes=default.number_of_processes, gpu_mode=default.gpu_mode):
        self.name = name
        self.fixed_effects = {}
        self.priors = {}
        self.population_random_effects = {}
        self.individual_random_effects = {}
        self.has_maximization_procedure = None

        self.number_of_processes = number_of_processes
        self.gpu_mode = gpu_mode
        self.pool = None

    @abstractmethod
    def get_fixed_effects(self):
        raise NotImplementedError

    @abstractmethod
    def setup_multiprocess_pool(self, dataset):
        raise NotImplementedError

    def _setup_multiprocess_pool(self, initargs=()):
        if self.number_of_processes > 1:
            logger.info('Starting multiprocess using ' + str(self.number_of_processes) + ' processes')
            assert len(mp.active_children()) == 0, 'This should not happen. Has the cleanup() method been called ?'
            start = time.perf_counter()
            process_id = mp.Value('i', 0, lock=True)    # shared between processes
            initargs = (process_id, initargs)

            self.pool = mp.Pool(processes=self.number_of_processes, maxtasksperchild=None,
                                initializer=_initializer, initargs=initargs)
            logger.info('Multiprocess pool started using start method "' + mp.get_sharing_strategy() + '"' +
                        ' in: ' + str(time.perf_counter()-start) + ' seconds')

            if torch.cuda.is_available() and self.number_of_processes > torch.cuda.device_count():
                logger.warning("You are trying to run more processes than there are available GPUs, "
                               "it is advised to run `nvidia-cuda-mps-control` to leverage concurrent cuda executions. "
                               "If run in background mode, don't forget to stop the daemon when done.")

    def _cleanup_multiprocess_pool(self):
        if self.pool is not None:
            self.pool.terminate()

    ####################################################################################################################
    ### Common methods, not necessarily useful for every model.
    ####################################################################################################################

    def cleanup(self):
        self._cleanup_multiprocess_pool()

    def clear_memory(self):
        pass

