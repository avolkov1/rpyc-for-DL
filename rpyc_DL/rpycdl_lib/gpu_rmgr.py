'''
'''
import os
import time
import logging
import warnings

import fasteners

from rpycdl_lib.common import get_logger


__all__ = ('GPURmgr',)


class GPURmgr(object):
    '''Very Basic GPU Resource manager.
    Resource management and job-scheduling really should be decoupled.
    A hypothetical implementation would be dlrpyc_server that itself
    starts services such as a tfrpyc_server, cafferpyc_server, etc.

    Requires a shared file-system between servers/services. This manager will
    use lock files in "/tmp/tmp_lock_*". It iterates through system GPUs and
    tries to acquire available GPUs per ngpus requested. Then sets
    CUDA_VISIBLE_DEVICES and NV_GPU, and relies on the services to adhere to
    this environment variables for the GPUs that get used.
    '''
    def __init__(self, ngpus, log=None, gpus_on_system=None):
        '''
        :param gpus_on_system: Only used if pynvml is not installed.
        '''
        log = get_logger(__file__, level=logging.INFO) if log is None else log

        self._log = log
        self._ngpus = ngpus

        try:
            import pynvml
            pynvml.nvmlInit()
            self._gpus_on_system = pynvml.nvmlDeviceGetCount()
            pynvml.nvmlShutdown()
        except Exception:
            self._gpus_on_system = gpus_on_system
        if self._gpus_on_system is None:
            warnings.warn(
                'UKNOWN NUMBER OF GPUS ON SYSTEM. INSTALL NVML AND PYNVML. '
                'SETTING TO 4', RuntimeWarning)
            self._gpus_on_system = 4

        self._lock_list = {}
        self._acquired = False

    def acquire(self):
        '''
        :returns: List of GPUs being used. An integer list. Can be used with
            CUDA_VISIBLE_DEVICES or maybe NV_GPU
        :rtype: list
        '''
        if self._acquired:
            raise RuntimeError('GPU Resources already acquired. Release '
                               'resources to acquire again.')

        pid = os.getpid()
        log = self._log
        ngpus = self._ngpus
        gpus_on_system = self._gpus_on_system

        # --------------------------------------------- Lock hardware resources

        # if running in container use options -v /tmp:/tmp for sharing
        lockfile_tmp = '/tmp/tmp_lock_gpu{}'

        if ngpus == -1:  # special case means all gpus
            ngpus = gpus_on_system

        if ngpus < 1 or ngpus > gpus_on_system:
            raise RuntimeError('GPU RESOURCES NOT AVAILABLED')

        # Using fasteners package to implement file-locking. Even if the client
        # code crashes resources will be released. The fasteners package is
        # smart. This logic should not deadlock but beware. Using lock_sys to
        # prevent deadlock.
        lock_list = {}
        lock_sys = fasteners.InterProcessLock('/tmp/tmp_lock_sysgpus')
        lock_sys.acquire(blocking=True)
        log.debug('PS: {}; SYS LOCKED'.format(pid))
        while True:
            for ig in range(gpus_on_system):
                if ig in lock_list:
                    continue

                lock = fasteners.InterProcessLock(lockfile_tmp.format(ig))
                # time.sleep(randint(1, 10))  # Test for DEADLOCK
                gotten = lock.acquire(blocking=False, delay=0)
                if gotten:
                    log.debug('PS: {}; GPU {} LOCKED'.format(pid, ig))
                    lock_list[ig] = lock

                if len(lock_list.keys()) == ngpus:
                    break

            if len(lock_list.keys()) == ngpus:
                break

            time.sleep(1)

        log.debug('PS: {}; SYS UNLOCKED'.format(pid))
        lock_sys.release()

        self._lock_list = lock_list
        self._acquired = True

        gpus_list = self._lock_list.keys()
        visible_devices = ','.join((str(kk) for kk in gpus_list))
        if visible_devices:
            # print visible_devices
            os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices
            os.environ['NV_GPU'] = visible_devices

        return gpus_list

    def release(self):
        # =====================================================================
        # Release hardware resources
        # =====================================================================
        lock_list = self._lock_list
        for ilock in lock_list.values():
            ilock.release()

        self._lock_list = {}

        self._acquired = False

        return True
