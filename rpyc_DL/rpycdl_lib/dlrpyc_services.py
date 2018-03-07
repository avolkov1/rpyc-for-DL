'''
'''
import os
import sys
import logging

# import time
# from random import randint

import dill

from multiprocessing import active_children

from rpycdl_lib.common import get_logger
from rpycdl_lib.gpu_rmgr import GPURmgr

__all__ = ('DLServiceMixin', 'server_builder',)


class DLServiceMixin(object):
    _gpus_on_system = None

    @classmethod
    def set_log(cls, log):
        cls._log = log

    @classmethod
    def set_gpus_on_system(cls, ngpus):
        '''This will be used only if cannot autodetect GPUs on the system.
        Passed through to :class:`GPURmgr`.
        '''
        cls._gpus_on_system = ngpus

    def on_connect(self):
        '''Code that runs when a connection is created to init the service.
        According to documentation:
            Try to avoid overriding the __init__ method of the service. Place
            all initialization-related code in on_connect.
        '''
        # refer to: rpyc/core/protocol.py for various config options
        # self._conn._config.update(dict(
        #     # allow_all_attrs=True,
        #     # allow_pickle=True,
        #     # allow_getattr=True,
        #     # allow_setattr=False,
        #     # allow_delattr=False,
        #     # import_custom_exceptions=True,
        #     # instantiate_custom_exceptions=True,
        #     # instantiate_oldstyle_exceptions=True,
        # ))

        # https://stackoverflow.com/questions/13247956/python-django-spawn-background-process-and-avoid-zombie-process @IgnorePep8
        # cleanup spawn background process and avoid zombie process
        active_children()

        try:
            # If the logger was set via classmethod set_log
            self._log = self._log
        except AttributeError:
            log = get_logger(__file__, level=logging.INFO)
            # log = get_logger(__file__, level=logging.DEBUG)
            self._log = log

        log = self._log

        pid = os.getpid()
        log.debug('STARTING PROCESS: {} SERVICE: {}'.format(
            pid, self.get_service_name()))

    def exposed_run_code(
            self,
            code_callback=None,
            code_args=None,
            ngpus=1,
            stdout=None,
            stderr=None):
        '''Run some client code. The code has to be pickled using dill package.
        Sets CUDA_VISIBLE_DEVICES and NV_GPU environment variables according
        to ngpus and whatever set_gpus_on_system was set to (default assumes
        4 GPUs on the system).

        The GPUs are managed via :class:`rpycdl_lib.gpu_rmgr.GPURmgr` class.

        :param code_callback: Serialized code via dill.dumps(some_function)
        :type code_callback: str

        :param code_args: Argument to pass to the code_callback. Keep these
            simple. The attributes of the code_args should return simple types
            like numbers, strings, maybe tuples, lists and dicts will work.

        :param ngpus: Number of GPUs to use. -1 means all GPUs. Default is 1.
        :type int:

        :param stdout: Override sys.stdout with client stream.

        :param stderr: Override sys.stderr with client stream.

        '''
        sys.stdout = stdout if stdout is not None else sys.__stdout__
        sys.stderr = stderr if stderr is not None else sys.__stderr__

        log = self._log

        pid = os.getpid()
        log.debug('PROCESS: {} SERVICE: {} REQUIRES NGPUS={}'.format(
            pid, self.get_service_name(), ngpus))

        # time.sleep(randint(1, 10))  # Test for DEADLOCK

        # =====================================================================
        # Basic resource acquisition and locking
        # =====================================================================
        grmgr = GPURmgr(ngpus, log, self._gpus_on_system)
        # gpus_list = grmgr.acquire()
        grmgr.acquire()

        log.debug('RUNNING PROCESS: {} SERVICE: {}'.format(
            pid, self.get_service_name()))

        # =====================================================================
        # Orchestrating some Tensorflow code
        # =====================================================================
        cfcall = dill.loads(code_callback)
        cfcall(code_args)  # CALLING CLIENT CODE!!! =========================

        # =====================================================================
        # Release hardware resources
        # =====================================================================
        grmgr.release()

        log.debug('PROCESS: {} SERVICE: {} GPUS RELEASED'.format(
            pid, self.get_service_name()))

    def on_disconnect(self):
        '''Called when the connection had already terminated for cleanup
        (must not perform any IO on the connection)

        When using ForkingServer and Tensorflow, disconnecting like this will
        release whatever resources Tensorflow was using.
        '''
        pid = os.getpid()
        log = self._log
        log.debug('ENDING PROCESS: {} SERVICE: {}'.format(
            pid, self.get_service_name()))

        active_children()


def server_builder(service_cls, protocol_config=None, log_obj=None,
                   gpus_on_system=None):
    '''
    :param service_cls: A class derived from DLServiceMixin and rpyc.Service
    :type service_cls: DLServiceMixin
    '''
    from rpyc.utils.server import ForkingServer as Server

    # refer to: rpyc/core/protocol.py for various config options
    protocol_config_default = {
        # 'allow_pickle': True,
        # 'allow_public_attrs': True
    }

    protocol_config = protocol_config_default if protocol_config is None \
        else protocol_config

    rpyc_server = Server(
        service_cls, auto_register=True,
        protocol_config=protocol_config)

    if log_obj is not None:
        rpyc_server.service.set_log(log_obj)

    if gpus_on_system is not None:
        rpyc_server.service.set_gpus_on_system(gpus_on_system)

    return rpyc_server
