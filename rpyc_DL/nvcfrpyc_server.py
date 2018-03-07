#!/usr/bin/env python
'''
For this service to self register must start the registry client prior to
starting this service.


'''
import sys
import logging

import shlex
import subprocess

import rpyc

from cli_servers_common import parser_

from rpycdl_lib.common import get_logger
from rpycdl_lib.dlrpyc_services import DLServiceMixin, server_builder


class NVCaffeService(DLServiceMixin, rpyc.Service):
    '''Caffe RPyC Service'''
    # def __init__(self, *args, **kwargs):
    #     rpyc.Service.__init__(self, *args, **kwargs)


def main(argv=None):
    '''Run NVCaffe service over RPyC. Launch this service with NVCaffe
istalled on the system. If using containers, then launch this service from
within the container.
    '''
    desc = '{}{}'.format(__doc__, main.__doc__)
    argv = sys.argv if argv is None else sys.argv.extend(argv)
    # CLI parser
    args = parser_(desc)

    debug_flag = args.debug
    gpus_on_system = args.ngpus

    # import caffe as cf  # @UnresolvedImport
    # cfserv_alias = 'Caffe-{}'.format(cf.__version__)  # @UndefinedVariable

    cfver = subprocess.check_output(shlex.split('caffe --version')).split()[-1]

    cfserv_alias = 'NVCaffe-{}'.format(cfver)
    NVCaffeService.ALIASES = ['NVCaffe', cfserv_alias]

    log = None
    if debug_flag:
        # print('DEBUGGING TF')
        log = get_logger(__file__, level=logging.DEBUG)

    server = server_builder(
        NVCaffeService, log_obj=log, gpus_on_system=gpus_on_system)

    server.start()


if __name__ == "__main__":
    main()
