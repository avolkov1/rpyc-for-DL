'''
'''
import sys


import logging

import rpyc

from cli_servers_common import parser_

from rpycdl_lib.common import get_logger
from rpycdl_lib.dlrpyc_services import DLServiceMixin, server_builder


class DeepLearningService(DLServiceMixin, rpyc.Service):
    '''DeepLearning RPyC Service'''


def main(argv=None):
    '''Run DeepLearning service over RPyC.'''
    desc = '{}{}'.format(__doc__, main.__doc__)
    argv = sys.argv if argv is None else sys.argv.extend(argv)
    # CLI parser
    args = parser_(desc)

    debug_flag = args.debug
    gpus_on_system = args.ngpus

    log = None
    if debug_flag:
        # print('DEBUGGING TF')
        log = get_logger(__file__, level=logging.DEBUG)

    server = server_builder(
        DeepLearningService, log_obj=log, gpus_on_system=gpus_on_system)

    server.start()


if __name__ == "__main__":
    main()
