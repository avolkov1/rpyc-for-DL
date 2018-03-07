'''
'''
import argparse
from textwrap import dedent

import rpyc


__all__ = ('parser_', 'get_server_connection',)


class CustomFormatter(argparse.RawDescriptionHelpFormatter):
    '''Convenience formatter_class for argparse help print out.'''


def parser_(desc):
    parser = argparse.ArgumentParser(description=dedent(desc),
                                     formatter_class=CustomFormatter)

    parser.add_argument(
        '--ngpus', type=int, default=2,
        help='Number of GPUs to use. Default: %(default)s')

    args = parser.parse_args()

    return args


def get_server_connection(sevice_name, config=None):
    # refer to: rpyc/core/protocol.py for various config options
    config_default = {
        'allow_public_attrs': True,
        # 'allow_pickle': True,
        # 'allow_getattr': True
    }

    config = config_default if config is None else config
    # refer to: rpyc/core/protocol.py for various config options
    cserv = rpyc.connect_by_service(
        sevice_name,
        config=config)

    return cserv
