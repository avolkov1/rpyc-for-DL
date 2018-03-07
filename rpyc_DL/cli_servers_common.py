'''
'''
import argparse
from textwrap import dedent


__all__ = ('parser_',)


class CustomFormatter(argparse.RawDescriptionHelpFormatter):
    '''Convenience formatter_class for argparse help print out.'''


def parser_(desc):
    parser = argparse.ArgumentParser(description=dedent(desc),
                                     formatter_class=CustomFormatter)

    parser.add_argument(
        '--ngpus', type=int, default=None,
        help='Number of GPUs on the system. Default auto detect if nvml is '
        'setup. Otherwise set this manually. GPURmgr will set to 4 if cannot '
        'autodetect and not set manually.')

    parser.add_argument(
        '--debug', action='store_true', default=False,
        help='Have the logger print debug messages. Default: %(default)s')

    args = parser.parse_args()

    return args
