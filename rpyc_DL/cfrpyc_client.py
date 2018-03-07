'''
'''
import os
import sys
import argparse
from textwrap import dedent

import dill

from cli_clients_common import get_server_connection

_topdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cfclient_mnist_code(args_dict):
    '''Client code must be self contained. It's going to run remotely.
    This versoin only works with bvlc/caffe:gpu code. WILL NOT WORK WITH
    NVCAFFE.
    '''
    # re-import because client side code does not see the global context
    # of this module. Remember, client code has to be self-contained.
    import sys  # @Reimport

    # copying args like this sometimes prevents rpyc errors regarding brine???
    codeargs = {kk: vv for kk, vv in args_dict.items()}
    # for kk, vv in codeargs.items():
    #     print('kk {}; vv type {}'.format(kk, type(vv)))
    codelibdir = args_dict['codelibdir']
    timeout = args_dict['timeout']

    sys.path.insert(1, codelibdir)

    from pycaffe_mnist import main  # @UnresolvedImport

    print('If you don''t see "pycaffe_mnist DONE!!!!!!" then did not run '
          'successfuly.')
    main(codeargs, timeout=timeout)
    # main(codeargs)

    print('pycaffe_mnist DONE!!!!!!')


def cfclient_mnist_code_spawn(args_dict):
    code_path = args_dict['code_path']
    code_args = args_dict['code_args']

    print('If you don''t see "pycaffe_mnist DONE!!!!!!" then did not run '
          'successfuly.')

    import subprocess

    cmd = '{} {}'.format(code_path, code_args)
    print('RUNNING COMMAND: {}'.format(cmd))
    popen = subprocess.Popen(
        cmd, shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)

    for stdout_line in iter(popen.stdout.readline, ""):
        print stdout_line.strip()

    print('pycaffe_mnist DONE!!!!!!')


class CustomFormatter(argparse.RawDescriptionHelpFormatter):
    '''Convenience formatter_class for argparse help print out.'''


def parser_(desc):
    parser = argparse.ArgumentParser(description=dedent(desc),
                                     formatter_class=CustomFormatter)

    parser.add_argument(
        '--ngpus', type=int, default=2,
        help='Number of GPUs to use. Default: %(default)s')

    parser.add_argument(
        '--timeout', type=int, default=30,
        help='How long (sec.) to wait for process to finish. '
        'Default: %(default)s')

    args = parser.parse_args()

    return args


def main(argv=None):
    desc = '{}{}'.format(__doc__, main.__doc__)
    argv = sys.argv if argv is None else sys.argv.extend(argv)
    # CLI parser
    args = parser_(desc)

    # ----------------------------------------------- Run MNIST solver on Caffe
    # This approach seems very unreliable. Often hangs on the server side.
    # Using multiprocessing.Process directly inside a client funtion within
    # rpyc causes problems. The subprocess approach below which invokes python
    # code with multiprocessing.Process has no problems.
    # cfserv = get_cfserv()
    # # ngpus = -1
    # ngpus = args.ngpus
    # # print('NGPUS: {}'.format(ngpus))
    # timeout = args.timeout
    #
    # cfcode_mnist = dill.dumps(cfclient_mnist_code)
    #
    # # codelibdir = '{}/rpyc_DL'.format(_topdir)
    # examples_dir = os.path.join(_topdir, 'examples')
    # codedir = '{}/caffe'.format(examples_dir)
    # codelibdir = os.path.join(codedir, 'mnist')
    # datadir = '{}/data/mnist'.format(examples_dir)
    # workdir = '{}/workdir/caffe'.format(_topdir)
    # from rpycdl_lib.common import mkdir_p
    # mkdir_p(workdir)
    # codeargs = {
    #     'codelibdir': codelibdir,
    #     'codedir': codedir,
    #     'datadir': datadir,
    #     'workdir': workdir,
    #     'ngpus': ngpus,
    #     'timeout': timeout
    # }
    #
    # cfserv.root.run_code(cfcode_mnist, code_args=codeargs, ngpus=ngpus,
    #                      stdout=sys.stdout)
    # cfserv.close()

    # ----------------------------------------------- Run MNIST solver on Caffe
    cfserv = get_server_connection('caffe')
    # ngpus = -1
    ngpus = args.ngpus
    cfcode_mnist = dill.dumps(cfclient_mnist_code_spawn)

    codeargs = {}

    examples_dir = os.path.join(_topdir, 'examples')
    codedir = '{}/caffe'.format(examples_dir)
    code_path = 'python {}'.format(
        os.path.join(codedir, 'mnist/pycaffe_mnist.py'))

    codeargs['code_path'] = code_path
    codeargs['code_args'] = '--ngpus {}'.format(ngpus)
    cfserv.root.run_code(cfcode_mnist, codeargs, ngpus=ngpus,
                         stdout=sys.stdout, stderr=sys.stderr)
    cfserv.close()


if __name__ == "__main__":
    main()
