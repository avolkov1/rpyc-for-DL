#!/usr/bin/env python
'''
Implementing another approach to calling Tensorflow code via RPyC. The TF code
is encapsulated in a function that's pickled/transported to the RPC server.

Compare this implementation to the tfdocker_client approach. In the other
approach one basically passes a command line to be run on the RPC server.


Timing test:
time bash -c '
    python tfrpyc_client.py & python tfrpyc_client.py &
    python tfrpyc_client.py & python tfrpyc_client.py &
    python tfrpyc_client.py & python tfrpyc_client.py & wait'

'''
import os
import sys
import shlex
# from random import randint

import dill

from rpycdl_lib.common import mkdir_p

from cli_clients_common import parser_, get_server_connection

_topdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tfclient_querygpus_code(gpu_opts=None):
    '''Query GPUs. Demonstrate how to run remote tensorflow code that's self
    contained in a function.

    :param gpu_opts: Dictionary to pass to tf.GPUOptions(**gpu_opts)
    :type gpu_opts: dict

    '''
    import tensorflow as tf
    gpu_opts = {} if gpu_opts is None else \
        {kk: vv for kk, vv in gpu_opts.items()}

    # print('GPUOPTS: {}'.format(gpu_opts))
    gpu_options = tf.GPUOptions(**gpu_opts)
    config = tf.ConfigProto(gpu_options=gpu_options)
    # Start server with configs such as gpu memory limits, etc.
    # server = \
    tf.train.Server.create_local_server(config=config, start=True)
    # config = server.server_def.default_session_config
    # sess = tf.Session(target=server.target, config=config)

    print('===========================\nRUNNING GPUS LIST QUERY\n'
          '===========================')

    from tensorflow.python.client import device_lib

    def get_available_gpus(device_lib, ngpus=-1):
        '''
        :param int ngpus: GPUs max to use. Default -1 means all gpus.
        :returns: List of gpu devices. Ex.: ['/gpu:0', '/gpu:1', ...]
        '''
        local_device_protos = device_lib.list_local_devices()
        gpus_list = [x.name for x in local_device_protos
                     if x.device_type == 'GPU']
        return gpus_list[:ngpus] if ngpus > -1 else gpus_list

    print('TF version: {}'.format(tf.__version__))  # @UndefinedVariable
    gpus = get_available_gpus(device_lib)
    print('GPUS: {}'.format(gpus))

    print('===========================\nFINISHED GPUS LIST QUERY\n'
          '===========================')


def tfclient_mnist_code(args_dict):
    '''Run a somewhat non-trivial piece of Tensorflow code. The code is in a
    module "mnist_deep.py". This module is imported and its main method is
    invoked.
    '''
    print('===========================\nRUNNING MNIST\n'
          '===========================')

    # Test for DEADLOCK
    # import time
    # time.sleep(20)

    import sys  # @Reimport
    code_path = args_dict['code_path']
    sys.path.insert(1, code_path)

    import mnist_deep  # @UnresolvedImport

    argv = args_dict['argv']
    mnist_deep.main(argv)

    print('===========================\nFINISHED MNIST\n'
          '===========================')


def tfclient_cifar_code(args_dict):
    code_path = args_dict['code_path']
    code_args = args_dict['code_args']

    import subprocess

    cmd = '{} {}'.format(code_path, code_args)
    popen = subprocess.Popen(
        cmd, shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)

    for stdout_line in iter(popen.stdout.readline, ""):
        print stdout_line.strip()


def main(argv=None):
    desc = '{}{}'.format(__doc__, main.__doc__)
    argv = sys.argv if argv is None else sys.argv.extend(argv)
    # CLI parser
    args = parser_(desc)

    def get_tfserv():
        tfserv = get_server_connection('tensorflow')

        return tfserv

    # Query GPUs via Tensorflow -----------------------------------------------
    tfserv = get_tfserv()
    # gpu_opts = None
    gpu_opts = {'per_process_gpu_memory_fraction': 0.1}
    # ngpus = -1
    ngpus = args.ngpus

    tfcode = dill.dumps(tfclient_querygpus_code)
    tfserv.root.run_code(tfcode, code_args=gpu_opts, ngpus=ngpus,
                         stdout=sys.stdout, stderr=sys.stderr)
    tfserv.close()

    # Run mnist_deep training -------------------------------------------------
    tfserv = get_tfserv()
    gpu_opts = None

    examples_dir = os.path.join(_topdir, 'examples')

    datadir = '{}/data/mnist'.format(examples_dir)
    workdir = '{}/workdir/tensorflow/mnist/'.format(_topdir)
    mkdir_p(workdir)

    tfcode_mnist = dill.dumps(tfclient_mnist_code)
    argv = ' '.join(['--data_dir="{}"'.format(datadir),
                     '--workdir="{}"'.format(workdir)])
    code_path = '{}/tensorflow/mnist'.format(examples_dir)
    # codeargs = {'argv': None,  # will download to /tmp/...
    #             'code_path': code_path}
    codeargs = {'argv': shlex.split(argv),
                'code_path': code_path}
    tfserv.root.run_code(tfcode_mnist, codeargs, ngpus=1,
                         stdout=sys.stdout, stderr=sys.stderr)
    tfserv.close()

    # ------------------------------------------------------ Run cifar training
    tfcifar = dill.dumps(tfclient_cifar_code)
    tfserv = get_tfserv()
    # usage: cifar10_multi_gpu_train.py [-h]
    #     [--batch_size BATCH_SIZE] [--data_dir DATA_DIR]
    #     [--use_fp16 [USE_FP16]] [--nouse_fp16]
    #     [--train_dir TRAIN_DIR] [--max_steps MAX_STEPS]
    #     [--num_gpus NUM_GPUS]
    #     [--log_device_placement [LOG_DEVICE_PLACEMENT]]
    #     [--nolog_device_placement]
    codeargs = {}
    cifarcode = os.path.join(
        examples_dir, 'tensorflow/cifar_mgpu/cifar10_multi_gpu_train.py')
    codeargs['code_path'] = 'python {}'.format(cifarcode)
    datadir = '{}/data/cifar'.format(examples_dir)
    ngpus = args.ngpus
    max_steps = 3000
    codeargs['code_args'] = '--num_gpus {} --data_dir {} --max_steps {}'\
        .format(ngpus, datadir, max_steps)
    tfserv.root.run_code(tfcifar, codeargs, ngpus=ngpus,
                         stdout=sys.stdout, stderr=sys.stderr)
    tfserv.close()


if __name__ == "__main__":
    main()
