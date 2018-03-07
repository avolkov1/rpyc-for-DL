'''
'''
import os
import sys

import dill

from rpycdl_lib.common import mkdir_p

from cli_clients_common import parser_, get_server_connection

_topdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def nvcfclient_mnist_code(args_dict):
    '''Client code must be self contained. It's going to run remotely.
    '''
    import os  # @Reimport
    import subprocess
    import shlex
    import shutil
    import errno

    from google.protobuf import text_format  # @UnresolvedImport
    from caffe.proto import caffe_pb2  # @UnresolvedImport
    import caffe  # @UnresolvedImport

    def mkdir_p(path):
        """ 'mkdir -p' in Python """
        try:
            os.makedirs(path)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                raise

    netp = caffe_pb2.NetParameter()

    codedir = args_dict['codedir']
    datadir = args_dict['datadir']
    workdir = args_dict['workdir']
    os.chdir(workdir)
    ngpus = args_dict['ngpus']

    # print('CODEDIR: {}'.format(codedir))
    # print('WORKDIR: {}'.format(workdir))
    # print('CWD: {}'.format(os.getcwd()))

    mnist_dst = os.path.join(workdir, 'mnist')
    mkdir_p(mnist_dst)

    lenet_file_src = '{}/mnist/lenet_train_test.prototxt.tmpl'.format(codedir)
    with open(lenet_file_src, 'r') as fo:
        text_format.Parse(fo.read(), netp)

    for ll in netp.layer:
        # print ll
        if ll.name == "mnist" and ll.type == "Data":
            # print 'setting path'
            if ll.include[0].phase == caffe.TRAIN:
                ll.data_param.source = '{}/mnist_test_lmdb'.format(datadir)
                ll.data_param.batch_size = ll.data_param.batch_size * ngpus
            if ll.include[0].phase == caffe.TEST:
                ll.data_param.source = '{}/mnist_train_lmdb'.format(datadir)

    lenet_file = '{}/mnist/lenet_train_test.prototxt'.format(workdir)
    with open(lenet_file, 'w') as fo:
        fo.write(str(netp))

    solver_file_src = '{}/mnist/lenet_solver.prototxt'.format(codedir)
    solver_file = '{}/mnist/lenet_solver.prototxt'.format(workdir)
    shutil.copy(solver_file_src, solver_file)

    # Works, but cannot redirect stdout/stderr nor run multigpu.
    # caffe.set_mode_gpu()
    # solver = caffe.get_solver('mnist/lenet_solver.prototxt')
    # solver.solve()

    visible_devices = ','.join((str(kk) for kk in range(ngpus)))
    gpus_cmd = '-gpu {}'.format(visible_devices) if ngpus > 1 else ''
    caffe_train_cmd = \
        'caffe train --solver=mnist/lenet_solver.prototxt {gpus_cmd}'\
        .format(gpus_cmd=gpus_cmd)
    print('RUNNING COMMAND: {}'.format(caffe_train_cmd))
    popen = subprocess.Popen(
        shlex.split(caffe_train_cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)

    for stdout_line in iter(popen.stdout.readline, ""):
        print stdout_line.strip()


def main(argv=None):
    desc = '{}{}'.format(__doc__, main.__doc__)
    argv = sys.argv if argv is None else sys.argv.extend(argv)
    # CLI parser
    args = parser_(desc)

    # ----------------------------------------------- Run MNIST solver on Caffe
    cfserv = get_server_connection('nvcaffe')
    # ngpus = -1
    ngpus = args.ngpus

    cfcode_mnist = dill.dumps(nvcfclient_mnist_code)

    examples_dir = os.path.join(_topdir, 'examples')
    codedir = '{}/caffe'.format(examples_dir)
    datadir = '{}/data/mnist'.format(examples_dir)
    workdir = '{}/workdir/caffe'.format(_topdir)
    mkdir_p(workdir)
    codeargs = {
        'codedir': codedir,
        'datadir': datadir,
        'workdir': workdir,
        'ngpus': ngpus
    }

    cfserv.root.run_code(cfcode_mnist, code_args=codeargs, ngpus=ngpus,
                         stdout=sys.stdout)
    cfserv.close()


if __name__ == "__main__":
    main()
