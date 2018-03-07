'''
'''

import os  # @Reimport
import shutil
import errno

from multiprocessing import Process, active_children

_topdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


def mkdir_p(path):
    """ 'mkdir -p' in Python """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def main(args_dict, daemon_flag=True, timeout=None):
    workdir = args_dict['workdir']
    os.chdir(workdir)
    mnist_dst = os.path.join(workdir, 'mnist')
    mkdir_p(mnist_dst)

    # https://github.com/BVLC/caffe/issues/4715
    os.environ['GLOG_log_dir'] = mnist_dst
    # os.environ['GLOG_logtostderr'] = "1"
    import caffe  # @UnresolvedImport
    caffe.init_log()
    # caffe.set_mode_gpu()

    print('LOGS WILL APPEAR IN: {}'.format(workdir))

    from google.protobuf import text_format  # @UnresolvedImport
    from caffe.proto import caffe_pb2  # @UnresolvedImport

    netp = caffe_pb2.NetParameter()

    codedir = args_dict['codedir']
    datadir = args_dict['datadir']

    # print('CODEDIR: {}'.format(codedir))
    # print('WORKDIR: {}'.format(workdir))
    # print('WORKDIR: {}'.format(datadir))
    # print('CWD: {}'.format(os.getcwd()))

    lenet_file_src = '{}/mnist/lenet_train_test.prototxt.tmpl'.format(codedir)
    with open(lenet_file_src, 'r') as fo:
        text_format.Parse(fo.read(), netp)

    for ll in netp.layer:
        # print ll
        if ll.name == "mnist" and ll.type == "Data":
            # print 'setting path'
            if ll.include[0].phase == caffe.TRAIN:
                ll.data_param.source = '{}/mnist_test_lmdb'.format(datadir)
            if ll.include[0].phase == caffe.TEST:
                ll.data_param.source = '{}/mnist_train_lmdb'.format(datadir)

    lenet_file = '{}/mnist/lenet_train_test.prototxt'.format(workdir)
    with open(lenet_file, 'w') as fo:
        fo.write(str(netp))

    solver_file_src = '{}/mnist/lenet_solver.prototxt'.format(codedir)
    solver_file = '{}/mnist/lenet_solver.prototxt'.format(workdir)
    shutil.copy(solver_file_src, solver_file)

    ngpus = args_dict['ngpus']
    gpus = range(ngpus)

    # Latest bvlc/caffe:gpu has a working multigpu implemenation via python.
    # https://github.com/BVLC/caffe/blob/master/python/train.py
    from train import solve  # @UnresolvedImport

    # Customize the multigpu solver if you'd like. Change the caffe_mgpu_train
    # file locally and use that.
    # codelibdir = args_dict['codelibdir']
    # sys.path.insert(1, codelibdir)  #
    # import caffe_mgpu_train as train
    # train.train(solver_file, None, gpus, False)

    def train(
            solver,  # solver proto definition
            snapshot,  # solver snapshot to restore
            gpus,  # list of device ids
            timing=False,  # show timing info for compute and communications
    ):
        # NCCL uses a uid to identify a session
        uid = caffe.NCCL.new_uid()

        # caffe.init_log()
        caffe.log('Using devices %s' % str(gpus))

        procs = []
        for rank in range(len(gpus)):
            p = Process(
                target=solve,
                args=(solver, snapshot, gpus, timing, uid, rank))
            # p.daemon = True
            p.daemon = daemon_flag
            p.start()
            procs.append(p)

        for p in procs:
            p.join(timeout=timeout)

    train(solver_file, None, gpus, False)
    active_children()


if __name__ == "__main__":

    import argparse
    from textwrap import dedent

    class CustomFormatter(argparse.RawDescriptionHelpFormatter):
        '''Convenience formatter_class for argparse help print out.'''

    def parser_(desc):
        parser = argparse.ArgumentParser(description=dedent(desc),
                                         formatter_class=CustomFormatter)

        parser.add_argument(
            '--ngpus', type=int, default=4,
            help='Number of GPUs on the system. Default: %(default)s')

        args = parser.parse_args()

        return args

    args = parser_('')

    ngpus = args.ngpus

    examples_dir = os.path.join(_topdir, 'examples')
    codedir = '{}/caffe'.format(examples_dir)
    datadir = '{}/data/mnist'.format(examples_dir)
    workdir = '{}/workdir/caffe'.format(_topdir)
    mkdir_p(workdir)
    codeargs = {
        # 'codelibdir': codelibdir,
        'codedir': codedir,
        'datadir': datadir,
        'workdir': workdir,
        'ngpus': ngpus
    }

    main(codeargs)
