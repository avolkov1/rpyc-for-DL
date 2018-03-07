'''
'''
import os
import sys

import dill

from rpycdl_lib.common import mkdir_p

from cli_clients_common import get_server_connection


_topdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def dltfclient_code(codeargs):

    import subprocess

    container = codeargs['container']
    codefile = codeargs['codefile']
    cliargs = codeargs['cliargs']
    datadir = codeargs['datadir']
    workdir = codeargs['workdir']

    envvars = {
        'DATA': datadir,
        'WORKDIR': workdir
    }

    envvars_cmd = ' '.join(('-e {vn}="{vv}"'.format(vn=vn, vv=vv)
                            for vn, vv in envvars.items()))

    recommended_opts = '--shm-size=1g --ulimit memlock=-1 '\
        '--ulimit stack=67108864'

    workdir_cmd = '-w "{}"'.format(workdir)

    code_cmd = 'python {codefile} {cliargs}'.format(
        codefile=codefile, cliargs=cliargs)

    runcmd = 'nvidia-docker run --rm -t '\
        '-u $(id -u):$(id -g) -e HOME=$HOME -e USER=$USER -v $HOME:$HOME '\
        '{envvars_cmd} {recommended_opts} '\
        '{workdir_cmd} --entrypoint=bash {container} -c \' '\
        '{code_cmd}  \' '\
        .format(
            envvars_cmd=envvars_cmd,
            recommended_opts=recommended_opts,
            workdir_cmd=workdir_cmd, container=container,
            code_cmd=code_cmd)

    print('RUNNING NVIDIA-DOCKER CMD:\n\n{}\n\n'.format(runcmd))

    rp = subprocess.Popen(
        runcmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # rp.wait()

    for stdout_line in iter(rp.stdout.readline, ""):
        print stdout_line.strip()


def main():

    # Run mnist_deep training -------------------------------------------------
    tfserv = get_server_connection('deeplearning')

    ngpus = 1

    examples_dir = os.path.join(_topdir, 'examples')
    container = 'nvcr.io/nvidia/tensorflow:17.12'
    codefile = '{}/tensorflow/mnist/mnist_deep.py'.format(examples_dir)
    cliargs = '--data_dir "$DATA" --workdir "$WORKDIR"'
    datadir = '{}/data/mnist'.format(examples_dir)
    workdir = '{}/workdir/tensorflow/mnist/'.format(_topdir)

    mkdir_p(workdir)

    codeargs = {
        'container': container,
        'codefile': codefile,
        'cliargs': cliargs,
        'datadir': datadir,
        'workdir': workdir
    }

    tfcode_mnist = dill.dumps(dltfclient_code)

    tfserv.root.run_code(tfcode_mnist, codeargs, ngpus=ngpus,
                         stdout=sys.stdout, stderr=sys.stderr)
    tfserv.close()


if __name__ == "__main__":
    main()
