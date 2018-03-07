#!/bin/bash
# file: run_tf_mnist.sh

_basedir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


function join { local IFS="$1"; shift; echo "$*"; }


CONTAINER=nvcr.io/nvidia/tensorflow:17.12
TFCODE=${_basedir}/examples/tensorflow/mnist/mnist_deep.py
DATA=${_basedir}/examples/data/mnist
TFWORKDIR=${_basedir}/workdir/tensorflow/mnist

envlist=''

usage() {
cat <<EOF
Usage: $0 [-h|--help]
    [--container=docker-container] [--envlist=env1,env2,...] [--tfcode=file_path]
    [--data=data-directory] [--workdir=dir-path] [--<remain_args>]

    Run Tensorflow Mnist example. The options for this launcher script must use
    equal sign "=" for argruments, do not separate by space.

    --container - Caffe container to use.
        Default: ${CONTAINER}

    --envlist - Environment variable(s) to add into the container. Comma separated.
        Useful for CUDA_VISIBLE_DEVICES for example.

    --tfcode - Tensorflow code.
        Default: ${TFCODE}

    --data - Data directory where the mnist data already resides. If the data
        is not found in that location an attempt is made to download the data.
        Default path: ${DATA}

    --workdir - The work directory from which Tensorflow will be launched.
        Default: ${TFWORKDIR}

    -h|--help - Displays this help.

EOF
}


remain_args=()

while getopts ":h-" arg; do
    case "${arg}" in
    h ) usage
        exit 2
        ;;
    - ) [ $OPTIND -ge 1 ] && optind=$(expr $OPTIND - 1 ) || optind=$OPTIND
        eval _OPTION="\$$optind"
        OPTARG=$(echo $_OPTION | cut -d'=' -f2)
        OPTION=$(echo $_OPTION | cut -d'=' -f1)
        case $OPTION in
        --container ) larguments=yes; CONTAINER="$OPTARG"  ;;
        --envlist ) larguments=yes; envlist="$OPTARG"  ;;
        --tfcode ) larguments=yes; TFCODE="$OPTARG"  ;;
        --data ) larguments=yes; DATA="$OPTARG"  ;;
        --workdir ) larguments=yes; TFWORKDIR="$OPTARG"  ;;
        --help ) usage; exit 2 ;;
        --* ) remain_args+=($_OPTION) ;;
        esac
        OPTIND=1
        shift
        ;;
    esac
done

# grab all other remaning args.
remain_args+=($@)

# arguments to passthrough. Not used here.
script_args="$(join : ${remain_args[@]})"

# envlist="CUDA_VISIBLE_DEVICES"
envvars=''
if [ ! -z "${envlist// }" ]; then
    for evar in ${envlist//,/ } ; do
        envvars="-e ${evar}=${!evar} ${envvars}"
    done
fi

# echo $envvars


mkdir -p $DATA
mkdir -p $TFWORKDIR

dname=${USER}_tfmnist

# Orchestrate Docker container with user's privileges
nvidia-docker run --rm -t --name=$dname \
  -u $(id -u):$(id -g) -e HOME=$HOME -e USER=$USER -v $HOME:$HOME \
  $envvars \
  -e DATA=$DATA -v $DATA:$DATA -e TFCODE=$TFCODE -v $TFCODE:$TFCODE \
  --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
  -w ${TFWORKDIR} --entrypoint=bash $CONTAINER -c '
  python $TFCODE --data_dir $DATA
  '

