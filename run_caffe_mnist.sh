#!/bin/bash
# file: run_caffe_mnist.sh

_basedir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


function join { local IFS="$1"; shift; echo "$*"; }


CONTAINER=nvcr.io/nvidia/caffe:17.12
CAFFECODEDIR=${_basedir}/examples/caffe/mnist
DATA=${_basedir}/examples/data/mnist
CAFFEWORKDIR=${_basedir}/workdir/caffe


envlist=''

usage() {
cat <<EOF
Usage: $0 [-h|--help]
    [--container=docker-container] [--envlist=env1,env2,...] [--caffecodedir=dir_path]
    [--data=data-directory] [--caffeworkdir=dir-path] [--<remain_args>]

    Run Caffe Mnist example. The options for this launcher script must use
    equal sign "=" for argruments, do not separate by space. The remaining
    arguments can be specified with space as those are just passed through.

    --container - Caffe container to use.
        Default: ${CONTAINER}

    --envlist - Environment variable(s) to add into the container. Comma separated.
        Useful for CUDA_VISIBLE_DEVICES for example.

    --caffecodedir - Directory where the code resides. Specifically there
        should be two files in that directory:
            lenet_solver.prototxt - options for the solver
            lenet_train_test.prototxt.tmpl - The Caffe network template. The
                template file is assumed to have \$DATA environment variable
                that is substituted at runtime with the --data option or its
                default.
        Default: ${CAFFECODEDIR}

    --data - Data directory where the mnist data already resides. If the data
        is not found in that location an attempt is made to download the data.
        Default path: ${DATA}

    --caffeworkdir - The work directory from which Caffe will be launched. This
        is also the directory where the code files are copied to with
        environment variables substituted.
        Default: ${CAFFEWORKDIR}

    --<remain_args> - Additional args to pass through to Caffe. For example to
        run on some desired GPU specify: -gpu <commad separated gpu-indeces>
        For example on gpu 3: -gpu 3
        On GPUs 2 and 5: -gpu 2,5
        On all GPUs: -gpu all
        By default Caffe runs on GPU 0.

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
        --caffecodedir ) larguments=yes; CAFFECODEDIR="$OPTARG"  ;;
        --data ) larguments=yes; DATA="$OPTARG"  ;;
        --caffeworkdir ) larguments=yes; CAFFEWORKDIR="$OPTARG"  ;;
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

# arguments to passthrough to caffe such as "-gpu all" or "-gpu 0,1"
script_args="$(join : ${remain_args[@]})"


envvars=''
if [ ! -z "${envlist// }" ]; then
    for evar in ${envlist//,/ } ; do
        envvars="-e ${evar}=${!evar} ${envvars}"
    done
fi


mkdir -p $DATA
mkdir -p $CAFFEWORKDIR/mnist

# Backend storage for Caffe data.
BACKEND="lmdb"

dname=${USER}_caffe

# Orchestrate Docker container with user's privileges
nvidia-docker run -d -t --name=$dname \
  -u $(id -u):$(id -g) -e HOME=$HOME -e USER=$USER -v $HOME:$HOME \
  -e DATA=$DATA -v $DATA:$DATA \
  $envvars \
  -e BACKEND=$BACKEND -e script_args="$script_args" \
  --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
  -w $CAFFEWORKDIR $CONTAINER

sleep 1 # wait for container to come up

# download and convert data into lmdb format.
docker exec -it $dname bash -c '
  pushd $DATA

  for fname in train-images-idx3-ubyte train-labels-idx1-ubyte \
      t10k-images-idx3-ubyte t10k-labels-idx1-ubyte ; do
    if [ ! -e ${DATA}/$fname ]; then
        wget --no-check-certificate http://yann.lecun.com/exdb/mnist/${fname}.gz
        gunzip ${fname}.gz
    fi
  done

  popd

  TRAINDIR=$DATA/mnist_train_${BACKEND}
  if [ ! -d "$TRAINDIR" ]; then
    convert_mnist_data \
      $DATA/train-images-idx3-ubyte $DATA/train-labels-idx1-ubyte \
      $TRAINDIR --backend=${BACKEND}
  fi

  TESTDIR=$DATA/mnist_test_${BACKEND}
  if [ ! -d "$TESTDIR" ]; then
    convert_mnist_data \
      $DATA/t10k-images-idx3-ubyte $DATA/t10k-labels-idx1-ubyte \
      $TESTDIR --backend=${BACKEND}
  fi
  '

# =============================================================================
# SETUP CAFFE NETWORK TO TRAIN/TEST/SOLVER
# =============================================================================
cp ${CAFFECODEDIR}/lenet_solver.prototxt ${CAFFEWORKDIR}/mnist

DATA=$DATA envsubst < ${CAFFECODEDIR}/lenet_train_test.prototxt.tmpl > \
    ${CAFFEWORKDIR}/mnist/lenet_train_test.prototxt

# RUN TRAINING WITH CAFFE -----------------------------------------------------
docker exec -it $dname bash -c '
  # workdir is CAFFEWORKDIR when container was started.
  caffe train --solver=mnist/lenet_solver.prototxt ${script_args//:/ }
  '

docker stop $dname && docker rm $dname

