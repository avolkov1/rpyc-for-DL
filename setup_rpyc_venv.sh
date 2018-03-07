#!/bin/bash

_basedir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p $HOME/pyenvs

rpycvenv=$HOME/pyenvs/rpycvenv

rm -r $rpycvenv

virtualenv $rpycvenv

source ${rpycvenv}/bin/activate

pip install -U pip
pip install numpy scipy  # these are needed for bvlc/caffe container
# pip install rpyc dill fasteners  # these are needed for running rpyc
# pip install nvidia-ml-py  # needed for querying GPUs

pushd $_basedir
pip install -e .
popd
