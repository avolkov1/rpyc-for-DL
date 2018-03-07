

_basedir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


# After the script run can kill all the services via: pkill -f rpyc_server


DEBUG=true
# DEBUG=false

if [ "$DEBUG" = true ] ; then
    # for debugging send to background otherwise use daemon process.
    nvcmd="nvidia-docker run"
else
    nvcmd="nvidia-docker run -d"
    # When using daemon no need to send to background via &, but it was just
    # easier then having an extra if condition for ampersand.
fi

pushd ${_basedir}

datamnts="/tmp"
mntdata=''
if [ ! -z "${datamnts// }" ]; then
    for mnt in ${datamnts//,/ } ; do
        mntdata="-v ${mnt}:${mnt} ${mntdata}"
    done
fi


USEROPTS="-u $(id -u):$(id -g) -e HOME=$HOME -e USER=$USER -v $HOME:$HOME"
getent group > ${_basedir}/group
getent passwd > ${_basedir}/passwd


tfcontainer="nvcr.io/nvidia/tensorflow:17.12"


# nvidia-docker run -d --rm --name=tfserv --net=host \
${nvcmd} --rm --name=tfserv --net=host \
  $USEROPTS $mntdata \
  --hostname "$(hostname)_contain" \
  -v ${_basedir}/passwd:/etc/passwd:ro -v ${_basedir}/group:/etc/group:ro \
  --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
  -w ${_basedir} --entrypoint=bash $tfcontainer -c '
  source bash_init_rpyc.sh
  python tfrpyc_server.py --debug
  ' &
# Ampersand not needed if running as daemon


nvcfcontainer="nvcr.io/nvidia/caffe:17.12"

# for debugging send to background otherwise use daemon process
# nvidia-docker run -d --rm --name=nvcfserv --net=host \
${nvcmd} --rm --name=nvcfserv --net=host \
  $USEROPTS $mntdata \
  --hostname "$(hostname)_contain" \
  -v ${_basedir}/passwd:/etc/passwd:ro -v ${_basedir}/group:/etc/group:ro \
  --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
  -w ${_basedir} --entrypoint=bash $nvcfcontainer -c '
  source bash_init_rpyc.sh
  python nvcfrpyc_server.py --debug
  ' &


cfcontainer="bvlc/caffe:gpu"
# On Volta hardware rebuild Caffe with CUDA9 and NCCL2.
# cfcontainer="bvlc_caffe_nccl2"

# for debugging send to background otherwise use daemon process
# nvidia-docker run -d --rm --name=cfserv --net=host \
${nvcmd} --rm --name=cfserv --net=host \
  $USEROPTS $mntdata \
  --hostname "$(hostname)_contain" \
  -v ${_basedir}/passwd:/etc/passwd:ro -v ${_basedir}/group:/etc/group:ro \
  --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
  -w ${_basedir} --entrypoint=bash $cfcontainer -c '
  source bash_init_rpyc.sh
  python cfrpyc_server.py --debug
  ' &


# Start generic service that can be used to spawn docker services.
source bash_init_rpyc.sh
python dlrpyc_server.py --debug &  # send to background

popd
