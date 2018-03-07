
Running and Managing Deep Learning as a Service on GPU Servers with RPyC 
and Containers
-------------------------------------------------------------------------------

Two demo scripts are included to illustrate the basics of running code via
containers.

* [`run_caffe_mnist.sh`](run_caffe_mnist.sh) -
    See help `--help`. Default container: `nvcr.io/nvidia/caffe:17.12`

* [`run_tf_mnist.sh`](run_tf_mnist.sh) -
    See help `--help`. Default container: `nvcr.io/nvidia/tensorflow:17.12`

The above scripts run an mnist example on NVCaffe and Tensorflow respectively.
See the help for instructions to run the examples:
```
run_caffe_mnist.sh --help
run_tf_mnist.sh --help
```

## Using Environment Variables: `NV_GPU` and `CUDA_VISIBLE_DEVICES`

NVCaffe will always allocate some memory on GPU 0 for management purposes
unless it is masked off (in which case this management happens on the first
visible GPU). Tensorflow will allocate most if not all of available GPU memory
on all GPUs that are visible to it. One way to mask what GPUs are used via
containers is by using the `NV_GPU` flag.<br/>
[https://github.com/NVIDIA/nvidia-docker/wiki/nvidia-docker#gpu-isolation](https://github.com/NVIDIA/nvidia-docker/wiki/nvidia-docker#gpu-isolation)

##### Examples using NV_GPU

```
# Caffe will only run on the second GPU visible on the system
NV_GPU=1 ./run_caffe_mnist.sh

# [0] Tesla P40        | 22'C,   0 % |    10 / 22912 MB |
# [1] Tesla P40        | 33'C,  63 % |   429 / 22912 MB | mnistex(411M)
# [2] Tesla P40        | 22'C,   0 % |    10 / 22912 MB |
# [3] Tesla P40        | 22'C,   0 % |    10 / 22912 MB |


# TF will only run on the second GPU visible on the system
NV_GPU=1 ./run_tf_mnist.sh

# [0] Tesla P40        | 22'C,   0 % |    10 / 22912 MB |
# [1] Tesla P40        | 32'C,  76 % | 21981 / 22912 MB | mnistex(21963M)
# [2] Tesla P40        | 23'C,   0 % |    10 / 22912 MB |
# [3] Tesla P40        | 22'C,   0 % |    10 / 22912 MB |


```

The NV_GPU can be used for multi-GPU configuration as well. The TF mnist
example is not multi-gpu enabled, but the Caffe example is. Caffe enables
multigpu data-parallelism via command line interface (cli) with
`-gpu <comma_sep_ids>` option.
```
# Caffe will run on the second and third GPU visible on the system
NV_GPU=1,2 ./run_caffe_mnist.sh -gpu 0,1

# [0] Tesla P40        | 22'C,   0 % |    10 / 22912 MB |
# [1] Tesla P40        | 32'C,  53 % |   435 / 22912 MB | mnistex(417M)
# [2] Tesla P40        | 29'C,  53 % |   435 / 22912 MB | mnistex(417M)
# [3] Tesla P40        | 22'C,   0 % |    10 / 22912 MB |

```

##### Examples using `CUDA_VISIBLE_DEVICES`

Alternatively or in addition to `NV_GPU` within a container one can also use
`CUDA_VISIBLE_DEVICES` to isolate which GPUs are used. Although it is possible
to by-pass this variable, most frameworks will adhere to it.
```
# TF will only run on the third GPU visible on the system
CUDA_VISIBLE_DEVICES=2 ./run_tf_mnist.sh --envlist=CUDA_VISIBLE_DEVICES

# [0] Tesla P40        | 22'C,   0 % |    10 / 22912 MB |
# [1] Tesla P40        | 23'C,   0 % |    10 / 22912 MB |
# [2] Tesla P40        | 25'C,  77 % | 21981 / 22912 MB | mnistex(21963M)
# [3] Tesla P40        | 22'C,   0 % |    10 / 22912 MB |


# Caffe will run on the third and fourth GPU visible on the system
CUDA_VISIBLE_DEVICES=2,3 ./run_caffe_mnist.sh --envlist=CUDA_VISIBLE_DEVICES -gpu 0,1

# [0] Tesla P40        | 23'C,   0 % |    10 / 22912 MB |
# [1] Tesla P40        | 24'C,   0 % |    10 / 22912 MB |
# [2] Tesla P40        | 25'C,  55 % |   435 / 22912 MB | mnistex(417M)
# [3] Tesla P40        | 25'C,  55 % |   435 / 22912 MB | mnistex(417M)
```

Note that unlike `CUDA_VISIBLE_DEVICES` the `NV_GPU` is external to the
container. The `NV_GPU` variable is only useful when starting the container.
Whereas `CUDA_VISIBLE_DEVICES` needs to be set internally in the container
(typically done via `-e` docker option). Refer to the source code of the
`run_caffe_mnist.sh` and `run_tf_mnist.sh` scripts for how `--envlist` is used
to inject `CUDA_VISIBLE_DEVICES` into the environment of the container.


## Using RPyC To Launch Containerized Services

A close alternative to RPyC is Pyro. Pyro is a bit more complicated, but
has more features in regards to hardening and security of RPC. RPyC has an API
for authentication and authorization. The examples below do not use it.
This would be a potential future improvement.

### INSTALL

For convenience there is a `setup.py` to install the `rpycdl_lib` with its
dependencies via:
```
pip install -e .
```

The example services/servers can be deployed in a variety of ways. Start out by
setting up a python virtualenv with `rpyc` installed. Use the
[`setup_rpyc_venv.sh`](setup_rpyc_venv.sh) script or use it as a guide.
The [`rpyc_DL`](rpyc_DL) examples require packages:
`rpyc, dill, fasteners, nvidia-ml-py (or nvidia-ml-py3)`. (These demos have
been tested only on Python 2.7.) Alternatively instead of setting up a
virtualenv install/extend these packages directly into a desired framework
container.

Here is an expanded example assuming that the virtualenv `$HOME/pyenvs/rpycvenv`
has been created using [`setup_rpyc_venv.sh`](rpyc_DL/setup_rpyc_venv.sh).
```bash
# RUN THIS ON A GPU SERVER FROM "rpyc_DL" directory or wherever the
# tfrpyc_server.py is installed. 

_basedir=$PWD

USEROPTS="-u $(id -u):$(id -g) -e HOME=$HOME -e USER=$USER -v $HOME:$HOME"
getent group > ${_basedir}/group
getent passwd > ${_basedir}/passwd

tfcontainer="nvcr.io/nvidia/tensorflow:17.12"

nvidia-docker run -d --rm --name=tfserv --net=host \
  $USEROPTS -v /tmp:/tmp \
  --hostname "$(hostname)_contain" \
  -v ${_basedir}/passwd:/etc/passwd:ro -v ${_basedir}/group:/etc/group:ro \
  --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
  -w ${_basedir} --entrypoint=bash $tfcontainer -c '
  source $HOME/pyenvs/rpycvenv/bin/activate
  # http://docs.nvidia.com/deeplearning/dgx/best-practices/index.html#venvfns
  source $HOME/venvfns.sh
  # Global site packages are needed because that is typically how one interfaces
  # with the frameworks, but enabling explicitly might or might not be needed
  # depending on how python is setup in the container.
  enablevenvglobalsitepackages
  python tfrpyc_server.py --debug
  # disablevenvglobalsitepackages
  '

```

Important things to note about the example above. The `-v /tmp:/tmp` option
mounts the `/tmp` directory into the container. These servers were written to
use file-locking via `fasteners` python package to manage GPU resources. All
containers being launched with these servers should map `/tmp` for this to work
with multiple containerized services. Refer to the API section regarding details.

Somewhere else on the network the `rpyc registry` should be running. Launch it
via: `rpyc_registry.py`. You could re-use the `rpycvenv` environment. Once
the server is running you should get a message from the registry:
```
DEBUG:REGSRV/UDP/18811:server started on 0.0.0.0:<some-port>
DEBUG:REGSRV/UDP/18811:registering <some-ip-adress>:<some-port> as TENSORFLOW, TENSORFLOW-1.4.0
```
At this point a client can be launched that uses the "tensorflow" or
"tensorflow-1.4.0" service. Modify the examples to suite your needs.

Neither the registry nor the client code has to run on a GPU server, but they
need to have visibility on the network to the server. Refer to RPyC official
documenation for further details.

Refer to the startup script for reference
[`start_TFandCFandDL_rpyc_servers.sh`](rpyc_DL/start_TFandCFandDL_rpyc_servers.sh).

### Examples for RPyC Servers and Clients

Several examples of servers are given that run containerized services. The
Caffe and Tensorflow mnist examples are used for client code running on
these services. Refer to the script
[`start_TFandCFandDL_rpyc_servers.sh`](rpyc_DL/start_TFandCFandDL_rpyc_servers.sh)
for how services are launched on a GPU server. Refer to [INSTALL](#install)
section for details.

* [`tfrpyc_server.py`](rpyc_DL/tfrpyc_server.py)

    RPyC server for running Tensorflow code. This server needs to run from
    within a Tensorflow container.

* [`tfrpyc_client.py`](rpyc_DL/tfrpyc_client.py)

    RPyC client code. Connects to a service named "tensorflow". Then runs three
    examples, one that queries the GPU devices on the server, another that
    runs mnist code, and a third example to run multigpu Cifar10.

    Between the examples the client closes the first session/connection,
    and opens a new connection. When closing the first session Tensorflow
    releases whatever memory it might have allocated on the GPUs. The memory
    is released as long as the service runs on a "forking type RPyC server".
    ```
    from rpyc.utils.server import ForkingServer as Server
    ```
    The docker container within which the server is running does not have to be
    restarted to release the GPU memory.

* [`nvcfrpyc_server.py`](rpyc_DL/nvcfrpyc_server.py)

    RPyC server for running NVCaffe code. This server needs to run from
    within an nvcaffe container.

* [`nvcfrpyc_client.py`](rpyc_DL/nvcfrpyc_client.py)

    Corresponding NVCaffe client. Prior to running this example run the
    `run_caffe_mnist.sh` code to download mnist data and prepare the lmdb
    databases.

    This example does the same thing as `run_caffe_mnist.sh` script, but
    partially using python API as well. The training is done via cli using
    subprocess to invoke the caffe cli. NVCaffe does not expose multigpu
    managing via python, only via C++:<br/>
    [https://github.com/NVIDIA/caffe/blob/caffe-0.16/docs/multigpu.md](https://github.com/NVIDIA/caffe/blob/caffe-0.16/docs/multigpu.md)

* [`cfrpyc_server.py`](rpyc_DL/cfrpyc_server.py)

    RPyC server for running Caffe code. This server needs to run from
    within a `bvlc/caffe:gpu` container. On Volta GPUs the `bvlc/caffe:gpu`
    container does not work. Use the dockerfile
    [`Dockerfile.caffe_gpu`](dockerfiles/Dockerfile.caffe_gpu) to compile
    BVLC Caffe version compatible for Volta GPUs.

* [`cfrpyc_client.py`](rpyc_DL/cfrpyc_client.py)

    Corresponding Caffe client. Prior to running this example run the
    `run_caffe_mnist.sh` code to download mnist data and prepare the lmdb
    databases.

    A nice feature of BVLC/Caffe is that the multi-gpu training has
    been exposed through the python API. Officially this is not supported:<br/>
    [https://github.com/BVLC/caffe/blob/master/docs/multigpu.md](https://github.com/BVLC/caffe/blob/master/docs/multigpu.md)<br/>
    But when testing I was able to run multigpu from python interface. For
    further details refer to:<br/>
    [https://github.com/BVLC/caffe/blob/master/python/train.py](https://github.com/BVLC/caffe/blob/master/python/train.py)<br/>

    This example is mostly the same as `nvcfrpyc_client.py`, but it uses the
    python API to run multi-gpu Caffe training. 
    Running python's `multiprocessing.Process` module directly within rpyc does
    not seem to be reliable. Such an approach sometimes runs and other times
    fails randomly. A more reliable approach is to have the client code invoke
    the multigpu Python code via subprocess similarly to how it's done in
    NVCaffe case.

    The non-reliable code is demonstrated in the function `cfclient_mnist_code`
    and the reliable approach is shown in function `cfclient_mnist_code_spawn`.
    Only the reliable approach is used. If you would like to run/test unreliable
    approach uncomment that portion of the code in the `main` routine.

* [`dlrpyc_server.py`](rpyc_DL/dlrpyc_server.py)

    Generic RPyC server for running client code. This implementation is to
    illustrate the idea of building on Inversion of Control principle.

* [`dlrpyc_client.py`](rpyc_DL/dlrpyc_client.py)

    Corresponding example of a client code to run on the DL server. The client
    is responsible for setting up the container, environment, and command. The
    example shows how to run tensorflow mnist code using the
    `nvcr.io/nvidia/tensorflow:17.12` container.

The examples above illustrate how one might go about building up a framework
to run heterogeneous Deep Learning tasks on shared GPU resources. By using
`NV_GPU`, `CUDA_VISIBLE_DEVICES`, and frameworks' own internal resource
managing APIs, it is possible to manage the GPUs and memory on the GPUs.

Once the registry is running and the services are launched run the client code
via:
```
source $HOME/pyenvs/rpycvenv/bin/activate
python <example_above>_client.py
```


### API for RPyC Deep Learning Services

The core API is implemented via classes `DLServiceMixin` and`GPURmgr`. The
main function exposed to the client is `exposed_run_code`.
```python
def exposed_run_code(
        self,
        code_callback=None,  # Serialized code via dill.dumps(some_function)
        code_args=None,
        ngpus=1,  # Number of GPUs to reserve. Passed to GPURmgr
        stdout=None,  # for redirection to client's terminal
        stderr=None  # for redirection to client's terminal
        ):
```
The client is expected to serialize the code via `dill` package. The
`code_args` is passed through to the `code_callback`. The `code_args` should be
brineable by RPyC's definition.<br/>
[https://rpyc.readthedocs.io/en/latest/api/core_brine.html#api-brine](https://rpyc.readthedocs.io/en/latest/api/core_brine.html#api-brine)<br/>
If `code_args` has public attributes the client needs to connect with that option:
```
config = {'allow_public_attrs': True}
cserv = rpyc.connect_by_service(sevice_name, config=config)
```

The `GPURmgr` is a very basic resource manager. It is not topology aware and
does a first-come-first-serve acquisition. The acquisition works via file
locking using the `fasteners` package. If more GPUs are requested than the
system contains then an exception is raised.

First the system is "locked" via lock file: `/tmp/tmp_lock_sysgpus`. Then
iterate through the GPUs on the system and attempt to create a lock for each
GPU `/tmp/tmp_lock_gpu{}` where `{}` is the GPU index. Break out of the loop if
the number of locked GPUs equals requested number of GPUs. Otherwise sleep for
1 second and try again to lock additional GPUs that were not locked on the
previous try. This repeats until the required number of GPUs have been acquired
at which point the the system lock is released. The clients environment is
modified with `CUDA_VISIBLE_DEVICES` and `NV_GPU` environment variables
corresponding to the locked GPUs. Then the client code runs. Once the client
code completes the lock files for the GPUs are released.

Per the logic above only one service at any given time can attempt to acquire
GPUs on a system as the other attempts will block on the syslock. The service
trying to lock the GPUs will loop until enough GPUs become available waiting on
a previous service to complete and releas its GPUs. Say service request A is
using 2 GPUs, 0 and 1, on a system with 4 GPUs (0, 1, 2, 3). The next service B
needs 3 GPUs. Service B will lock 2 and 3 and loop/wait until either 0 or 1
will be released by service A. Let service C come after B and require 1 GPU, it
will block until service B acquires its GPUs. Once A completes B has locks on
GPUs 2, 3, and 0 launching its client code and releasing the syslock. Service C
will then acquire GPU 1 and also run at the same time as B. And so on.


### Ideas for improvements

These examples are very basic. There is no sophisticated scheduling nor resource
management. To take the idea further one could decouple the job management and
queueing through an intermediary scheduler which would then send off the work
to the appropriate workers/services.

The interfacing with client code/jobs is very primitive. No ability to kill
or cancel a launched job (although killing the client connection seems to kill
the job in most cases). No ability to set time limits on the jobs (unless these
are explicitly coded into the client code). There are many things left to be
desired in terms of functionality compared to a sophisticated scheduler
framework or platform such as slurm, dask, etc.

The examples here are to illustrate how one might go about setting up a
micro-service oriented software architecture with Deep Learning frameworks and
containers.

