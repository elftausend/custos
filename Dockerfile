# this Dockerfile requires nvidia-docker in order to test CUDA & OpenCL features with the --gpus=all option
#
# nvidia-docker: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
#
# usage:
#   $ docker build .
#   $ # ...
#   $ Successfully built <IMAGE ID>
#   $ docker run --gpus=all -it <IMAGE ID>
#   $ cargo test --features=cuda
#
# alternate usage:
#   $ docker run --gpus=all -it $(docker build -q .)
#   $ cargo test --features=opencl

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
LABEL Name=cuda-opencl-rust Version=0.0.1
ARG TZ=America/New_York
ENV TZ=${TZ}

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get -y update

RUN apt-get install -y libblas-dev liblapack-dev

RUN apt-get install -y clinfo nvidia-opencl-dev

RUN apt-get install -y curl pkg-config sudo

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

ADD . /root/crate/

WORKDIR /root/crate

RUN . $HOME/.cargo/env cargo update

RUN . $HOME/.cargo/env cargo build --features=all
