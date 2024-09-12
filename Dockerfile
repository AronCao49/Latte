FROM nvcr.io/nvidia/pytorch:22.09-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
RUN apt-get update
RUN apt-get install \
    # Open3D
    xorg-dev \
    libxcb-shm0 \
    libglu1-mesa-dev \
    python3-dev \
    # Filament build-from-source
    clang \
    libc++-dev \
    libc++abi-dev \
    libsdl2-dev \
    ninja-build \
    libxi-dev \
    # ML
    libtbb-dev \
    # Headless rendering
    libosmesa6-dev \
    # RealSense
    libudev-dev \
    autoconf \
    libtool \
    libssl-dev \
    --assume-yes

# Update cmake
# RUN conda upgrade -y cmake

# Basic install
RUN apt-get update
RUN apt-get -y install nano htop python3-pip tmux

# HrDC requirement
RUN apt-get install -y libsparsehash-dev libeigen3-dev
RUN apt-get update
RUN pip install numpy open3d
# RUN pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
# RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cu118.html
# RUN pip install spconv-cu118
# RUN pip install --upgrade git+https://github.com/facebookresearch/SparseConvNet.git
