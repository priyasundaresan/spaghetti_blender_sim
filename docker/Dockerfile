#FROM nvidia/cudagl:11.1-base-ubuntu18.04
#FROM nvidia/cudagl:11.3-base-ubuntu20.04
#FROM nvidia/cudagl:11.3.0-devel-ubuntu20.04
FROM nvidia/cudagl:11.2.0-base-ubuntu18.04

# Enviorment variables
ENV DEBIAN_FRONTEND noninteractive
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
ENV PATH "$PATH:/bin/3.1/python/bin/"
ENV BLENDER_PATH "/bin/3.1"
ENV BLENDERPIP "/bin/3.1/python/bin/pip3"
ENV BLENDERPY "/bin/3.1/python/bin/python3.10"

RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN rm /etc/apt/sources.list.d/*
RUN apt-get clean

# Install dependencies
RUN apt-get update && apt-get install -y \ 
    wget \ 
    libopenexr-dev \ 
    bzip2 \ 
    build-essential \ 
    zlib1g-dev \ 
    libxmu-dev \ 
    libxi-dev \ 
    libxxf86vm-dev \ 
    libfontconfig1 \ 
    libxrender1 \ 
    libgl1-mesa-glx \ 
    xz-utils \
    python-opengl 

RUN apt install xvfb -y

# Download and install Blender

RUN wget https://mirror.clarkson.edu/blender/release/Blender3.1/blender-3.1.2-linux-x64.tar.xz \
    && tar -xvf blender-3.1.2-linux-x64.tar.xz --strip-components=1 -C /bin \ 
    && rm -rf blender-3.1.2-linux-x64.tar.xz \ 
    && rm -rf blender-3.1.2-linux-x64

RUN wget https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tgz \ 
    && tar -xzf Python-3.10.0.tgz \ 
    && cp -r Python-3.10.0/Include/* $BLENDER_PATH/python/include/python3.10/ \ 
    && rm -rf Python-3.10.0.tgz \ 
    && rm -rf Python-3.10.0 

## Blender comes with a super outdated version of numpy (which is needed for matplotlib / opencv) so override it with a modern one
#RUN rm -rf ${BLENDER_PATH}/python/lib/python3.7/site-packages/numpy 
#


## Must first ensurepip to install Blender pip3 and then new numpy
RUN ${BLENDERPY} -m ensurepip && ${BLENDERPIP} install --upgrade pip && \
    ${BLENDERPIP} install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cu113/torch_nightly.html && \
    ${BLENDERPIP} install tqdm==4.42.1 && \
    ${BLENDERPIP} install gym==0.16.0 && \
    ${BLENDERPIP} install tensorboard==1.15.0 && \
    ${BLENDERPIP} install opencv-python && \
    ${BLENDERPIP} install scikit-learn 
#
## Fake a display 
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y --no-install-recommends python3.10 python3.10-dev python3-pip python3-tk && \
    pip3 install --no-cache-dir --upgrade pip setuptools && \
    echo "alias python='python3'" >> /root/.bash_aliases && \
    echo "alias pip='pip3'" >> /root/.bash_aliases

RUN pip install --upgrade pip && \
    pip install \
        pyvirtualdisplay \
        piglet

RUN apt-get install -y vim

WORKDIR /host
