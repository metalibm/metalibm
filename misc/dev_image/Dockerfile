#
# Metalibm Development Dockerfile
#
# https://github.com/metalibm/metalibm


# Pull base image
FROM ubuntu:22.04 AS ml_dev_base_deps

RUN apt update

# setup dpkg (behind apt) to be non-interactive
# prevent some dependencies (e.g. tzdata) from spuriously
# prompting the user during apt-get install
ARG DEBIAN_FRONTEND=noninteractive

RUN apt install -y git
RUN apt install -y python3.10 python3-setuptools libpython3-dev python3-pip
# install sollya's dependency
RUN apt install -y libmpfr-dev libmpfi-dev libfplll-dev libxml2-dev wget
RUN apt install -y dh-autoreconf
RUN apt install -y build-essential
RUN apt install -y bison flex
RUN apt install -y python3-venv
RUN apt install -y python3-six
RUN apt install -y libboost-dev
# installing papi (required for performance counter access)
RUN apt install -y libpapi-dev

FROM ml_dev_base_deps AS ml_dev_base

# install sollya
WORKDIR  /home/
RUN git clone https://gitlab.inria.fr/sollya/sollya.git -b master sollya_git
WORKDIR /home/sollya_git/
RUN sh autogen.sh && ./configure && make && make install

# retrieving pythonsollya
WORKDIR /home/
RUN git clone https://gitlab.com/metalibm-dev/pythonsollya -b dirty-21May2024
WORKDIR /home/pythonsollya
RUN python3 -m venv pythonsollya_venv
# adding the path to the python virtual env into $PATH
ENV PATH=/home/pythonsollya/pythonsollya_venv/bin/:$PATH
# building and installing pythonsollya
RUN pip install  .
ARG LD_LIBRARY_PATH=/usr/local/lib/
RUN export LD_LIBRARY_PATH=/usr/local/lib/
WORKDIR /home/app/

# checking pythonsollya install
RUN LD_LIBRARY_PATH="/usr/local/lib/" python3 -c "import sollya"

# installing gappa
WORKDIR /home/
RUN wget https://gappa.gitlabpages.inria.fr/releases/gappa-1.6.0.tar.gz
RUN tar -xzf gappa-1.6.0.tar.gz
WORKDIR /home/gappa-1.6.0/
RUN ./configure
RUN ./remake
RUN ./remake install

# installing ASMDE (metalibm dependency) directly
RUN pip install git+https://github.com/nibrunie/asmde@63e8789a43023967f3d20cead525434920dd8bcd
RUN pip install pyyaml

WORKDIR /home/metalibm/
ENV ML_SRC_DIR=/home/metalibm/
ENV PYTHONPATH=/home/metalibm/
ENV LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH

FROM ml_dev_base AS ml_dev

# installing metalibm proper
WORKDIR /home/
RUN git clone https://github.com/metalibm/metalibm.git -b asmde-status-cleanup
WORKDIR /home/metalibm/
ENV ML_SRC_DIR=/home/metalibm/
ENV PYTHONPATH=/home/metalibm/

# minimal testing
RUN python3 ./valid/soft_unit_test.py


