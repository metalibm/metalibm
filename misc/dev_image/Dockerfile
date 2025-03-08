#
# RISC-V Development Dockerfile
#
# Inspired by https://github.com/sbates130272/docker-riscv
#
# This Dockerfile creates a container full of lots of useful tools for
# RISC-V development.

# Pull base image (use Wily for now).
FROM ubuntu:22.04 AS ml_ci_base_deps

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

# install sollya
WORKDIR  /home/

RUN git clone https://gitlab.inria.fr/sollya/sollya.git -b master sollya_git
WORKDIR /home/sollya_git/
RUN sh autogen.sh && ./configure && make && make install

#
WORKDIR /home/

RUN  apt install -y python3-venv

#RUN apt install python3.8-venv
RUN git clone https://gitlab.com/metalibm-dev/pythonsollya -b dirty-21May2024
WORKDIR /home/pythonsollya
RUN python3 -m venv pythonsollya_venv
# RUN . pythonsollya_venv/bin/activate
ENV PATH=/home/pythonsollya/pythonsollya_venv/bin/:$PATH
#RUN pythonsollya_venv/bin/pip install  .
RUN pip install  .
ARG LD_LIBRARY_PATH=/usr/local/lib/
RUN export LD_LIBRARY_PATH=/usr/local/lib/
WORKDIR /home/app/


# FROM ubuntu:18.04 AS ml_ci_base_deps
# RUN apt-get update
# RUN apt-get install -y git
# RUN apt-get install -y python3 python3-setuptools libpython3-dev python3-pip
# RUN apt-get install -y libmpfr-dev libmpfi-dev libfplll-dev libxml2-dev wget

FROM ml_ci_base_deps AS ml_ci_base

RUN apt-get install -y build-essential
# install sollya's dependency

# install sollya
WORKDIR  /home/


# checking pythonsolya install
# RUN /home/pythonsollya/pythonsollya_venv/bin/pip install bigfloat
RUN apt-get install python3-six
RUN LD_LIBRARY_PATH="/usr/local/lib/" /home/pythonsollya/pythonsollya_venv/bin/python3 -c "import sollya"

# installing gappa
WORKDIR /home/
RUN apt-get install -y libboost-dev
RUN wget https://gappa.gitlabpages.inria.fr/releases/gappa-1.6.0.tar.gz
# RUN wget https://gforge.inria.fr/frs/download.php/file/37624/gappa-1.3.3.tar.gz
# RUN wget https://gappa.gitlabpages.inria.fr/releases/gappa-1.3.5.tar.gz
RUN tar -xzf gappa-1.6.0.tar.gz
WORKDIR /home/gappa-1.6.0/
RUN ./configure
RUN ./remake
RUN ./remake install

#
WORKDIR /home/
#RUN wget https://github.com/ghdl/ghdl/releases/download/v0.37/ghdl-0.37-ubuntu16-mcode.tgz
#RUN tar -xzf ghdl-0.37-ubuntu16-mcode.tgz

# RUN apt install libgnat-8
# debian packages does not come with ieee libraries (not GPL ?)
# RUN wget http://ftp.fr.debian.org/debian/pool/main/g/ghdl/ghdl_0.35+git20181129+dfsg-3_amd64.deb
# RUN wget http://ftp.fr.debian.org/debian/pool/main/g/ghdl/ghdl-mcode_0.35+git20181129+dfsg-3_amd64.deb 
# RUN dpkg -i ghdl_0.35+git20181129+dfsg-3_amd64.deb ghdl-mcode_0.35+git20181129+dfsg-3_amd64.deb

# WORKDIR /home/ghdl/
# RUN wget https://github.com/ghdl/ghdl/releases/download/v0.37/ghdl-0.37-buster-mcode-synth.tgz
# RUN tar -xzf ghdl-0.37-buster-mcode-synth.tgz
# testing ghdl
# RUN PATH=/home/ghdl/bin/:$PATH ghdl --version

# installing papi (required for performance counter access)
RUN apt install -y libpapi-dev

FROM ml_ci_base AS ml_ci_test


# installing ASMDE (metalibm dependency) directly
RUN pip install git+https://github.com/nibrunie/asmde@63e8789a43023967f3d20cead525434920dd8bcd
RUN pip install pyyaml

WORKDIR /home/
RUN git clone https://github.com/metalibm/metalibm.git -b asmde-status-cleanup
WORKDIR /home/metalibm/
ENV ML_SRC_DIR=/home/metalibm/
ENV PYTHONPATH=/home/metalibm/

# RUN /home/pythonsollya/pythonsollya_venv/bin/python3 ./valid/hw_non_regression.py
RUN python3 ./valid/soft_unit_test.py
# RUN /home/pythonsollya/pythonsollya_venv/bin/python3 ./valid/rtl_unit_test.py


