apt-get update && apt-get upgrade -y
apt-get install -y python-dev python-pip python-nose g++ libblas-dev git cmake gfortran liblapack-dev zlib1g-dev libjpeg-dev libav-tools python-opengl libboost-all-dev libsdl2-dev swig clang unzip htop python-setuptools libibnetdisc-dev wget curl
pip install -U pip
pip install -r requirements.txt

# CatBoost
pip install catboost==0.1.1

# XGBoost
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost
git checkout 14fba01b5ac42506741e702d3fde68344a82f9f0
make -j4 && cd python-package; python setup.py install
cd ../../ && rm -rf xgboost

# OpenMpi
wget https://www.open-mpi.org/software/ompi/v2.1/downloads/openmpi-2.1.0.tar.gz
tar -xvzf openmpi-2.1.0.tar.gz && cd openmpi-2.1.0 && ./configure --prefix="/home/$USER/.openmpi" && make && make install
cd ../ && rm openmpi-2.1.0.tar.gz && rm -rf openmpi-2.1.0

# LightGBM
git clone --recursive https://github.com/Microsoft/LightGBM
cd LightGBM
git checkout 60c77487c15a69c9451be80fa9b15e987559a995
mkdir build && cd build && cmake -DUSE_MPI=ON .. && make && cd ../python-package/ && python setup.py install
cd ../../ && rm -rf LightGBM

