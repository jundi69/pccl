###### CUDA:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda-repo-ubuntu2204-12-9-local_12.9.0-575.51.03-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-9-local_12.9.0-575.51.03-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-9-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-9

export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

source ~/.bashrc

# nvcc --version

###### CMAKE:
sudo apt update
sudo apt install -y build-essential git cmake
# cmake --version


###### PYTHON3.12:
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev python3-pip


###### PCCL:
git clone https://github.com/PrimeIntellect-ai/pccl.git
cd pccl
git submodule update --init --recursive

mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DPCCL_BUILD_CUDA_SUPPORT=ON ..
cmake --build . --config Release --parallel $(nproc)

cd ../python
python3.12 -m venv venv_pccl
source venv_pccl/bin/activate

pip install ./framework/

python -c 'import pccl; print(pccl.__version__)'
