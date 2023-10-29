FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

RUN apt -f install
RUN apt update && apt -y dist-upgrade 
RUN apt install -y python3-pip git ffmpeg libsm6 libxext6 

RUN pip3 install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118

RUN apt install nvidia-cuda-toolkit nvidia-cuda-toolkit-gcc

COPY requirements.txt /opt/app/requirements.txt
RUN pip3 install -r /opt/app/requirements.txt
RUN rm /opt/app/requirements.txt

RUN pip3 install -U mypy

RUN git config --global core.autocrlf true

RUN apt install imagemagick
