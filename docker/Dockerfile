ARG CUDA_DOCKER_VERSION=11.7.1-cudnn8-devel-ubuntu20.04

FROM nvidia/cuda:${CUDA_DOCKER_VERSION}

ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update 
RUN apt-get install vim -y
RUN apt-get install curl -y
RUN apt-get install wget -y
RUN apt-get install git -y
RUN apt-get install python3-pip -y
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /root


ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN apt-get install libglib2.0-0 -y --no-install-recommends
RUN apt-get install ffmpeg -y
