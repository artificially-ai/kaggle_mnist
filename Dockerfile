FROM ekholabs/nvidia-cuda

MAINTAINER Wilder Rodrigues <wilder.rodrigues@ekholabs.ai>

RUN conda install -c conda-forge tensorflow-gpu -y && \
    conda install -c conda-forge numpy keras -y && \
    pip install aws-shell

RUN apt-get update && \
    apt-get install groff -y

RUN mkdir -p $HOME/.aws

WORKDIR /ekholabs/mnist
ADD . /ekholabs/mnist

ENV PYTHONPATH=$PYTHONPATH:.

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]