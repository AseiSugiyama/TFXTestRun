FROM tensorflow/tensorflow:1.13.1-py3-jupyter
RUN apt-get update && \
    apt-get install --no-install-recommends -y -q software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    add-apt-repository ppa:maarten-fonville/protobuf && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install --only-upgrade libstdc++6 -y -q && \
    apt-get install --no-install-recommends -y -q \
    build-essential \
    ca-certificates \
    libsnappy-dev \
    protobuf-compiler \
    python3.5-dev \
    python3-pip \
    python3-setuptools \
    python3-virtualenv \
    python3-virtualenv \
    python3-wheel \
    wget \
    unzip \
    git && \
    pip install 'apache-airflow[gcp]' docker && \
    git clone https://github.com/tensorflow/tfx.git && \
    pip install -e ./tfx  && \
    mkdir notebooks && \
    jupyter nbextension install --py --symlink --sys-prefix tensorflow_model_analysis && \
    jupyter nbextension enable --py --sys-prefix tensorflow_model_analysis
VOLUME /notebooks
EXPOSE 8888
CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root"]
