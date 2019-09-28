FROM tensorflow/tensorflow:1.14.0-py3-jupyter
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
        libprotobuf-dev \
        python3.6-dev \
        python3-pip \
        python3-setuptools \
        python3-virtualenv \
        python3-wheel \
        wget \
        unzip \
        git && \
    add-apt-repository -r ppa:ubuntu-toolchain-r/test && \
    add-apt-repository -r ppa:maarten-fonville/protobuf && \
    apt-get autoclean && \
    apt-get autoremove --purge && \
    pip install --upgrade pip && \
    pip install 'apache-airflow[gcp_api]'==1.10.5 && \
    pip install docker && \
    git clone https://github.com/tensorflow/tfx.git && \
    cd ./tfx && \
    python setup.py bdist_wheel && \
    pip install -e .  && \
    jupyter nbextension enable --py widgetsnbextension && \
    jupyter nbextension install --py --symlink --sys-prefix tensorflow_model_analysis && \
    jupyter nbextension enable --py --sys-prefix tensorflow_model_analysis && \
    mkdir notebooks

VOLUME /notebooks
EXPOSE 8888
CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root"]
