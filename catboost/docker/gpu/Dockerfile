FROM nvidia/cuda:10.2-base-ubuntu18.04

ENV CATBOOST_VERSION=0.24.1

ENV JUPYTER_TOKEN=""

WORKDIR /catboost

EXPOSE 8888

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl wget \
        software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y -u && \
    apt install -y --no-install-recommends python3.8 python3-distutils && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    curl -k https://bootstrap.pypa.io/get-pip.py | python3.8 - && \
    apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false && \
    apt-get clean && apt-get autoclean && \
	rm -rf /usr/src/python /var/lib/apt/lists/* /tmp/* /var/lib/apt/lists/* /var/cache/apt/*.bin && \
    find /var/log -iname '*.gz' -delete && \
    find /var/log -iname '*.1' -delete && \
    rm -rf /var/tmp/*

RUN python3.8 -m pip install catboost==$CATBOOST_VERSION hyperopt scikit-learn matplotlib seaborn plotly jupyter ipywidgets && jupyter nbextension enable --py widgetsnbextension && pip cache purge

RUN echo $CATBOOST_VERSION

RUN wget https://github.com/catboost/catboost/releases/download/v$CATBOOST_VERSION/catboost-linux-$CATBOOST_VERSION --tries=10 --retry-connrefused --progress=dot:giga -O catboost && chmod +x catboost

COPY docker/entry_point.py /catboost/entry_point.py

RUN chmod +x /catboost/entry_point.py

COPY tutorials /catboost/tutorials

CMD /catboost/entry_point.py