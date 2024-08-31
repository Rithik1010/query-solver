FROM continuumio/miniconda3

ARG APP_NAME
ARG ENVIRONMENT
ARG APP_VERSION

RUN apt-get -y update; apt-get -y install curl
RUN conda create -n p3.11 python=3.11 -y

ADD requirements.txt /root/${APP_NAME}/requirements.txt
RUN /opt/conda/envs/p3.11/bin/pip install -r /root/${APP_NAME}/requirements.txt

ENV PATH /opt/conda/envs/p3.11/bin:$PATH
RUN /bin/bash -c "source activate p3.11"

RUN mkdir -p /root/${APP_NAME}/src

RUN mkdir -p /root/${APP_NAME}/resources/${ENVIRONMENT}

COPY ./src /root/${APP_NAME}/src

COPY ./docs /root/${APP_NAME}/docs

ENV APP_VERSION $APP_VERSION

WORKDIR /root/${APP_NAME}/

CMD ["/bin/bash", "-c", "python src/main.py"]
