# Using my base rapids-dask-pytorch container
FROM quay.io/donchesworth/rapids-dask-pytorch:py38-cuda10.2-rapids0.18-pytorch1.7-ubi8

# Labels
LABEL maintainer="Don Chesworth<donald.chesworth@gmail.com>"
LABEL org.label-schema.schema-version="0.1"
LABEL org.label-schema.name="pqsp_dev_container"
LABEL org.label-schema.description="a development container for pytorch-quik and solution-prediction-train"

ENV PATH="/opt/conda/envs/rdp/bin:/opt/conda/bin:$PATH"
ENV IPYTHONDIR="/opt/rdp/.ipythondir/"

SHELL ["/bin/bash", "-c"]
RUN source /opt/conda/bin/activate && conda activate rdp && \
    conda install --name rdp ipykernel -y

COPY requirements/ /tmp/
RUN pip install -r /tmp/r1.txt && pip install -r /tmp/r2.txt && \
    pip install flake8 black

# run crazy ceean_connectors install/uninstall
COPY pip.conf /opt/rdp/
ENV PIP_CONFIG_FILE="/opt/rdp/pip.conf"

RUN pip install ceean_connectors[vault] && pip install --upgrade boto3==1.16.52 botocore==1.19.52 && \
    pip uninstall psycopg2 -y && pip install --upgrade --force-reinstall psycopg2-binary==2.8.6

RUN ipython profile create
