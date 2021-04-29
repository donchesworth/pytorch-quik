FROM quay.io/donchesworth/rapids-dask-pytorch:py38-cuda10.2-rapids0.18-pytorch1.7-ubi8

# Labels
LABEL maintainer="Don Chesworth<donald.chesworth@gmail.com>"
LABEL org.label-schema.schema-version="0.1"
LABEL org.label-schema.name="pytorch-quik-test"
LABEL org.label-schema.description="Utilities for training with pytorch quik-er"

RUN pip install matplotlib sklearn

# Project installs
WORKDIR /opt/pq
COPY ./ /opt/pq/
RUN pip install .

RUN chgrp -R 0 /opt/pq/ && \
    chmod -R g+rwX /opt/pq/ && \
    chmod +x /opt/pq/entrypoint.sh

ENTRYPOINT ["/opt/pq/entrypoint.sh"]
