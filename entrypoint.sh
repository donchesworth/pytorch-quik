#!/bin/sh

cd /opt/dq
# pytest
pytest --cov=/opt/pq/pytorch_quik --cov-config=.coveragerc --cov-report=xml:coverage_cpu.xml
# curl https://codecov.io/bash
