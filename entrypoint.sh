#!/bin/sh

cd /opt/pq
# pytest
pytest --mpl --cov=/opt/pq/pytorch_quik --cov-config=.coveragerc --cov-report=xml:coverage_cpu.xml --ignore=test_mlflow.py
echo $(xmllint --xpath "string(//coverage/@line-rate)" coverage_cpu.xml)
# curl https://codecov.io/bash
