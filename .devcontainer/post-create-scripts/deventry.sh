#!/usr/bin/env bash

mkdir $IPYTHONDIR
ipython profile create
unalias cp
cp -r $DEVDIR/ipython_config.json $IPYTHONDIR/profile_default

for dir in $BASEDIR/*; do
    pip install -e "$dir"
done