#!/usr/bin/env bash

if [ -d data ]; then
    echo "data directory already present"
else
    mkdir data
fi

wget --recursive --level=1 --cut-dirs=3 --no-host-directories \
  --directory-prefix=data --accept '*.gz' http://yann.lecun.com/exdb/mnist/

pushd data
gunzip *
popd
