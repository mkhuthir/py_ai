#!/usr/bin/env bash

if [ -d data ]; then
    echo "data directory already present"
else
    echo "creating folder /data"
    mkdir data
fi

echo "Downloading data.."
pushd data

wget "https://s3.amazonaws.com/img-datasets/mnist.pkl.gz"
gunzip mnist.pkl.gz
popd
