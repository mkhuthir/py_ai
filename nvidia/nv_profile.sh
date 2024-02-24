#!/usr/bin/sh

echo "saxpy.."
nvprof ./saxpy
echo "vector_add .."
nvprof ./vector_add
echo "done!"

