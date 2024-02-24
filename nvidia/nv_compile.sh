#!/usr/bin/sh

echo "saxpy.."
nvcc saxpy.cu -o saxpy
echo "vector_add.."
nvcc vector_add.cu -o vector_add
echo "vector_add_thread.."
nvcc vector_add_thread.cu -o vector_add_thread
echo "done!"

