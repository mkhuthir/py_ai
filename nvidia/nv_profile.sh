#!/usr/bin/sh

echo "saxpy.."
nvprof ./saxpy
echo "vector_add.."
nvprof ./vector_add
echo "vector_add_thread.."
nvprof ./vector_add_thread
echo "done!"

