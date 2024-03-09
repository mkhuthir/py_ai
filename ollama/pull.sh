#!/bin/sh

for model in llama llava mistral falcon orca-mini; do ollama pull $model; done
