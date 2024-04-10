#! /usr/bin/python3

from llama_index.llms.ollama import Ollama

llm = Ollama(model="llama2", request_timeout=300.0)

resp = llm.complete("Who is Paul Graham?")
print(resp)