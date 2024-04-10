#! /usr/bin/python3

from llama_index.embeddings.ollama import OllamaEmbedding

ollama_embedding = OllamaEmbedding(
    model_name="llama2",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)

pass_embedding = ollama_embedding.get_text_embedding_batch(
    ["This is a passage!", "This is another passage"], show_progress=True
)
print(len(pass_embedding[0]))
print(len(pass_embedding[1]))

query_embedding = ollama_embedding.get_query_embedding("Where is blue?")
print(len(query_embedding))