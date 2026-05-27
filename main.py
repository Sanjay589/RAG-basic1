from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import ollama
import numpy as np
import faiss

# LOAD PDF
reader = PdfReader("data/sample.pdf")

full_text = ""

for page in reader.pages:
    text = page.extract_text()

    if text:
        full_text += text

# CHUNK TEXT
chunk_size = 100

chunks = []

for i in range(0, len(full_text), chunk_size):
    chunk = full_text[i:i + chunk_size]
    chunks.append(chunk)

# LOAD EMBEDDING MODEL
model = SentenceTransformer('all-MiniLM-L6-v2')

# CREATE EMBEDDINGS
chunk_embeddings = model.encode(chunks)

chunk_embeddings = np.array(chunk_embeddings).astype('float32')

# CREATE FAISS INDEX
dimension = chunk_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(chunk_embeddings)

# USER QUERY
query = input("Ask a question: ")

# QUERY EMBEDDING
query_embedding = model.encode([query])

query_embedding = np.array(query_embedding).astype('float32')

# SEARCH SIMILAR CHUNKS
k = 1

distances, indices = index.search(query_embedding, k)

retrieved_chunk = chunks[indices[0][0]]

print("\nRetrieved Context:\n")
print(retrieved_chunk)

# FINAL PROMPT
final_prompt = f"""
Answer the question using the context below.

Context:
{retrieved_chunk}

Question:
{query}
"""

# GENERATE RESPONSE USING OLLAMA
response = ollama.chat(
    model='llama3',
    messages=[
        {
            'role': 'user',
            'content': final_prompt
        }
    ]
)

answer = response['message']['content']

print("\nAI Response:\n")
print(answer)