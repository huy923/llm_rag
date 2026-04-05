import ollama, os, glob,json
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.responses import StreamingResponse
from typing import Dict
app = FastAPI()

EMBEDDING_MODEL = 'nomic-embed-text'
LANGUAGE_MODEL = 'openhermes'

VECTOR_DB_PATH = 'vector_db/vector.json'

# Each element in the VECTOR_DB will be a tuple (chunk, embedding)
# The embedding is a list of floats, for example: [0.1, 0.04, -0.34, 0.21, ...]

VECTOR_DB = []
if os.path.exists(VECTOR_DB_PATH) is None:
    raise FileNotFoundError(f'No vector database found at {VECTOR_DB_PATH}. Please run textToVector.py first.')

with open(VECTOR_DB_PATH, 'r', encoding='utf-8') as file:
    VECTOR_DB = json.load(file)
    
print(f'Loaded {len(VECTOR_DB)} entries from saved vector database.')

def cosine_similarity(a, b):
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0.0

def retrieve(query, top_n, temperature, max_tokens): # take back

    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    similarities = []
    
    for item in VECTOR_DB:
        chunk, embedding = item
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    print("\n".join([f'{chunk} (similarity: {similarity})' for chunk, similarity in similarities[:top_n]]))
    return similarities[:top_n]

class ChatRequest(BaseModel):
    input_query: str = ""
    history: Dict[str, str] = {}
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: int = 10
    

@app.post('/chatbot')
async def chatbot(chat_request: ChatRequest):
    input_query = chat_request.input_query
    history = chat_request.history
    top_p = chat_request.top_p
    temperature = chat_request.temperature
    max_tokens = chat_request.max_tokens
    
    retrieved_knowledge = retrieve(query=input_query,top_n=top_p,temperature=temperature,max_tokens=max_tokens)
    # legal_context = '\n'.join([f'- {chunk}' for chunk, _ in retrieved_knowledge])
    
    instruction_prompt = """
You are a professional English teacher who explains grammar, vocabulary, and pronunciation in a clear and friendly way. 
You can speak both English and Vietnamese to support learners.
When a user asks a question, first explain it in English, then provide a Vietnamese explanation if needed.
Use examples to help understanding.
"""

    messages = [{'role': 'system', 'content': instruction_prompt}]

    for i, (user_msg, bot_msg) in enumerate(history.items()):
        messages.append({'role': 'user', 'content': user_msg})
        messages.append({'role': 'assistant', 'content': bot_msg})

    messages.append({'role': 'user', 'content': input_query})

    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=messages,
        stream=True,
    )
    def token_generator():
        for chunk in stream:
            token = chunk.get("message", {}).get("content", "")
            yield token

    return StreamingResponse(token_generator(), media_type="text/plain")
