from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from prometheus_client import Counter, Histogram, generate_latest
from slowapi import Limiter
from slowapi.util import get_remote_address
import os
import time
import requests
print("ok1")
from   main import embed, result, clear_db
from chat_history import init_db
print("ok2")
app = FastAPI(title="Papyrus RAG Backend")

@app.on_event("startup")
def startup_event():
    init_db()
    os.makedirs("data/uploaded_files", exist_ok=True)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# ----------- Metrics -----------
GEN_TIME = Histogram("gen_time_seconds", "Time taken for text generation")
FIRST_TOKEN_LATENCY = Histogram("first_token_latency_seconds", "Time taken to receive the first token")
EMBED_TIME = Histogram("embed_time_seconds", "Time taken for embeddings")
LLM_REQ_COUNT = Counter("requests_total", "Total requests received")
EMBED_REQ_COUNT = Counter("embed_requests_total", "Total embed requests")

# ----------- Endpoints -----------
@app.post("/embed")
async def embed_endpoint(req: Request):
    EMBED_REQ_COUNT.inc()
    body = await req.json()
    pdfs = body.get("text", [])
    start = time.time()
    embed(pdfs)
    EMBED_TIME.observe(time.time() - start)
    return [{"status": "success"}]

@app.post("/generate")
@limiter.limit("10/minute")
async def generate(req: Request):
    LLM_REQ_COUNT.inc()
    body = await req.json()
    query = body.get("query", "")
    new_chat = body.get("new_chat", False)

    def metric_wrapper():
        attempt = 0
        while attempt < MAX_RETRIES:
            start = time.time()
            full_started = False
            try:
                for token in result(query, new_chat):
                    if not full_started:
                        full_started = True
                        FIRST_TOKEN_LATENCY.observe(time.time() - start)
                    yield token
                GEN_TIME.observe(time.time() - start)
                return

            except Exception as e:
                if full_started:
                    # Streaming already began, do not retry
                    yield "\n[Generation interrupted]"
                    return
                attempt += 1
                time.sleep(RETRY_DELAY)
        yield "\n[LLM unavailable after retries]"
        return
    
    return StreamingResponse(
        metric_wrapper(), 
        media_type="text/plain"
        )

@app.post("/clear")
def clear():
    clear_db()
    return {"status": "cleared"}

@app.get("/metrics")
def metrics():
    return generate_latest()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def readiness():
    try:
        r = requests.get(LLM_URL, timeout=2)
        return {"status": "ready"}
    except requests.RequestException:
        return {"status": "not ready"}, 503
    
print("ok3")