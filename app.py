import os
import logging
import time
from typing import List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

from src.document_processor import DocumentProcessor
from src.rag_engine import RAGEngine
from src.llm_client import LLMClient

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Corporate Policy RAG API",
    description="Hybrid RAG system for HR Policies & Memos using TOON Format + Ollama",
    version="2.0.0"
)

rag_engine = None
llm_client = None
doc_processor = None

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    expanded_query: str
    retrieved_docs: List[dict]
    processing_time: float

class RebuildResponse(BaseModel):
    status: str
    message: str

@app.on_event("startup")
async def startup_event():
    global rag_engine, llm_client, doc_processor
    
    logger.info("Starting Corporate Policy RAG Server...")
    
    doc_processor = DocumentProcessor()
    rag_engine = RAGEngine()
    llm_client = LLMClient() 
    
    json_path = os.path.join(os.getenv("OUTPUT_PATH", "ingested_data/"), "ingested_documents.json")
    
    if os.path.exists(json_path) and not rag_engine.load_index():
        logger.info("Index not found on disk. Building from JSON...")
        docs = rag_engine.load_documents_from_json(json_path)
        rag_engine.build_index(docs)
    elif rag_engine.load_index():
        logger.info("Database loaded successfully.")
    else:
        logger.warning(f"No data found at {json_path}. Please call /rebuild-index endpoint.")

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "Corporate-Policy-RAG"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    start_time = time.time()
    query = request.question
    
    if not rag_engine:
        raise HTTPException(status_code=503, detail="System is initializing or index not ready.")

    try:
        expanded_query = llm_client.expand_query(query)
        
        retrieved_docs = rag_engine.search(expanded_query)
        
        final_answer = llm_client.generate_answer(query, retrieved_docs)
        
        docs_metadata = []
        for d in retrieved_docs:
            page_num = d.metadata.get("logical_page") or d.metadata.get("page", "-")
            docs_metadata.append({
                "source": d.metadata.get("source", "unknown"),
                "page": page_num,
                "preview": d.page_content[:100] + "..."
            })

        process_time = time.time() - start_time
        
        return ChatResponse(
            answer=final_answer,
            expanded_query=expanded_query,
            retrieved_docs=docs_metadata,
            processing_time=round(process_time, 2)
        )

    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rebuild-index", response_model=RebuildResponse)
async def rebuild_index_endpoint(background_tasks: BackgroundTasks):
    def task():
        logger.info("Rebuilding Index started...")
        try:
            dataset_path = os.getenv("DATASET_PATH", "dataset/")
            docs = doc_processor.ingest_manual(dataset_path) 
            rag_engine.build_index(docs)
            logger.info("Rebuild Complete!")
        except Exception as e:
            logger.error(f"Rebuild failed: {e}")

    background_tasks.add_task(task)
    return RebuildResponse(status="accepted", message="Rebuilding started in background. Check logs for progress.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)