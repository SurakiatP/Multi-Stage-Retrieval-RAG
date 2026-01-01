import os
import json
import logging
import pickle
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from pythainlp.tokenize import word_tokenize
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

def thai_tokenizer(text: str) -> List[str]:
    return word_tokenize(text, engine="newmm")

def results_to_toon_context(documents: List[Document]) -> str:
    if not documents:
        return ""
    
    toon_output = f"[{len(documents)}]{{source,page,content}}:\n"
    
    for doc in documents:
        source = doc.metadata.get('source', 'unknown')
        page = doc.metadata.get('logical_page', '0')
        
        raw_content = doc.page_content.replace('\n', ' ').strip()
        safe_content = raw_content.replace('"', "'")
        
        toon_output += f'  {source},{page},"{safe_content}"\n'
        
    return toon_output

class RAGEngine:
    def __init__(self, db_path=None):
        self.db_path = db_path or os.getenv("DATABASE_PATH", "database")
        self.faiss_path = os.path.join(self.db_path, "faiss_index")
        self.bm25_path = os.path.join(self.db_path, "bm25_retriever.pkl")
        
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-small")
        self.embedding_device = os.getenv("EMBEDDING_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        self.reranker_model_name = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-base")
        
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1100"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        self.retrieval_k = int(os.getenv("RETRIEVAL_K", "15"))
        self.rerank_top_n = int(os.getenv("RERANK_TOP_N", "5"))

        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)

        logger.info(f"Loading Embedding Model ({self.embedding_model_name})...")
        logger.info(f"Using device: {self.embedding_device}")
        
        batch_size = 64 if self.embedding_device == 'cuda' else 32
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': self.embedding_device}, 
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': batch_size
            }
        )
        
        logger.info(f"Loading Reranker Model ({self.reranker_model_name})...")
        self.reranker_model = HuggingFaceCrossEncoder(
            model_name=self.reranker_model_name,
            model_kwargs={'device': self.embedding_device}
        )
        
        self.vector_store = None
        self.bm25_retriever = None
        self.compression_retriever = None

    def load_documents_from_json(self, json_path: str) -> List[Document]:
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")
            
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        documents = []
        for item in data:
            content = item.get('content', '')
            
            if 'metadata' in item:
                metadata = item['metadata']
            else:
                metadata = {
                    "source": item.get('source', 'unknown'),
                    "logical_page": item.get('page', '0')
                }

            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
            
        logger.info(f"Loaded {len(documents)} source pages from JSON.")
        return documents

    def build_index(self, documents: List[Document]):
        if not documents:
            logger.warning("No documents to index!")
            return

        logger.info(f"Splitting documents (Chunk: {self.chunk_size}, Overlap: {self.chunk_overlap})...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,      
            chunk_overlap=self.chunk_overlap,   
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(documents)
        logger.info(f"Created {len(splits)} chunks from {len(documents)} pages.")

        logger.info("Building FAISS Vector Index...")
        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        self.vector_store.save_local(self.faiss_path)
        logger.info(f"FAISS index saved to {self.faiss_path}")

        logger.info("Building BM25 Keyword Index...")
        self.bm25_retriever = BM25Retriever.from_documents(
            splits, 
            preprocess_func=thai_tokenizer
        )
        self.bm25_retriever.k = self.retrieval_k
        
        with open(self.bm25_path, 'wb') as f:
            pickle.dump(self.bm25_retriever, f)
        logger.info(f"BM25 index saved to {self.bm25_path}")

        self._setup_retrieval_pipeline()

    def load_index(self):
        if os.path.exists(self.faiss_path) and os.path.exists(self.bm25_path):
            logger.info("Loading indexes from disk...")
            
            self.vector_store = FAISS.load_local(
                self.faiss_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            with open(self.bm25_path, 'rb') as f:
                self.bm25_retriever = pickle.load(f)
            
            logger.info("Indexes loaded successfully.")
            self._setup_retrieval_pipeline()
            return True
        else:
            logger.warning("Indexes not found. Please run build_index() first.")
            return False

    def _setup_retrieval_pipeline(self):
        if not self.vector_store or not self.bm25_retriever:
            raise ValueError("Indexes not loaded!")

        faiss_retriever = self.vector_store.as_retriever(search_kwargs={"k": self.retrieval_k})
        self.bm25_retriever.k = self.retrieval_k

        ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, faiss_retriever],
            weights=[0.4, 0.6] 
        )

        compressor = CrossEncoderReranker(model=self.reranker_model, top_n=self.rerank_top_n)
        
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble_retriever
        )
        logger.info("Retrieval Pipeline Ready (Hybrid + Rerank).")

    def search(self, query: str) -> List[Document]:
        if not self.compression_retriever:
            raise ValueError("Engine not ready! Load or Build index first.")
        
        logger.info(f"Searching for: '{query}'")
        return self.compression_retriever.invoke(query)

if __name__ == "__main__":
    engine = RAGEngine()
    
    json_input = "ingested_data/ingested_documents.json"
    
    if not engine.load_index():
        print("\n--- Building new index from JSON ---")
        if os.path.exists(json_input):
            docs = engine.load_documents_from_json(json_input)
            engine.build_index(docs)
            print("Build Complete!")
        else:
            print(f"Error: {json_input} not found.")
            exit()

    query = "ค่าเดินทางรถยนต์ส่วนตัวเบิกได้กี่บาท?"
    print(f"\nQuery: {query}")
    
    results = engine.search(query)
    
    toon_context = results_to_toon_context(results)
    
    print("\n--- FINAL CONTEXT FOR LLM (TOON FORMAT) ---")
    print(toon_context)
    print("-------------------------------------------")
    
    print(f"\n(Found {len(results)} chunks used for context)")