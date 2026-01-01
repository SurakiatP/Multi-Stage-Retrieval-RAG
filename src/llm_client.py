import os
import logging
import re
from typing import List
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, model_name=None):
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.expand_model_name = os.getenv("LLM_EXPAND_MODEL_NAME", "scb10x/typhoon2.1-gemma3-4b:latest")
        self.generate_model_name = os.getenv("LLM_GENERATE_MODEL_NAME", "qwen2.5:7b-instruct-q4_0")
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
        timeout = float(os.getenv("LLM_TIMEOUT", "120.0"))
        
        logger.info(f"Initializing Expander LLM: {self.expand_model_name}")
        self.llm_expand = ChatOllama(
            model=self.expand_model_name,
            base_url=base_url,
            temperature=temperature,
            keep_alive="1h",
            num_predict=100,
            request_timeout=timeout
        )

        logger.info(f"Initializing Generator LLM: {self.generate_model_name}")
        self.llm_generate = ChatOllama(
            model=self.generate_model_name,
            base_url=base_url,
            temperature=temperature,
            keep_alive="1h",
            num_predict=350,
            request_timeout=timeout
        )

    def expand_query(self, query: str) -> str:
        logger.info(f"Expanding query: '{query}'")
        
        system_prompt = """You are a Bilingual Search Query Optimizer for Corporate Policy and Employee Handbook queries.

TASK: Convert user queries into alternating Thai-English keyword pairs for hybrid search.

RULES:
1. Extract 2-6 core concepts from the query.
2. For each concept, provide:
   - Thai translation (formal + colloquial terms)
   - English technical term or specific policy name (e.g., WFH, Per Diem, Probation)
3. Output format: <Thai1> <English1> <Thai2> <English2> [<Thai3> <English3>]
4. Use ONLY spaces as separators (no commas, pipes, or special characters).
5. **OUTPUT KEYWORDS ONLY** - NO explanations, NO reasoning, NO query repetition.

EXAMPLES:

Input: "How many days for sick leave?"
Output: วันลาป่วย Sick Leave จำนวนวัน Days กฎระเบียบ Policy ใบรับรองแพทย์ Medical Certificate

Input: "Travel allowance rates for 2025"
Output: ค่าเดินทาง Travel Allowance อัตรา Rates ปี 2568 2025 ค่าพาหนะ Transportation เบิกจ่าย Reimbursement

Input: "Work from home requirements"
Output: การทำงานที่บ้าน Work from Home นโยบาย WFH การเข้าออฟฟิศ Office Attendance ข้อกำหนด Requirements

Query:"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{query}")
        ])
        
        chain = prompt | self.llm_expand | StrOutputParser()
        
        try:
            keywords = chain.invoke({"query": query})
            keywords = keywords.replace('\n', ' ').replace('"', '').replace('Output: ','').strip()
            keywords = re.sub(r'(Here is|The translation|However).*', '', keywords, flags=re.IGNORECASE)
            
            expanded_query = f"{keywords}"
            logger.info(f"Expanded: {expanded_query}")
            return expanded_query
        except Exception as e:
            logger.error(f"Expansion failed: {e}")
            return query 

    def generate_answer(self, query: str, context_docs: List[Document]) -> str:
        if not context_docs:
            return "I cannot find relevant information in the provided documents."

        logger.info(f"Generating answer from {len(context_docs)} documents...")

        # --- Construct TOON Context String ---
        # Header: [Count]{source,page,content}:
        context_text = f"[{len(context_docs)}]{{source,page,content}}:\n"
        
        for doc in context_docs:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("logical_page", "0")
            
            # Clean Content: Flatten to single line and escape double quotes
            content = doc.page_content.replace("\n", " ").strip()
            content = content.replace('"', "'") 
            
            # Row: source,page,"content"
            context_text += f'  {source},{page},"{content}"\n'
        
        # --- System Prompt ---
        system_prompt = """You are a Corporate Policy Assistant answering questions using ONLY the provided Context.

CORE RULES:
1. Base answers EXCLUSIVELY on the Context provided below.
2. The Context is formatted in **TOON (Tabular)** format: `[N]{{source,page,content}}:`.
3. Cite every claim immediately using: `[Source: filename, Page: X]`.
4. Output in ENGLISH only (translate Thai terms to English).
5. If answer is not in Context, respond: "I cannot find this information in the provided documents."

CONFLICT RESOLUTION (CRITICAL):
- If you find conflicting information (e.g., between an old 'Policy 2024' and a new 'Memo 2025'), **PRIORITIZE the document with the LATEST date** or labeled as 'Update/Memo'.
- Explicitly state that the old rule has been superseded by the new one.

EXAMPLES:

Example 1: Single Source
Context:
[1]{{source,page,content}}:
  Employee-Handbook.pdf,4,"Standard working hours are Mon-Fri, 09:00 - 18:00."
Question: What are working hours?
Answer: Standard working hours are Monday to Friday, from 09:00 to 18:00 [Source: Employee-Handbook.pdf, Page: 4].

Example 2: Conflict Resolution (Policy vs Memo)
Context:
[2]{{source,page,content}}:
  Benefits-Policy-2024.pdf,8,"Personal car mileage rate is 8 THB per km."
  Internal-Memo-2025.pdf,1,"Effective Jan 2, 2025, mileage rate increases to 12 THB per km."
Question: What is the mileage rate?
Answer: The current mileage reimbursement rate is 12 THB per km. The Internal Memo [Source: Internal-Memo-2025.pdf, Page: 1] explicitly updates and overrides the previous rate of 8 THB found in the 2024 policy [Source: Benefits-Policy-2024.pdf, Page: 8].

Context:
{context}

Question: {question}

Answer:"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "User Question: {question}")
        ])

        chain = prompt | self.llm_generate | StrOutputParser()

        try:
            response = chain.invoke({
                "context": context_text,
                "question": query
            })
            response = response.strip()
            return response
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return "Sorry, I encountered an error while generating the answer."

if __name__ == "__main__":
    try:
        client = LLMClient()
        
        q = "How many days can I work from home per week?"
        print(f"\n--- Test Expansion ---")
        print(client.expand_query(q))
        
        print(f"\n--- Test Generation (TOON Format) ---")
        # Mocking documents to test conflict resolution
        mock_docs = [
            Document(
                page_content="พนักงานทุกคนต้องเข้ามาทำงานที่สำนักงานอย่างน้อย 3 วันต่อสัปดาห์ (Hybrid Work Policy เริ่ม 2025)", 
                metadata={"source": "Internal-Memo-2025.pdf", "logical_page": "3"}
            ),
            Document(
                page_content="พนักงานสามารถทำงานที่บ้านได้ 2 วันต่อสัปดาห์ แบบยืดหยุ่น", 
                metadata={"source": "Benefits-Policy-2024.pdf", "logical_page": "15"}
            )
        ]
        
        answer = client.generate_answer(q, mock_docs)
        print(f"\nModel Answer:\n{answer}")
        
    except Exception as e:
        print(f"Error: {e}")