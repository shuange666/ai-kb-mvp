from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from pydantic import BaseModel
import uvicorn, os, uuid, json, datetime

# 全局配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://www.chataiapi.com/v1")
PERSIST_DIR = "./data"
os.makedirs(PERSIST_DIR, exist_ok=True)

app = FastAPI(title="AI 企业 SaaS")

# CORS 允许前端跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 持久化向量库
embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_API_BASE
)
vector_store = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
    collection_name="all_docs"
)

# 额度文件
CREDIT_FILE = "credits.json"

def load_credits():
    if not os.path.exists(CREDIT_FILE):
        return {}
    return json.load(open(CREDIT_FILE))

def save_credits(data):
    json.dump(data, open(CREDIT_FILE, "w"))

# 工具：扣除额度
def deduct_credit(user_id: str, amount: int):
    credits = load_credits()
    current = credits.get(user_id, 0)
    if current < amount:
        raise HTTPException(status_code=402, detail="额度不足")
    credits[user_id] = current - amount
    save_credits(credits)
    return credits[user_id]

# 通用问答
@app.post("/chat")
def chat(user_id: str = Form(...), query: str = Form(...)):
    deduct_credit(user_id, 1)
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=OPENAI_API_BASE
        ),
        retriever=vector_store.as_retriever()
    )
    answer = qa.run(query)
    return {"answer": answer, "credit_left": load_credits().get(user_id, 0)}

# 上传 PDF
@app.post("/upload")
def upload_pdf(user_id: str = Form(...), file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="请上传 PDF")
    reader = PdfReader(file.file)
    text = "".join(page.extract_text() or "" for page in reader.pages)
    if not text.strip():
        raise HTTPException(status_code=400, detail="PDF 无可读内容")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    vector_store.add_texts(chunks)
    return {"total_chunks": len(chunks)}

# 充值额度
@app.post("/credit/topup")
def topup_credit(user_id: str = Form(...), amount: int = Form(...)):
    credits = load_credits()
    credits[user_id] = credits.get(user_id, 0) + amount
    save_credits(credits)
    return {"credit": credits[user_id]}

# 查询额度
@app.get("/credit/{user_id}")
def get_credit(user_id: str):
    return {"credit": load_credits().get(user_id, 0)}
