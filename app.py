from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import shutil, os, re, uvicorn

app = FastAPI(title="AI KB SaaS")

# 统一环境变量
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://www.chataiapi.com/v1")
PERSIST_DIR = "./data/chroma"

# 持久化向量库（重启不丢）
os.makedirs(PERSIST_DIR, exist_ok=True)
embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_API_BASE
)
vector_store = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
    collection_name="all_docs"
)

@app.get("/")
async def root():
    return {
        "status": "healthy",
        "message": "AI 客服知识库已上线",
        "endpoints": {
            "upload": "/upload (单文件)",
            "batch_upload": "/batch_upload (多文件)",
            "chat": "/chat (问答)"
        }
    }

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """单文件上传"""
    try:
        safe_name = re.sub(r'[^\w\.-]', '_', file.filename)
        if not safe_name.lower().endswith(".pdf"):
            return JSONResponse(status_code=400, content={"error": "请上传 .pdf 文件"})
        reader = PdfReader(file.file)
        text = "".join(page.extract_text() or "" for page in reader.pages)
        if not text.strip():
            return JSONResponse(status_code=400, content={"error": "PDF 无可读内容"})
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text)
        vector_store.add_texts(chunks)
        return {"status": "ok", "chunks": len(chunks)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/batch_upload")
async def batch_upload(files: list[UploadFile] = File(...)):
    """批量上传多个 PDF"""
    total_chunks = 0
    for file in files:
        try:
            if not file.filename.lower().endswith(".pdf"):
                continue
            reader = PdfReader(file.file)
            text = "".join(page.extract_text() or "" for page in reader.pages)
            if text.strip():
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_text(text)
                vector_store.add_texts(chunks)
                total_chunks += len(chunks)
        except Exception:
            continue  # 跳过损坏文件
    return {"status": "ok", "total_chunks": total_chunks}

@app.post("/chat")
async def chat(query: str = Form(...)):
    """AI 客服问答，检索全部已上传文档"""
    try:
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(
                openai_api_key=OPENAI_API_KEY,
                openai_api_base=OPENAI_API_BASE
            ),
            retriever=vector_store.as_retriever()
        )
        answer = qa.run(query)
        return {"answer": answer}
    except Exception as e:
        return JSONResponse(status_code=500, content={"answer": "服务器内部错误", "error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
