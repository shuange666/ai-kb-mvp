from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import shutil, os, re, uvicorn

app = FastAPI(title="AI KB MVP")
persist_dir = "./chroma"
embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE", "https://www.chataiapi.com/v1")
)

@app.get("/")
async def root():
    return {"status": "healthy", "endpoints": {"upload": "/upload (POST)", "chat": "/chat (POST)"}}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        safe_name = re.sub(r'[^\w\.-]', '_', file.filename)
        tmp_path = f"/tmp/{safe_name}"
        with open(tmp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 用 pypdf 读取
        reader = PdfReader(tmp_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        if not text.strip():
            return JSONResponse(status_code=400, content={"error": "PDF 无可读内容，请换文件"})

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text)
        Chroma.from_texts(chunks, embeddings, persist_directory=persist_dir)
        return {"status": "ok", "chunks": len(chunks)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/chat")
async def chat(query: str = Form(...)):
    try:
        store = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                openai_api_base=os.getenv("OPENAI_API_BASE", "https://www.chataiapi.com/v1")
            ),
            retriever=store.as_retriever()
        )
        return {"answer": qa.run(query)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"answer": "请先上传有效 PDF 文档。", "error": str(e)})
