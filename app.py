from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import shutil, os, re, uvicorn

app = FastAPI(title="AI KB MVP")
persist_dir = "./chroma"

# 兼容 DeepSeek 中转
embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE", "https://www.chataiapi.com/v1")
)

@app.get("/")
async def root():
    return {"message": "AI 知识库 MVP 已上线，请访问 /docs 查看接口文档。"}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    # 清理文件名，防止中文/空格导致 500
    safe_name = re.sub(r'[^\w\.-]', '_', file.filename)
    tmp_path = f"/tmp/{safe_name}"
    try:
        with open(tmp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        loader = PyPDFLoader(tmp_path)
        docs = loader.load_and_split(
            RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        )
        Chroma.from_documents(docs, embeddings, persist_directory=persist_dir)
        return {"status": "ok", "chunks": len(docs)}
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
        return JSONResponse(status_code=500, content={"answer": "请先上传文档后再提问。", "error": str(e)})
