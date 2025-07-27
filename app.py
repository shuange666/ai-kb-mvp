from fastapi import FastAPI, File, UploadFile, Form
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import shutil, os, uvicorn

app = FastAPI()
persist_dir = "./chroma"
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    tmp = f"/tmp/{file.filename}"
    with open(tmp, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    docs = PyPDFLoader(tmp).load_and_split(
        RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    )
    Chroma.from_documents(docs, embeddings, persist_directory=persist_dir)
    return {"status": "ok", "chunks": len(docs)}

@app.post("/chat")
async def chat(query: str = Form(...)):
    store = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY")),
        retriever=store.as_retriever()
    )
    return {"answer": qa.run(query)}
