from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import shutil, os, uvicorn

app = FastAPI(title="AI KB MVP")

persist_dir = "./chroma"
embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE", "https://www.chataiapi.com/v1")
)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    tmp = f"/tmp/{file.filename}"
    with open(tmp, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    loader = PyPDFLoader(tmp)
    docs = loader.load_and_split(
        RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    )
    Chroma.from_documents(docs, embeddings, persist_directory=persist_dir)
    return {"status": "ok", "chunks": len(docs)}

@app.post("/chat")
async def chat(query: str = Form(...)):
    store = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base=os.getenv("OPENAI_API_BASE", "https://www.chataiapi.com/v1")
        ),
        retriever=store.as_retriever()
    )
    answer = qa.run(query)
    return {"answer": answer}
