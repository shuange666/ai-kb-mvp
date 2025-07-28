from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import shutil
import os
import re
import uvicorn
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI KB MVP")
persist_dir = "./chroma"

# 确保存储目录存在
os.makedirs(persist_dir, exist_ok=True)

# 初始化Embeddings
embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE", "https://www.chataiapi.com/v1")
)

@app.get("/")
async def root():
    return {"message": "AI 知识库 MVP 已上线，请访问 /docs 查看接口文档。"}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    # 清理文件名，防止中文/空格导致问题
    safe_name = re.sub(r'[^\w\.-]', '_', file.filename)
    tmp_path = f"/tmp/{safe_name}"
    
    try:
        # 保存临时文件
        with open(tmp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 加载和分割PDF
        loader = PyPDFLoader(tmp_path)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = loader.load_and_split(text_splitter)
        
        # 创建向量存储
        Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_dir
        )
        
        return {
            "status": "ok",
            "chunks": len(docs),
            "filename": safe_name
        }
        
    except Exception as e:
        logger.error(f"文件上传失败: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"文件处理失败: {str(e)}"}
        )
        
    finally:
        # 确保删除临时文件
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception as e:
                logger.warning(f"临时文件删除失败: {str(e)}")

@app.post("/chat")
async def chat(query: str = Form(...)):
    try:
        # 检查知识库是否存在
        if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
            return JSONResponse(
                status_code=400,
                content={"answer": "请先上传文档后再提问。"}
            )
        
        # 加载向量存储
        store = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
        
        # 初始化问答链
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                openai_api_base=os.getenv("OPENAI_API_BASE", "https://www.chataiapi.com/v1"),
                temperature=0
            ),
            chain_type="stuff",
            retriever=store.as_retriever()
        )
        
        # 执行查询
        result = qa.invoke({"query": query})
        return {"answer": result["result"]}
        
    except Exception as e:
        logger.error(f"查询处理失败: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "answer": "查询处理失败",
                "error": str(e)
            }
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
