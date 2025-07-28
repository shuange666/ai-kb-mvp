from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import shutil
import os
import re
import uvicorn
import logging
from typing import List

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Knowledge Base API",
    description="API for document upload and querying",
    version="1.0.0"
)

# 配置
PERSIST_DIR = "./chroma_db"
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# 初始化Embeddings
embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE"),  # 确保使用变量
    model="text-embedding-3-small"
)

# 确保存储目录存在
os.makedirs(PERSIST_DIR, exist_ok=True)

@app.get("/")
async def health_check():
    return {
        "status": "healthy",
        "message": "AI Knowledge Base API is running",
        "endpoints": {
            "upload": "/upload (POST)",
            "chat": "/chat (POST)"
        }
    }

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Handle file upload and create vector store"""
    try:
        # 1. 验证文件类型
        file_ext = file.filename.split('.')[-1].lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}"
            )

        # 2. 验证文件大小
        file.file.seek(0, os.SEEK_END)
        file_size = file.file.tell()
        file.file.seek(0)
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {MAX_FILE_SIZE//(1024*1024)}MB"
            )

        # 3. 安全保存临时文件
        safe_name = re.sub(r'[^\w\.-]', '_', file.filename)
        tmp_path = f"/tmp/{safe_name}"
        
        with open(tmp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 4. 加载文档
        loader = {
            'pdf': PyPDFLoader,
            'txt': TextLoader,
            'docx': Docx2txtLoader
        }[file_ext](tmp_path)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = loader.load_and_split(text_splitter)

        # 5. 验证文档内容
        if not docs or all(len(doc.page_content.strip()) == 0 for doc in docs):
            raise ValueError("No readable content found in document")

        # 6. 创建向量存储
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=PERSIST_DIR
        )

        # 7. 验证嵌入结果
        collection = vectordb.get()
        if not collection['documents']:
            raise ValueError("Failed to generate document embeddings")

        return {
            "status": "success",
            "filename": safe_name,
            "chunks": len(docs),
            "first_chunk": docs[0].page_content[:100] + "..." if docs else ""
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "solution": [
                    "Check if the file contains readable text",
                    "Verify your OpenAI API key",
                    "Try a different file format"
                ]
            }
        )
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/chat")
async def chat(query: str = Form(...)):
    """Handle user queries against the knowledge base"""
    try:
        # 1. 检查知识库是否存在
        if not os.path.exists(PERSIST_DIR) or not os.listdir(PERSIST_DIR):
            raise HTTPException(
                status_code=400,
                detail="No documents uploaded yet. Please upload documents first."
            )

        # 2. 加载向量存储
        vectordb = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )

        # 3. 初始化问答链
        qa_chain = RetrievalQA.from_chain_type(
            llm=OpenAI(
                model_name="gpt-3.5-turbo-instruct",
                temperature=0,
                max_tokens=1000,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                openai_api_base=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
            ),
            chain_type="stuff",
            retriever=vectordb.as_retriever(),
            return_source_documents=True
        )

        # 4. 执行查询
        result = qa_chain.invoke({"query": query})
        
        return {
            "answer": result["result"],
            "source_documents": [
                {
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata
                } 
                for doc in result["source_documents"]
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "solution": [
                    "Check if the knowledge base is properly initialized",
                    "Verify your OpenAI API key",
                    "Try a simpler query"
                ]
            }
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
