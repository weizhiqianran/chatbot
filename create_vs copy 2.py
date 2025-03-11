from pathlib import Path

import fitz
from langchain.docstore.document import Document
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# 加载环境变量（可选）
import os
from dotenv import load_dotenv
load_dotenv()

# 从环境变量获取 Ollama 的 base URL 和 embedding model
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")  # 默认本地 URL
embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest")  # 默认模型

# PDF 文件目录
doc_dir = "knowledge_base/pdf"
pdf_paths = sorted(list(Path(doc_dir).glob("*.pdf")))

# 定义分片器
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3_000,
    chunk_overlap=50,
)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=25,
)

# 设置存储
fs = LocalFileStore("knowledge_base/local_file_store")
store = create_kv_docstore(fs)

# 初始化向量存储，使用指定的 Ollama base URL 和 embedding model
vectorstore = Chroma(
    collection_name="parent_docs",
    embedding_function=OllamaEmbeddings(
        model=embedding_model,
        base_url=ollama_base_url  # 指定 Ollama 的 base URL
    ),
    persist_directory="knowledge_base/vectorstore",
)

# 设置检索器
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k": 1},
)

# 处理 PDF 文件
docs = []
for p in pdf_paths:
    pdf_doc = fitz.open(str(p))
    pdf_str = ""
    for page in pdf_doc:
        pdf_str += page.get_text()

    # 创建 LangChain 文档
    doc = Document(page_content=pdf_str)
    docs.append(doc)

# 将文档添加到检索器
retriever.add_documents(docs)

print(f"文档已处理并添加到检索器，使用模型: {embedding_model}，Ollama base URL: {ollama_base_url}")