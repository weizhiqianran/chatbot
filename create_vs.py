from pathlib import Path
import fitz
from langchain.docstore.document import Document
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings  # 添加 OpenAI 支持

import os
from dotenv import load_dotenv
load_dotenv()

# 从环境变量获取模型类型和参数
model_type = os.getenv("EMBEDDING_MODEL_TYPE", "ollama").lower()
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
embedding_model_ollama = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest")
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_base_url = os.getenv("OPENAI_BASE_URL", None)  # 可选 vLLM 端点
embedding_model_openai = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")

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

# 动态选择嵌入模型
if model_type == "ollama":
    embedding_function = OllamaEmbeddings(
        model=embedding_model_ollama,
        base_url=ollama_base_url
    )
elif model_type == "openai":
    embedding_function = OpenAIEmbeddings(
        model=embedding_model_openai,
        api_key=openai_api_key,
        base_url=openai_base_url
    )
else:
    raise ValueError(f"Unsupported embedding model type: {model_type}. Use 'ollama' or 'openai'.")

# 初始化向量存储
vectorstore = Chroma(
    collection_name="parent_docs",
    embedding_function=embedding_function,
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

print(f"文档已处理并添加到检索器，使用模型类型: {model_type}, 嵌入模型: {embedding_model_ollama if model_type == 'ollama' else embedding_model_openai}")