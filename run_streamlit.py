"""This is just a script to run on Streamlit Cloud
Does:
- create vector store
- launch streamlit app
"""

from pathlib import Path

import fitz
import streamlit as st
from langchain.docstore.document import Document
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from graph import workflow

doc_dir = "knowledge_base/pdf"

pdf_paths = sorted(list(Path(doc_dir).glob("*.pdf")))

parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3_000,
    chunk_overlap=50,
)

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=25,
)

fs = LocalFileStore("knowledge_base/local_file_store")
store = create_kv_docstore(fs)

vectorstore = Chroma(
    collection_name="parent_docs",
    embedding_function=OllamaEmbeddings(model="nomic-embed-text:latest"),
    persist_directory="knowledge_base/vectorstore",
)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k": 1},
)

docs = []

for p in pdf_paths:
    pdf_doc = fitz.open(str(p))
    pdf_str = ""
    for page in pdf_doc:
        pdf_str += page.get_text()

    # create langchian doc
    doc = Document(page_content=pdf_str)
    docs.append(doc)

retriever.add_documents(docs)


st.title("Starter Pack Chatbot")
st.markdown("by Kredivo")


def stream_data():
    for msg, metadata in workflow.stream(
        {
            "question": prompt,
        },
        stream_mode="messages",
    ):
        # print(metadata)
        if metadata["langgraph_node"] == "responder":
            stream = msg.content

            yield stream


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        print(prompt)

        response = st.write_stream(stream_data)
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )
