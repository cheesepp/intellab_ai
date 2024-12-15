"""
FILE FOR HANDLING DATA LOADING FROM DOCUMENTS AND WEB SEARCHES
"""

import re
import shutil
from typing import Iterator, List
from langchain_community.document_loaders import web_base
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.retrievers import BaseRetriever
import PyPDF2
from langchain_community.vectorstores import Chroma
from langchain.vectorstores.pgvector import PGVector
from dotenv import load_dotenv
import os
import glob
load_dotenv()
# Regular expression to match newlines
NEWLINE_RE = re.compile("\n+")
CHROMA_PATH = "chroma"

# URLs to be used for loading documents
SOURCE_URLS = [
    'https://pandas.pydata.org/docs/user_guide/indexing.html',
    'https://pandas.pydata.org/docs/user_guide/groupby.html',
    'https://pandas.pydata.org/docs/user_guide/merging.html'
]

class LangGraphDocsLoader(web_base.WebBaseLoader):
    def lazy_load(self) -> Iterator[Document]:
        """Lazy load text from the url(s) in web_path."""
        for path in self.web_paths:
            soup = self._scrape(path, bs_kwargs=self.bs_kwargs)
            text = soup.get_text(**self.bs_get_text_kwargs)
            text = NEWLINE_RE.sub("\n", text)
            metadata = web_base._build_metadata(soup, path)
            yield Document(page_content=text, metadata=metadata)

    def load_from_pdf(self, file_path: str) -> List[Document]:
        """Load text from a PDF file."""
        documents = []
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text = page.extract_text()
                    text = NEWLINE_RE.sub("\n", text)
                    documents.append(Document(page_content=text, metadata={"source": file_path, "page": page_num}))
        except Exception as e:
            print(f"Error loading PDF file: {e}")
        return documents

    def load_from_text(self, file_path: str) -> List[Document]:
        """Load text from a text file."""
        documents = []
        try:
            with open(file_path, 'r') as file:
                text = file.read()
                text = NEWLINE_RE.sub("\n", text)
                documents.append(Document(page_content=text, metadata={"source": file_path}))
        except Exception as e:
            print(f"Error loading text file: {e}")
        return documents

def prepare_documents(urls: list[str], pdf_paths: list[str] = [], text_paths: list[str] = []) -> list[Document]:
    """Prepare documents by loading and splitting text."""
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[
            r"In \[[0-9]+\]",
            r"\n+",
            r"\s+"
        ],
        is_separator_regex=True,
        chunk_size=1000
    )
    docs = [LangGraphDocsLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    loader = LangGraphDocsLoader()
    for pdf_path in pdf_paths:
        docs_list.extend(loader.load_from_pdf(pdf_path))
    for text_path in text_paths:
        docs_list.extend(loader.load_from_text(text_path))

    return text_splitter.split_documents(docs_list)

def persist_data():
    """Get a retriever for the documents."""
    # Define the directory and file pattern using relative paths
    pdf_directory = "../../data/"
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Search for PDF files in the directory
    pdf_files = glob.glob(os.path.join(pdf_directory, '*.pdf'))
    # Convert absolute paths to relative paths
    pdf_files = [os.path.relpath(pdf_file, start=os.path.dirname(__file__)) for pdf_file in pdf_files]
    
    # Load and print PDF contents
    loader = LangGraphDocsLoader()
    for pdf_path in pdf_files[:1]:
        documents = loader.load_from_pdf(pdf_path)
        print(documents)
        for doc in documents:
            print(f"Contents of {pdf_path}:\n{doc.page_content}\n")
    documents = prepare_documents(SOURCE_URLS, pdf_paths=[], text_paths=[])
    vectorstore = Chroma.from_documents(
        documents=documents,
        collection_name="pandas-rag-chroma",
        embedding=OllamaEmbeddings(model="llama3.2", base_url="http://host.docker.internal:11434"),
        persist_directory=CHROMA_PATH
    )
    vectorstore.persist()

def get_retriever() -> BaseRetriever:
    # Define embedding model
    embedding = OllamaEmbeddings(model="llama3.2", base_url="http://host.docker.internal:11434")

    # Load existing vector store
    vectorstore = Chroma(
        collection_name="pandas-rag-chroma",
        embedding=embedding,
        persist_directory=CHROMA_PATH
    )
    
    return vectorstore.as_retriever(search_kwargs={"k": 5})  # Top 5 results