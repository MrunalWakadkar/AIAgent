import os
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain

from transformers import pipeline

# Load .env (optional)
load_dotenv()

DOCS_FOLDER = "internal_docs"
VECTOR_STORE_FOLDER = "vectorstore"

def load_docs(folder):
    docs = []
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif filename.endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(file_path)
            elif filename.endswith(".txt"):
                loader = TextLoader(file_path)
            else:
                print(f"Skipping unsupported file: {filename}")
                continue
            docs.extend(loader.load())
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
    return docs

def split_docs(docs, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists(VECTOR_STORE_FOLDER):
        print("Loading existing vector store...")
        return FAISS.load_local(
            VECTOR_STORE_FOLDER,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        print("Creating new vector store...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(VECTOR_STORE_FOLDER)
        return vectorstore

def main():
    print("Loading documents...")
    documents = load_docs(DOCS_FOLDER)
    print(f"Loaded {len(documents)} documents.")

    print("Splitting documents...")
    chunks = split_docs(documents)

    print("Creating/retrieving vector store...")
    vectorstore = get_vectorstore(chunks)

    print("Loading HuggingFace LLM...")
    # Uses distilgpt2 or similar, you can replace with bigger models if you have GPU
    hf_pipeline = pipeline(
        "text-generation",
        model="distilgpt2",
        max_length=512,
        temperature=0
    )
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        vectorstore.as_retriever()
    )

    chat_history = []
    print("\n=== Internal Docs Q&A ===")
    while True:
        query = input("\nYou: ")
        if query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        result = qa_chain({"question": query, "chat_history": chat_history})
        answer = result["answer"]
        print(f"\nAI: {answer}")
        chat_history.append((query, answer))

if __name__ == "__main__":
    main()
