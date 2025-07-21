import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

from transformers import pipeline
import gradio as gr

# Load env
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

# Initialize chain
print("Loading documents...")
documents = load_docs(DOCS_FOLDER)
print(f"Loaded {len(documents)} documents.")

print("Splitting documents...")
chunks = split_docs(documents)

print("Creating/retrieving vector store...")
vectorstore = get_vectorstore(chunks)

print("Loading HuggingFace LLM...")
hf_pipeline = pipeline(
    "text-generation",
    model="sshleifer/tiny-gpt2",  # light-weight model for low-memory deployment
    max_new_tokens=128,
    temperature=0.7,
    do_sample=True
)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    vectorstore.as_retriever()
)

chat_history = []

def chat(query, history):
    global chat_history
    result = qa_chain({"question": query, "chat_history": chat_history})
    answer = result["answer"]
    chat_history.append((query, answer))
    return history + [{"role": "user", "content": query}, {"role": "assistant", "content": answer}], ""

def clear_chat():
    global chat_history
    chat_history = []
    return [], ""

# 🌈 Gradio app
custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="gray",
)

with gr.Blocks(
    theme=custom_theme,
    css="""
    button {
        background: linear-gradient(90deg, #4e54c8, #8f94fb) !important;
        color: white !important;
        border-radius: 8px !important;
    }
    .gradio-container {
        background: #f5f5f5;
    }
    """
) as demo:
    gr.HTML("""
    <div style="text-align:center; background:#4e54c8; color:white; padding:1rem; border-radius:10px; margin-bottom:10px; font-size:20px;">
        📄 <strong>Internal Docs Q&A Bot 🤖</strong>
        <div style="font-size:14px;">Ask your questions below and get instant answers!</div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="💬 Chat History",
                height=450,
                show_copy_button=True,
                bubble_full_width=False
            )
        with gr.Column(scale=1):
            gr.HTML("""
            <div style="background: #e0f7fa; padding:10px; border-radius:8px; font-size:14px;">
            🌟 <strong>Instructions:</strong><br>
            - Type your question below.<br>
            - Press <strong>Enter</strong> or <strong>Submit</strong>.<br>
            - Click 🧹 to clear the chat.<br>
            </div>
            """)
            clear_btn = gr.Button("🧹 Clear Chat")

    with gr.Row():
        msg = gr.Textbox(
            label="💌 Your Question",
            placeholder="E.g., What is the company leave policy?",
            lines=2,
        )

    msg.submit(chat, [msg, chatbot], [chatbot, msg])
    clear_btn.click(clear_chat, outputs=[chatbot, msg])

    gr.Markdown("---")
    gr.HTML("""
    <div style="text-align:center; font-size:12px; color:gray;">
    ⚡ Powered by <a href='https://www.langchain.com/' target='_blank'>LangChain</a> & 
    <a href='https://gradio.app/' target='_blank'>Gradio</a>
    </div>
    """)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
