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

# Load environment variables
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
    model="distilgpt2",
    max_new_tokens=256,
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


# 🌈 Decent, formal & attractive UI with Gradio
custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="teal",
    neutral_hue="slate",
)

with gr.Blocks(
    theme=custom_theme,
    css="""
    button {
        background: linear-gradient(90deg, #4a90e2, #50e3c2) !important;
        color: white !important;
        border-radius: 6px !important;
        font-weight: bold;
    }
    .gradio-container {
        background: linear-gradient(135deg, #f7f9fc, #e6eff7);
    }
    .message.user {
        background-color: #d9e8fb !important;
        border-radius: 8px;
        padding: 6px;
        color: #003366;
    }
    .message.assistant {
        background-color: #d3f4ef !important;
        border-radius: 8px;
        padding: 6px;
        color: #003333;
    }
    .clear-btn {
        background-color: #e0f0ff !important;
        color: #003366 !important;
        font-weight: bold;
    }
    .question-box textarea {
        background-color: #f9fbfd;
        border-radius: 6px;
        border: 1px solid #cce0f5;
        padding: 8px;
    }
    """
) as demo:
    gr.HTML("""
    <div style="text-align:center; background:linear-gradient(to right,#4a90e2,#50e3c2); color:white; padding:1rem; border-radius:8px; margin-bottom:10px; font-size:20px;">
        📄 <strong>Internal Docs Q&A Bot 🤖</strong>
        <div style="font-size:14px;">Ask your questions below and get clear, instant answers!</div>
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
            <div style="background: #f0f4f8; padding:10px; border-radius:8px; font-size:14px;">
            🌟 <strong>Instructions:</strong><br>
            - Type your question in the box below.<br>
            - Press <strong>Enter</strong> or <strong>Submit</strong>.<br>
            - Click 🧹 to clear the chat and start fresh.<br>
            </div>
            """)
            clear_btn = gr.Button("🧹 Clear Chat", elem_classes="clear-btn")

    with gr.Row():
        msg = gr.Textbox(
            label="💌 Your Question",
            placeholder="E.g., What is the company leave policy?",
            lines=2,
            elem_classes="question-box"
        )

    msg.submit(chat, [msg, chatbot], [chatbot, msg])
    clear_btn.click(clear_chat, outputs=[chatbot, msg])

    gr.Markdown("---")
    gr.HTML("""
    <div style="text-align:center; font-size:12px; color:gray;">
    ⚡ Made with ❤️ using <a href='https://www.langchain.com/' target='_blank'>LangChain</a> & 
    <a href='https://gradio.app/' target='_blank'>Gradio</a> | 🌟 Customized by <strong>You</strong>
    </div>
    """)

demo.launch()
