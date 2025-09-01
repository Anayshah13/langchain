
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import tempfile


st.set_page_config(
    page_title="Anay's PDF RAG Chat",
    page_icon="⚡",
    layout="wide"
)

st.markdown("""
    <style>
        /* Background gradient */
        .stApp {
            background: linear-gradient(110deg, #010101, #0a1f44, #1e6091);
            color: white;
            font-family: 'Poppins', sans-serif;
        }

        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;500;700&display=swap');

        .title-container {
            text-align: center;
            margin-bottom: 10px;
        }
        .title-container h1 {
            font-family: 'Poppins', sans-serif;
            font-weight: 700;
            color: linear-gradient(0deg, #98FBCB, #0077BE);
        }
        .title-container p {
            font-family: 'Poppins', sans-serif;
            font-weight: 300;
            font-size: 16px;
            color: linear-gradient(0deg, #98FBCB, #0077BE);
        }

        .chat-message {
            display: flex;
            align-items: flex-start;
            gap: 12px;
            margin: 10px 0;
        }

        .chat-message img {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            object-fit: cover;
            border: 2px solid #ffffff33;
        }

        .chat-bubble {
            border-radius: 12px;
            padding: 12px 15px;
            max-width: 80%;
            line-height: 1.4;
            font-size: 15px;
        }

        .user-bubble {
            background-color: rgba(0, 123, 255, 0.2);
            border-left: 4px solid #00b4d8;
        }

        .assistant-bubble {
            background-color: rgba(255, 255, 255, 0.1);
            border-left: 4px solid #ffd166;
        }

        .animated-bar {
            height: 5px;
            width: 100%;
            background: linear-gradient(-45deg, #000000, #0077b6, #48cae4);
            background-size: 400% 400%;
            animation: gradientMove 10s ease infinite;
            margin-bottom: 20px;
            border-radius: 10px;
        }

        @keyframes gradientMove {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }
    </style>
""", unsafe_allow_html=True)


st.markdown("<div class='animated-bar'></div>", unsafe_allow_html=True)
st.markdown("""
    <div class="title-container">
        <h1>📄 RAG with Google Gemini ⚡🖥️</h1>
        <p>Upload a PDF and chat with it. Your past Q&A will be remembered during this session.</p>
    </div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📂 Upload your PDF", type="pdf")

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        temp_path = tmp_file.name

    loader = PyPDFLoader(temp_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectordb = Chroma.from_documents(splits, embedding=embeddings)
    retriever = vectordb.as_retriever()

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=st.session_state.memory,
    )

    st.subheader("💬 Chat with your PDF")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.chat_input("Ask something about the PDF...")

    if query:
        with st.spinner("Thinking..."):
            result = qa_chain({"question": query})
            answer = result["answer"]
        st.session_state.chat_history.append({"role": "user", "content": query})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    user_img_url = "data:image/jpeg;base64,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"  # Change to your own
    assistant_img_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAOEAAADhCAMAAAAJbSJIAAAAdVBMVEUAAAD////6+vq6uro4ODjGxsaWlpY9PT1LS0vBwcFGRkanp6eCgoJtbW3c3NxycnLn5+djY2ORkZHy8vIjIyMQEBDLy8uvr69fX1/T09NBQUGfn59nZ2fj4+MVFRWKiootLS13d3ccHBwxMTFRUVEgICB+fn5X1pSiAAAD6klEQVR4nO3ZWXeiMACGYTYVRSuIu1Vwaf//TxyWJCxJ7Mw5U/TifW7mjAmYz4QkpI4DAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP9RNvZHtXH2mXfLltFuZDLtVltPSkG4FK778tPPsd3pMVA6x7nf9m6LF/vXVmngWrTqLPzY6xeHjvP4sF1bOmwGS3hMnVnv273oSxVPtMbXVVSFlaGClzuO/yyfOyqunA7UjYF7dj61Jp5UeT9+N2EYm0r3Tn5+GvCjuPTgzoZKWESc623IZYXFk4TGMvfTWRuDK3ExRLfugAnd2PnWM0xkjaU1odb3laTsnx8C3sqfYMCERcSr1o5UzQUrS8K8/wh65QfFNKrfrO28EX08ZMJioO6Xi8XiO2pNgFtVZxbOpGDbJGx6yjtE1/0svN/DMCgKns6iHzdnllYX5UMmdNW8FkSqLQtT/alKqB7COFt3qlzE59N546F+oqJCWt9jbbr//ycShs0nX7ILPkz1m4RyNln1q4jp92S4uua9OGE1z+kfCiqh3CgkWhUxAe1tX7l+fcJbPYzcsaG+ShhZeyqsSzLbVx7dlyeUs+fIUF8lrAdpath9iVumueUrJ2+QUDwppgdRJjzafwRHriKdzbofjcUSO3nymP8Cc8LRjwnFPiAy3dO6J51WPf4WCaMfE0ZPHrajLaF7vjhvknAn2nDZFDolMmFdw7Ik7KwRy1H9Fglj0Ybqn3TXKukl7L0Kd2uZ7N8j4Vq2Qew+tLZ7zqmuEltum6WWhNv3SBjJIVXPOO68KZIJ5Z7Ntq5fZl2J3EaohMZp+BeYEgaeDCbeqtKmTCa8e//aE1l9wfUdEsqN6UxusFqtUSu+7JOddkubeuCeXp9wI9tedo8Yps26pxKq19+/jZi7L03YnAp9qzliWfwvkf/Zij3WSCZs3gJj42uWZvTShOfNZrkPgmTuN3PgoSpv5v1pcr9cxF6mTNha11M/2weTSbkrOy4a86hFnk19y4T+kAnPG/29XLyCr9tHFWmxNGYyoeF0Y6fmE7ubTKi9V/5iwth08CDf+3rHTbtqJaxXSH3/OVYrjY3vvCChKWB6VDV6XVVFFHsAPc3ph4jp+gUJR4aA0/ZmdNE9U4uKDHKXc9VOvP2nEb1yZAyccHbQA4565+33bac4K6ZFWZRrO1D/ydtTWi1KAyf8ci7dE/jDKtdrPTqNXjmtDXfYzzhtdgM9u3pkHOUPNZQk8qdx6eBHmf2vJY9VVP1dLEk+e3vRzcpv2xX3iHzdSY78S1KyHlQBAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIz+ABbaLl12NmUfAAAAAElFTkSuQmCC"  # Change to your own

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
                <div class="chat-message">
                    <img src="{user_img_url}">
                    <div class="chat-bubble user-bubble">{msg['content']}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="chat-message">
                    <img src="{assistant_img_url}">
                    <div class="chat-bubble assistant-bubble">{msg['content']}</div>
                </div>
            """, unsafe_allow_html=True)