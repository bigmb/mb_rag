import streamlit as st
from mb_rag.rag.embeddings import Embeddings
import os

# Initialize the Embeddings class
embeddings = Embeddings()

st.title("MB RAG Chatbot")

# Sidebar for configuration
st.sidebar.header("Configuration")
s3_link = st.sidebar.text_input("S3 Link for Embeddings")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

if s3_link and api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    
    # Load embeddings from S3
    embeddings.load_embeddings(s3_link)
    
    # Load retriever
    retriever = embeddings.load_retriever(s3_link)
    
    # Generate RAG chain
    rag_chain = embeddings.generate_rag_chain(retriever=retriever)
    
    # Chat interface
    st.header("Chat with RAG")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            response = embeddings.conversation_chain(prompt, rag_chain)
            st.markdown(response['answer'])
        
        st.session_state.messages.append({"role": "assistant", "content": response['answer']})

else:
    st.warning("Please provide both S3 link and OpenAI API key to start the chatbot.")
