import streamlit as st
from mb_rag.rag.embeddings import embedding_generator
import os

# Initialize the Embeddings class
embeddings = embedding_generator(model='openai', model_type='text-embedding-3-small', vector_store_type='chroma')

st.title("MB RAG Chatbot")

# Sidebar for configuration
st.sidebar.header("Configuration")
embeddings_folder_path = st.sidebar.text_input("Path to Embeddings Folder")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

if embeddings_folder_path and api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    
    # Load retriever
    retriever = embeddings.load_retriever(embeddings_folder_path)
    
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
            embeddings.conversation_chain(prompt, rag_chain)
            response = embeddings.query_embeddings(prompt)
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.warning("Please provide both the path to the embeddings folder and OpenAI API key to start the chatbot.")
