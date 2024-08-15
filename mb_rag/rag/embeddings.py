## Function to generate embeddings for the RAG model

import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

test_file = '/home/malav/Desktop/mb_packages/mb_rag/examples/test.txt'
test_db = '/home/malav/Desktop/mb_packages/mb_rag/examples/db/test.db'

__all__ = ['embedding_generator']

class embedding_generator:
    """
    Class to generate embeddings for the RAG model
    """

    def __init__(self) -> None:
        pass

    def generate_embeddings():
        pass

    def load_embeddings():
        pass


