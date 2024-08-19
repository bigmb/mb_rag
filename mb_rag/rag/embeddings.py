## Function to generate embeddings for the RAG model

import os
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TokenTextSplitter,
)
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from mb_rag.utils.extra  import load_env_file

load_env_file()

test_file = '/home/malav/Desktop/mb_packages/mb_rag/examples/test.txt'
test_db = '/home/malav/Desktop/mb_packages/mb_rag/examples/db/test.db'

__all__ = ['embedding_generator']

class embedding_generator:
    """
    Class to generate embeddings for the RAG model
    """

    def __init__(self,model: str = 'openai',model_type: str = 'text-embedding-3-small',vector_store_type:str = 'chroma' ,logger= None,**kwargs) -> None:
        self.logger = logger
        self.model = self.load_model(model,model_type,**kwargs)
        self.vector_store = self.load_vectorstore(vector_store_type,**kwargs)

    def check_file(self, file_path):
        """
        Check if the file exists
        """
        if os.path.exists(file_path):
            return True
        else:
            return False

    def generate_text_embeddings(self,text_data_path: list = None,metadata: list = None,text_splitter_type: str = 'character',
                                 chunk_size: int = 1000,chunk_overlap: int = 5,file_save_path: str = './text_embeddings.db'):
        """
        Function to generate text embeddings
        Args:
            text_data_path: list of text files
            metadata: list of metadata for each text file. Dictionary format
            text_splitter_type: type of text splitter. Default is character
            chunk_size: size of the chunk
            chunk_overlap: overlap between chunks
            file_save_path: path to save the embeddings
        Returns:
            None   
        """

        if self.logger is not None:
            self.logger.info("Perforing basic checks")

        if self.check_file(file_save_path):
            return "File already exists"

        if text_data_path is None:
            return "Please provide text data path"

        assert isinstance(text_data_path, list), "text_data_path should be a list"
        if metadata is not None:
            assert isinstance(metadata, list), "metadata should be a list"
            assert len(text_data_path) == len(metadata), "Number of text files and metadata should be equal"

        if self.logger is not None:
            self.logger.info(f"Loading text data from {text_data_path}")

        doc_data = [] 
        for i in text_data_path:
            if self.check_file(i):
                text_loader = TextLoader(i)
                get_text = text_loader.load()
                if metadata is not None:
                    for j in get_text:
                        j.metadata = metadata[i]
                        doc_data.append(j)
            if self.logger is not None:
                self.logger.info(f"Text data loaded from {i}")
            else:
                return f"File {i} not found"

        if self.logger is not None:
            self.logger.info(f"Splitting text data into chunks of size {chunk_size} with overlap {chunk_overlap}")
        if text_splitter_type == 'character':
            text_splitter = CharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
        if text_splitter_type == 'recursive_character':
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
        if text_splitter_type == 'sentence_transformers_token':
            text_splitter = SentenceTransformersTokenTextSplitter(chunk_size=chunk_size)
        if text_splitter_type == 'token':
            text_splitter = TokenTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
        docs = text_splitter.split_documents(doc_data)

        print(docs)
        if self.logger is not None:
            self.logger.info(f"Generating embeddings for {len(docs)} documents")    
        self.vector_store.from_documents(docs, self.model,persist_directory=file_save_path)
        if self.logger is not None:
            self.logger.info(f"Embeddings generated and saved at {file_save_path}")

    def load_model(self,model: str,model_type: str):
        if model == 'openai':
            model_emb = OpenAIEmbeddings(model = model_type)
            if self.logger is not None:
                self.logger.info(f"Loaded model {model_type}")
            return model_emb
        else:
            return "Model not found"

    def load_vectorstore(self,vector_store_type: str):
        if vector_store_type == 'chroma':
            vector_store = Chroma()
            if self.logger is not None:
                self.logger.info(f"Loaded vector store {vector_store_type}")
            return vector_store
        else:
            return "Vector store not found"

    def load_embeddings(self,embeddings_path: str):
        """
        Function to load embeddings
        Args:
            embeddings_path: path to the embeddings
        Returns:
            embeddings
        """
        if self.check_file(embeddings_path):
            if self.vector_store_type == 'chroma':
                return Chroma(persist_directory = embeddings_path,embedding_function=self.model)
        else:
            return "Embeddings file not found"
        
    def load_retriever(self,embeddings_path: str,search_type: str = "similarity_score_threshold" ,search_params: dict = {"k": 3, "score_threshold": 0.9}):
        """
        Function to load retriever
        Args:
            embeddings_path: path to the embeddings
        Returns:
            retriever
        """
        if self.check_file(embeddings_path):
            if self.vector_store_type == 'chroma':
                db = Chroma(persist_directory = embeddings_path,embedding_function=self.model)
                self.retriever = db.as_retriever(search_type = search_type,search_kwargs=search_params)
                if self.logger:
                    self.logger.info("Retriever loaded")
                # return self.retriever
        else:
            return "Embeddings file not found"
        
    @staticmethod
    def query_embeddings(self,query: str):
        """
        Function to query embeddings
        Args:
            search_type: type of search
            query: query to search
        Returns:
            results
        """
        if self.vector_store_type == 'chroma':
            return self.retriever.invoke(query)