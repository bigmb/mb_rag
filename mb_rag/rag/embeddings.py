"""
RAG (Retrieval-Augmented Generation) Embeddings Module

This module provides functionality for generating and managing embeddings for RAG models.
It supports multiple embedding models (OpenAI, Ollama, Google, Anthropic) and includes
features for text processing, embedding generation, vector store management, and 
conversation handling.

Example Usage:
    ```python
    # Initialize embedding generator
    em_gen = embedding_generator(
        model="openai",
        model_type="text-embedding-3-small",
        vector_store_type="chroma"
    )

    # Generate embeddings from text
    em_gen.generate_text_embeddings(
        text_data_path=['./data/text.txt'],
        chunk_size=500,
        chunk_overlap=5,
        folder_save_path='./embeddings'
    )

    # Load embeddings and create retriever
    em_loading = em_gen.load_embeddings('./embeddings')
    em_retriever = em_gen.load_retriever(
        './embeddings',
        search_params=[{"k": 2, "score_threshold": 0.1}]
    )

    # Query embeddings
    results = em_retriever.invoke("What is the text about?")

    # Generate RAG chain for conversation
    rag_chain = em_gen.generate_rag_chain(retriever=em_retriever)
    response = em_gen.conversation_chain("Tell me more", rag_chain)
    ```

Features:
    - Multiple model support (OpenAI, Ollama, Google, Anthropic)
    - Text processing and chunking
    - Embedding generation and storage
    - Vector store management
    - Retrieval operations
    - Conversation chains
    - Web crawling integration

Classes:
    - ModelProvider: Base class for model loading and validation
    - TextProcessor: Handles text processing operations
    - embedding_generator: Main class for RAG operations
"""

import os
import shutil
import importlib.util
from typing import List, Dict, Optional, Union, Any
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter)
from langchain_community.document_loaders import TextLoader, FireCrawlLoader
from langchain_chroma import Chroma
from ..utils.extra import load_env_file
# from langchain.chains import create_history_aware_retriever, create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank

load_env_file()

__all__ = ['embedding_generator', 'load_embedding_model']

class ModelProvider:
    """
    Base class for managing different model providers and their loading logic.

    This class provides static methods for loading different types of embedding models
    and checking package dependencies.

    Methods:
        check_package: Check if a Python package is installed
        get_rag_openai: Load OpenAI embedding model
        get_rag_ollama: Load Ollama embedding model
        get_rag_anthropic: Load Anthropic model
        get_rag_google: Load Google embedding model

    Example:
        ```python
        # Check if a package is installed
        has_openai = ModelProvider.check_package("langchain_openai")

        # Load an OpenAI model
        model = ModelProvider.get_rag_openai("text-embedding-3-small")
        ```
    """
    
    @staticmethod
    def check_package(package_name: str) -> bool:
        """
        Check if a Python package is installed.

        Args:
            package_name (str): Name of the package to check

        """
        return importlib.util.find_spec(package_name) is not None

    @staticmethod
    def get_rag_openai(model_type: str = 'text-embedding-3-small', **kwargs):
        """
        Load OpenAI embedding model.

        Args:
            model_type (str): Model identifier (default: 'text-embedding-3-small')
            **kwargs: Additional arguments for model initialization

        Returns:
            OpenAIEmbeddings: Initialized OpenAI embeddings model
        """
        if not ModelProvider.check_package("langchain_openai"):
            raise ImportError("OpenAI package not found. Please install: pip install langchain-openai")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model_type, **kwargs)

    @staticmethod
    def get_rag_ollama(model_type: str = 'llama3', **kwargs):
        """
        Load Ollama embedding model.

        Args:
            model_type (str): Model identifier (default: 'llama3')
            **kwargs: Additional arguments for model initialization

        Returns:
            OllamaEmbeddings: Initialized Ollama embeddings model
        """
        if not ModelProvider.check_package("langchain_ollama"):
            raise ImportError("Ollama package not found. Please install: pip install langchain-ollama")
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=model_type, **kwargs)

    @staticmethod
    def get_rag_anthropic(model_name: str = "claude-3-opus-20240229", **kwargs):
        """
        Load Anthropic model.

        Args:
            model_name (str): Model identifier (default: "claude-3-opus-20240229")
            **kwargs: Additional arguments for model initialization

        Returns:
            ChatAnthropic: Initialized Anthropic chat model

        """
        if not ModelProvider.check_package("langchain_anthropic"):
            raise ImportError("Anthropic package not found. Please install: pip install langchain-anthropic")
        from langchain_anthropic import ChatAnthropic
        kwargs["model_name"] = model_name
        return ChatAnthropic(**kwargs)

    @staticmethod
    def get_rag_google(model_name: str = "gemini-1.5-flash", **kwargs):
        """
        Load Google embedding model.

        Args:
            model_name (str): Model identifier (default: "gemini-1.5-flash")
            **kwargs: Additional arguments for model initialization

        Returns:
            GoogleGenerativeAIEmbeddings: Initialized Google embeddings model
        """
        if not ModelProvider.check_package("google.generativeai"):
            raise ImportError("Google Generative AI package not found. Please install: pip install langchain-google-genai")
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        kwargs["model"] = model_name
        return GoogleGenerativeAIEmbeddings(**kwargs)

    @staticmethod
    def get_rag_qwen(model_name: str = "Qwen/Qwen3-Embedding-0.6B", **kwargs):
        """
        Load Qwen embedding model. 
        Uses Transformers for embedding generation.

        Args:
            model_name (str): Model identifier (default: "Qwen/Qwen3-Embedding-0.6B")
            **kwargs: Additional arguments for model initialization

        Returns:
            QwenEmbeddings: Initialized Qwen embeddings model
        """
        from langchain.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=model_name, **kwargs)

def load_embedding_model(model_name: str = 'openai', model_type: str = "text-embedding-ada-002", **kwargs):
    """
    Load a RAG model based on provider and type.

    Args:
        model_name (str): Name of the model provider (default: 'openai')
        model_type (str): Type/identifier of the model (default: "text-embedding-ada-002")
        **kwargs: Additional arguments for model initialization

    Returns:
        Any: Initialized model instance

    Example:
        ```python
        model = load_embedding_model('openai', 'text-embedding-3-small')
        ```
    """
    try:
        if model_name == 'openai':
            return ModelProvider.get_rag_openai(model_type, **kwargs)
        elif model_name == 'ollama':
            return ModelProvider.get_rag_ollama(model_type, **kwargs)
        elif model_name == 'google':
            return ModelProvider.get_rag_google(model_type, **kwargs)
        elif model_name == 'anthropic':
            return ModelProvider.get_rag_anthropic(model_type, **kwargs)
        elif model_name == 'qwen':
            return ModelProvider.get_rag_qwen(model_type, **kwargs)
        else:
            raise ValueError(f"Invalid model name: {model_name}")
    except ImportError as e:
        print(f"Error loading model: {str(e)}")
        return None

class TextProcessor:
    """
    Handles text processing operations including file checking and tokenization.

    This class provides methods for loading text files, processing them into chunks,
    and preparing them for embedding generation.

    Args:
        logger: Optional logger instance for logging operations

    Example:
        ```python
        processor = TextProcessor()
        docs = processor.tokenize(
            ['./data.txt'],
            'recursive_character',
            chunk_size=1000,
            chunk_overlap=5
        )
        ```
    """
    
    def __init__(self, logger=None):
        self.logger = logger

    def check_file(self, file_path: str) -> bool:
        """Check if file exists."""
        return os.path.exists(file_path)

    def tokenize(self, text_data_path: List[str], text_splitter_type: str,
                chunk_size: int, chunk_overlap: int) -> List:
        """
        Process and tokenize text data from files.

        Args:
            text_data_path (List[str]): List of paths to text files
            text_splitter_type (str): Type of text splitter to use
            chunk_size (int): Size of text chunks
            chunk_overlap (int): Overlap between chunks

        Returns:
            List: List of processed document chunks

        """
        doc_data = []
        for path in text_data_path:
            if self.check_file(path):
                text_loader = TextLoader(path)
                get_text = text_loader.load()
                file_name = path.split('/')[-1]
                metadata = {'source': file_name}
                if metadata is not None:
                    for doc in get_text:
                        doc.metadata = metadata
                        doc_data.append(doc)
                if self.logger:
                    self.logger.info(f"Text data loaded from {file_name}")
            else:
                return f"File {path} not found"

        splitters = {
            'character': CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator=["\n", "\n\n", "\n\n\n", " "]
            ),
            'recursive_character': RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n", "\n\n", "\n\n\n", " "]
            ),
            'sentence_transformers_token': SentenceTransformersTokenTextSplitter(
                chunk_size=chunk_size
            ),
            'token': TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            ),
            'markdown_header': MarkdownHeaderTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            ),
        }

        if text_splitter_type not in splitters:
            raise ValueError(f"Invalid text splitter type: {text_splitter_type}")

        text_splitter = splitters[text_splitter_type]
        docs = text_splitter.split_documents(doc_data)

        if self.logger:
            self.logger.info(f"Text data splitted into {len(docs)} chunks")
        else:
            print(f"Text data splitted into {len(docs)} chunks")
        return docs


class embedding_generator:
    """
    Main class for generating embeddings and managing RAG operations.

    This class provides comprehensive functionality for generating embeddings,
    managing vector stores, handling retrievers, and managing conversations.

    Args:
        model (str): Model provider name (default: 'openai')
        model_type (str): Model type/identifier (default: 'text-embedding-3-small')
        vector_store_type (str): Type of vector store (default: 'chroma')
        collection_name (str): Name of the collection (default: 'test')
        logger: Optional logger instance
        model_kwargs (dict): Additional arguments for model initialization
        vector_store_kwargs (dict): Additional arguments for vector store initialization

    Example:
        ```python
        # Initialize generator
        gen = embedding_generator(
            model="openai",
            model_type="text-embedding-3-small",
            collection_name='test'
        )

        # Generate embeddings
        gen.generate_text_embeddings(
            text_data_path=['./data.txt'],
            folder_save_path='./embeddings'
        )

        # Load retriever
        retriever = gen.load_retriever('./embeddings', collection_name='test')

        # Query embeddings
        results = gen.query_embeddings("What is this about?")
        ```
    """

    def __init__(self, model: str = 'openai', model_type: str = 'text-embedding-3-small',
                vector_store_type: str = 'chroma', collection_name: str = 'test',
                logger=None, model_kwargs: dict = None, vector_store_kwargs: dict = None) -> None:
        """Initialize the embedding generator with specified configuration."""
        self.logger = logger
        self.model = load_embedding_model(model_name=model, model_type=model_type, **(model_kwargs or {}))
        if self.model is None:
            raise ValueError(f"Failed to initialize model {model}. Please ensure required packages are installed.")
        self.vector_store_type = vector_store_type
        self.vector_store = self.load_vectorstore(**(vector_store_kwargs or {}))
        self.collection_name = collection_name
        self.text_processor = TextProcessor(logger)
        self.compression_retriever = None

    def check_file(self, file_path: str) -> bool:
        """Check if file exists."""
        return self.text_processor.check_file(file_path)

    def tokenize(self, text_data_path: List[str], text_splitter_type: str,
                chunk_size: int, chunk_overlap: int) -> List:
        """Process and tokenize text data."""
        return self.text_processor.tokenize(text_data_path, text_splitter_type,
                                          chunk_size, chunk_overlap)

    def generate_text_embeddings(self, text_data_path: List[str] = None,
                               text_splitter_type: str = 'recursive_character',
                               chunk_size: int = 1000, chunk_overlap: int = 5,
                               folder_save_path: str = './text_embeddings',
                               replace_existing: bool = False) -> str:
        """
        Generate text embeddings from input files.

        Args:
            text_data_path (List[str]): List of paths to text files
            text_splitter_type (str): Type of text splitter
            chunk_size (int): Size of text chunks
            chunk_overlap (int): Overlap between chunks
            folder_save_path (str): Path to save embeddings
            replace_existing (bool): Whether to replace existing embeddings

        Returns:
            str: Status message

        Example:
            ```python
            gen.generate_text_embeddings(
                text_data_path=['./data.txt'],
                folder_save_path='./embeddings'
            )
            ```
        """
        if self.logger:
            self.logger.info("Performing basic checks")

        if self.check_file(folder_save_path) and not replace_existing:
            return "File already exists"
        elif self.check_file(folder_save_path) and replace_existing:
            shutil.rmtree(folder_save_path)

        if text_data_path is None:
            return "Please provide text data path"

        if not isinstance(text_data_path, list):
            raise ValueError("text_data_path should be a list")

        if self.logger:
            self.logger.info(f"Loading text data from {text_data_path}")

        docs = self.tokenize(text_data_path, text_splitter_type, chunk_size, chunk_overlap)

        if self.logger:
            self.logger.info(f"Generating embeddings for {len(docs)} documents")

        self.vector_store.from_documents(docs, self.model, collection_name=self.collection_name,
                                       persist_directory=folder_save_path)

        if self.logger:
            self.logger.info(f"Embeddings generated and saved at {folder_save_path}")

    def load_vectorstore(self, **kwargs):
        """Load vector store."""
        if self.vector_store_type == 'chroma':
            vector_store = Chroma()
            if self.logger:
                self.logger.info(f"Loaded vector store {self.vector_store_type}")
            return vector_store
        else:
            return "Vector store not found"

    def load_embeddings(self, embeddings_folder_path: str,collection_name: str = 'test'):
        """
        Load embeddings from folder.

        Args:
            embeddings_folder_path (str): Path to embeddings folder
            collection_name (str): Name of the collection. Default: 'test'

        Returns:
            Optional[Chroma]: Loaded vector store or None if not found
        """
        if self.check_file(embeddings_folder_path):
            if self.vector_store_type == 'chroma':
                return Chroma(persist_directory=embeddings_folder_path,
                            embedding_function=self.model,
                            collection_name=collection_name)
        else:
            if self.logger:
                self.logger.info("Embeddings file not found")
            return None

    def load_retriever(self, embeddings_folder_path: str,
                      search_type: List[str] = ["similarity_score_threshold"],
                      search_params: List[Dict] = [{"k": 3, "score_threshold": 0.9}],
                      collection_name: str = 'test'):
        """
        Load retriever with search configuration.

        Args:
            embeddings_folder_path (str): Path to embeddings folder
            search_type (List[str]): List of search types
            search_params (List[Dict]): List of search parameters
            collection_name (str): Name of the collection. Default: 'test'

        Returns:
            Union[Any, List[Any]]: Single retriever or list of retrievers

        Example:
            ```python
            retriever = gen.load_retriever(
                './embeddings',
                search_type=["similarity_score_threshold"],
                search_params=[{"k": 3, "score_threshold": 0.9}]
            )
            ```
        """
        db = self.load_embeddings(embeddings_folder_path, collection_name)
        if db is not None:
            if self.vector_store_type == 'chroma':
                if len(search_type) != len(search_params):
                    raise ValueError("Length of search_type and search_params should be equal")
                if len(search_type) == 1:
                    self.retriever = db.as_retriever(search_type=search_type[0],
                                                   search_kwargs=search_params[0])
                    if self.logger:
                        self.logger.info("Retriever loaded")
                    return self.retriever
                else:
                    retriever_list = []
                    for i in range(len(search_type)):
                        retriever_list.append(db.as_retriever(search_type=search_type[i],
                                                            search_kwargs=search_params[i]))
                    if self.logger:
                        self.logger.info("List of Retriever loaded")
                    return retriever_list
        else:
            return "Embeddings file not found"

    def add_data(self, embeddings_folder_path: str, data: List[str],
                text_splitter_type: str = 'recursive_character',
                chunk_size: int = 1000, chunk_overlap: int = 5, collection_name: str = 'test'):
        """
        Add data to existing embeddings.

        Args:
            embeddings_folder_path (str): Path to embeddings folder
            data (List[str]): List of text data to add
            text_splitter_type (str): Type of text splitter
            chunk_size (int): Size of text chunks
            chunk_overlap (int): Overlap between chunks
            collection_name (str): Name of the collection. Default: 'test'
        """
        if self.vector_store_type == 'chroma':
            db = self.load_embeddings(embeddings_folder_path, collection_name)
            if db is not None:
                docs = self.tokenize(data, text_splitter_type, chunk_size, chunk_overlap)
                db.add_documents(docs)
                if self.logger:
                    self.logger.info("Data added to the existing db/embeddings")

    def query_embeddings(self, query: str, retriever=None):
        """
        Query embeddings.

        Args:
            query (str): Query string
            retriever: Optional retriever instance

        Returns:
            Any: Query results
        """
        if retriever is None:
            retriever = self.retriever
        return retriever.invoke(query)

    def get_relevant_documents(self, query: str, retriever=None):
        """
        Get relevant documents for query.

        Args:
            query (str): Query string
            retriever: Optional retriever instance

        Returns:
            List: List of relevant documents
        """
        if retriever is None:
            retriever = self.retriever
        return retriever.get_relevant_documents(query)

    # def load_flashrank_compression_retriever(self, base_retriever=None, model_name: str = "flashrank/flashrank-base", top_n: int = 5):
    #     """
    #     Load a ContextualCompressionRetriever using FlashrankRerank.

    #     Args:
    #         base_retriever: Existing retriever (if None, uses self.retriever)
    #         model_name (str): Flashrank model identifier (default: "flashrank/flashrank-base")
    #         top_n (int): Number of top documents to return after reranking

    #     Returns:
    #         ContextualCompressionRetriever: A compression-based retriever using Flashrank
    #     """
    #     if base_retriever is None:
    #         base_retriever = self.retriever
    #     if base_retriever is None:
    #         raise ValueError("Base retriever is required.")

    #     compressor = FlashrankRerank(model=model_name, top_n=top_n)
    #     self.compression_retriever = ContextualCompressionRetriever(
    #         base_compressor=compressor,
    #         base_retriever=base_retriever
    #     )

    #     if self.logger:
    #         self.logger.info("Loaded Flashrank compression retriever.")
    #     return self.compression_retriever

    def compression_invoke(self, query: str):
        """
        Invoke compression retriever. Only one compression retriever (Reranker) added right now. 

        Args:
            query (str): Query string

        Returns:
            Any: Query results
        """

        if self.compression_retriever is None:
            self.compression_retriever = self.load_flashrank_compression_retriever(base_retriever=self.retriever)
            print("Compression retriever loaded.")
        return self.compression_retriever.invoke(query)

    # def generate_rag_chain(self, context_prompt: str = None, retriever=None, llm=None):
    #     """
    #     Generate RAG chain for conversation.

    #     Args:
    #         context_prompt (str): Optional context prompt
    #         retriever: Optional retriever instance
    #         llm: Optional language model instance

    #     Returns:
    #         Any: Generated RAG chain

    #     Example:
    #         ```python
    #         rag_chain = gen.generate_rag_chain(retriever=retriever)
    #         ```
    #     """
    #     if context_prompt is None:
    #         context_prompt = ("You are an assistant for question-answering tasks. "
    #                         "Use the following pieces of retrieved context to answer the question. "
    #                         "If you don't know the answer, just say that you don't know. "
    #                         "Use three sentences maximum and keep the answer concise.\n\n{context}")

    #     contextualize_q_system_prompt = ("Given a chat history and the latest user question "
    #                                    "which might reference context in the chat history, "
    #                                    "formulate a standalone question which can be understood, "
    #                                    "just reformulate it if needed and otherwise return it as is.")

    #     contextualize_q_prompt = ChatPromptTemplate.from_messages([
    #         ("system", contextualize_q_system_prompt),
    #         MessagesPlaceholder("chat_history"),
    #         ("human", "{input}"),
    #     ])

    #     if retriever is None:
    #         retriever = self.retriever
    #     if llm is None:
    #         if not ModelProvider.check_package("langchain_openai"):
    #             raise ImportError("OpenAI package not found. Please install: pip install langchain-openai")
    #         from langchain_openai import ChatOpenAI
    #         llm = ChatOpenAI(model="gpt-4o", temperature=0.8)

    #     history_aware_retriever = create_history_aware_retriever(llm, retriever,
    #                                                            contextualize_q_prompt)
    #     qa_prompt = ChatPromptTemplate.from_messages([
    #         ("system", context_prompt),
    #         MessagesPlaceholder("chat_history"),
    #         ("human", "{input}"),
    #     ])
    #     question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    #     rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    #     return rag_chain

    def conversation_chain(self, query: str, rag_chain, file: str = None):
        """
        Create conversation chain.

        Args:
            query (str): User query
            rag_chain: RAG chain instance
            file (str): Optional file to save conversation

        Returns:
            List: Conversation history

        Example:
            ```python
            history = gen.conversation_chain(
                "Tell me about...",
                rag_chain,
                file='conversation.txt'
            )
            ```
        """
        if file is not None:
            try:
                chat_history = self.load_conversation(file, list_type=True)
                if len(chat_history) == 0:
                    chat_history = []
            except:
                chat_history = []
        else:
            chat_history = []

        query = "You : " + query
        res = rag_chain.invoke({"input": query, "chat_history": chat_history})
        print(f"Response: {res['answer']}")
        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=res['answer']))
        if file is not None:
            self.save_conversation(chat_history, file)
        return chat_history

    def load_conversation(self, file: str, list_type: bool = False):
        """
        Load conversation history.

        Args:
            file (str): Path to conversation file
            list_type (bool): Whether to return as list

        Returns:
            Union[str, List]: Conversation history
        """
        if list_type:
            chat_history = []
            with open(file, 'r') as f:
                for line in f:
                    chat_history.append(line.strip())
        else:
            with open(file, "r") as f:
                chat_history = f.read()
        return chat_history

    def save_conversation(self, chat: Union[str, List], file: str):
        """
        Save conversation history.

        Args:
            chat (Union[str, List]): Conversation to save
            file (str): Path to save file
        """
        if isinstance(chat, str):
            with open(file, "a") as f:
                f.write(chat)
        elif isinstance(chat, list):
            with open(file, "a") as f:
                for i in chat[-2:]:
                    f.write("%s\n" % i)
        print(f"Saved file : {file}")

    def firecrawl_web(self, website: str, api_key: str = None, mode: str = "scrape",
                      file_to_save: str = './firecrawl_embeddings', **kwargs):
        """
        Get data from website using FireCrawl.

        Args:
            website (str): Website URL to crawl
            api_key (str): Optional FireCrawl API key
            mode (str): Crawl mode (default: "scrape")
            file_to_save (str): Path to save embeddings
            **kwargs: Additional arguments for FireCrawl

        Returns:
            Chroma: Vector store with crawled data

        Example:
            ```python
            db = gen.firecrawl_web(
                "https://example.com",
                mode="scrape",
                file_to_save='./crawl_embeddings'
            )
            ```
        """
        if not ModelProvider.check_package("firecrawl"):
            raise ImportError("Firecrawl package not found. Please install: pip install firecrawl")

        if api_key is None:
            api_key = os.getenv("FIRECRAWL_API_KEY")

        loader = FireCrawlLoader(api_key=api_key, url=website, mode=mode)
        docs = loader.load()

        for doc in docs:
            for key, value in doc.metadata.items():
                if isinstance(value, list):
                    doc.metadata[key] = ", ".join(map(str, value))

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        split_docs = text_splitter.split_documents(docs)

        print("\n--- Document Chunks Information ---")
        print(f"Number of document chunks: {len(split_docs)}")
        print(f"Sample chunk:\n{split_docs[0].page_content}\n")

        embeddings = self.model
        db = Chroma.from_documents(split_docs, embeddings,
                                 persist_directory=file_to_save)
        print(f"Retriever saved at {file_to_save}")
        return db
