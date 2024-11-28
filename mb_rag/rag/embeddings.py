## Function to generate embeddings for the RAG model

import os
import shutil
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TokenTextSplitter)
from langchain_community.document_loaders import TextLoader,FireCrawlLoader
from langchain_community.vectorstores import Chroma
from ..utils.extra  import load_env_file
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import time

load_env_file()

# test_file = '/home/malav/Desktop/mb_packages/mb_rag/examples/test.txt'
# test_db = '/home/malav/Desktop/mb_packages/mb_rag/examples/db/test.db'

__all__ = ['embedding_generator', 'load_rag_model']

def get_rag_openai(model_type: str = 'text-embedding-3-small',**kwargs):
    """
    Load model from openai for RAG
    Args:
        model_type (str): Name of the model
        **kwargs: Additional arguments (temperature, max_tokens, timeout, max_retries, api_key etc.)
    Returns:
        ChatOpenAI: Chatbot model
    """
    return OpenAIEmbeddings(model = model_type,**kwargs)

def get_rag_ollama(model_type: str = 'llama3',**kwargs):
    """
    Load model from ollama for RAG
    Args:
        model_type (str): Name of the model
        **kwargs: Additional arguments (temperature, max_tokens, timeout, max_retries, api_key etc.)
    Returns:
        OllamaEmbeddings: Embeddings model
    """
    return OllamaEmbeddings(model = model_type,**kwargs)

def get_rag_anthropic(model_name: str = "claude-3-opus-20240229",**kwargs):
    """
    Load the chatbot model from Anthropic
    Args:
        model_name (str): Name of the model
        **kwargs: Additional arguments (temperature, max_tokens, timeout, max_retries, api_key etc.)
    Returns:
        ChatAnthropic: Chatbot model
    """
    kwargs["model_name"] = model_name
    return ChatAnthropic(**kwargs)

def get_rag_google(model_name: str = "gemini-1.5-flash",**kwargs):
    """
    Load the chatbot model from Google Generative AI
    Args:
        model_name (str): Name of the model 
        **kwargs: Additional arguments (temperature, max_tokens, timeout, max_retries, api_key etc.)
    Returns:
        ChatGoogleGenerativeAI: Chatbot model
    """
    kwargs["model"] = model_name
    return GoogleGenerativeAIEmbeddings(**kwargs)


def load_rag_model(model_name: str ='openai', model_type: str = "text-embedding-ada-002", **kwargs):
    """
    Load a RAG model from a given model name and type
    Args:
        model_name (str): Name of the model. Default is openai.
        model_type (str): Type of the model. Default is text-embedding-ada-002.
        **kwargs: Additional arguments (temperature, max_tokens, timeout, max_retries, api_key etc.)
    Returns:
        RAGModel: RAG model
    """
    if model_name == 'openai':
        return get_rag_openai(model_type, **(kwargs or {}))
    elif model_name == 'ollama':
        return get_rag_ollama(model_type, **(kwargs or {}))
    elif model_name == 'google':
        return get_rag_google(model_type, **(kwargs or {}))
    elif model_name == 'anthropic':
        return get_rag_anthropic(model_type, **(kwargs or {}))
    else:
        raise ValueError(f"Invalid model name: {model_name}")


class embedding_generator:
    """
    Class to generate embeddings for the RAG model abnd chat with data
    Args:
        model: type of model. Default is openai. Options are openai, anthropic, google, ollama
        model_type: type of model. Default is text-embedding-3-small. Options are text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002 for openai.
        vector_store_type: type of vector store. Default is chroma
        logger: logger
        model_kwargs: additional arguments for the model
        vector_store_kwargs: additional arguments for the vector store
        collection_name: name of the collection (default : test)
    """

    def __init__(self,model: str = 'openai',model_type: str = 'text-embedding-3-small',vector_store_type:str = 'chroma' ,collection_name: str = 'test',logger= None,model_kwargs: dict = None, vector_store_kwargs: dict = None) -> None:
        self.logger = logger
        self.model = load_rag_model(model_name=model, model_type=model_type, **(model_kwargs or {}))
        self.vector_store_type = vector_store_type
        self.vector_store = self.load_vectorstore(**(vector_store_kwargs or {}))
        self.collection_name = collection_name

    def check_file(self, file_path):
        """
        Check if the file exists
        """
        if os.path.exists(file_path):
            return True
        else:
            return False

    def tokenize(self,text_data_path :list,text_splitter_type: str,chunk_size: int,chunk_overlap: int):
        """
        Function to tokenize the text
        Args:
            text: text to tokenize
        Returns:
            tokens
        """
        doc_data = [] 
        for i in text_data_path:
            if self.check_file(i):
                text_loader = TextLoader(i)
                get_text = text_loader.load()
                # print(get_text) ## testing - Need to remove
                file_name = i.split('/')[-1]
                metadata = {'source': file_name}
                if metadata is not None:
                    for j in get_text:
                        j.metadata = metadata
                        doc_data.append(j)
                if self.logger is not None:
                    self.logger.info(f"Text data loaded from {file_name}")
            else:
                return f"File {i} not found"

        if self.logger is not None:
            self.logger.info(f"Splitting text data into chunks of size {chunk_size} with overlap {chunk_overlap}")
        if text_splitter_type == 'character':
            text_splitter = CharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap, separator=["\n","\n\n","\n\n\n"," "])
        if text_splitter_type == 'recursive_character':
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap,separators=["\n","\n\n","\n\n\n"," "])
        if text_splitter_type == 'sentence_transformers_token':
            text_splitter = SentenceTransformersTokenTextSplitter(chunk_size=chunk_size)
        if text_splitter_type == 'token':
            text_splitter = TokenTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
        docs = text_splitter.split_documents(doc_data) 
        if self.logger is not None:
            self.logger.info(f"Text data splitted into {len(docs)} chunks")
        else:
            print(f"Text data splitted into {len(docs)} chunks")
        return docs       

    def generate_text_embeddings(self,text_data_path: list = None,text_splitter_type: str = 'recursive_character',
                                 chunk_size: int = 1000,chunk_overlap: int = 5,folder_save_path: str = './text_embeddings',
                                 replace_existing: bool = False):
        """
        Function to generate text embeddings
        Args:
            text_data_path: list of text files
            # metadata: list of metadata for each text file. Dictionary format
            text_splitter_type: type of text splitter. Default is recursive_character
            chunk_size: size of the chunk
            chunk_overlap: overlap between chunks
            folder_save_path: path to save the embeddings
            replace_existing: if True, replace the existing embeddings
        Returns:
            None   
        """

        if self.logger is not None:
            self.logger.info("Perforing basic checks")

        if self.check_file(folder_save_path) and replace_existing==False:
            return "File already exists"
        elif self.check_file(folder_save_path) and replace_existing:
            shutil.rmtree(folder_save_path) 

        if text_data_path is None:
            return "Please provide text data path"

        assert isinstance(text_data_path, list), "text_data_path should be a list"
        # if metadata is not None:
        #     assert isinstance(metadata, list), "metadata should be a list"
        #     assert len(text_data_path) == len(metadata), "Number of text files and metadata should be equal"

        if self.logger is not None:
            self.logger.info(f"Loading text data from {text_data_path}")

        docs = self.tokenize(text_data_path,text_splitter_type,chunk_size,chunk_overlap)

        if self.logger is not None:
            self.logger.info(f"Generating embeddings for {len(docs)} documents")    

        self.vector_store.from_documents(docs, self.model,collection_name=self.collection_name,persist_directory=folder_save_path)

        if self.logger is not None:
            self.logger.info(f"Embeddings generated and saved at {folder_save_path}")

    def load_vectorstore(self):
        """
        Function to load vector store
        Args:
            vector_store_type: type of vector store
        Returns:
            vector store
        """
        if self.vector_store_type == 'chroma':
            vector_store = Chroma()
            if self.logger is not None:
                self.logger.info(f"Loaded vector store {self.vector_store_type}")
            return vector_store
        else:
            return "Vector store not found"

    def load_embeddings(self,embeddings_folder_path: str):
        """
        Function to load embeddings from the folder
        Args:
            embeddings_path: path to the embeddings
        Returns:
            embeddings
        """
        if self.check_file(embeddings_folder_path):
            if self.vector_store_type == 'chroma':
                # embeddings_path = os.path.join(embeddings_folder_path)
                return Chroma(persist_directory = embeddings_folder_path,embedding_function=self.model)
        else:
            if self.logger:
                self.logger.info("Embeddings file not found") 
            return None  
        
    def load_retriever(self,embeddings_folder_path: str,search_type: list = ["similarity_score_threshold"],search_params: list = [{"k": 3, "score_threshold": 0.9}]):
        """
        Function to load retriever
        Args:
            embeddings_path: path to the embeddings
            search_type: list of str: type of search. Default : ["similarity_score_threshold"]
            search_params: list of dict: parameters for the search. Default : [{"k": 3, "score_threshold": 0.9}]
        Returns:
            Retriever. If multiple search types are provided, a list of retrievers is returned
        """
        db = self.load_embeddings(embeddings_folder_path)
        if db is not None:
            if self.vector_store_type == 'chroma':
                assert len(search_type) == len(search_params), "Length of search_type and search_params should be equal"
                if len(search_type) == 1:
                    self.retriever = db.as_retriever(search_type = search_type[0],search_kwargs=search_params[0])
                    if self.logger:
                        self.logger.info("Retriever loaded")
                    return self.retriever
                else:
                    retriever_list = []
                    for i in range(len(search_type)):
                        retriever_list.append(db.as_retriever(search_type = search_type[i],search_kwargs=search_params[i]))
                    if self.logger:
                            self.logger.info("List of Retriever loaded")
                    return retriever_list
        else:
            return "Embeddings file not found"
        
    def add_data(self,embeddings_folder_path: str, data: list,text_splitter_type: str = 'recursive_character',
                 chunk_size: int = 1000,chunk_overlap: int = 5):
        """
        Function to add data to the existing db/embeddings
        Args:
            embeddings_path: path to the embeddings
            data: list of data to add
            text_splitter_type: type of text splitter. Default is recursive_character
            chunk_size: size of the chunk
            chunk_overlap: overlap between chunks
        Returns:
            None
        """
        if self.vector_store_type == 'chroma':
            db = self.load_embeddings(embeddings_folder_path)
            if db is not None:
                docs = self.tokenize(data,text_splitter_type,chunk_size,chunk_overlap)
                db.add_documents(docs)
                if self.logger:
                    self.logger.info("Data added to the existing db/embeddings")
    
    def query_embeddings(self,query: str,retriever = None):
        """
        Function to query embeddings
        Args:
            search_type: type of search
            query: query to search
        Returns:
            results
        """
        # if self.vector_store_type == 'chroma':
        if retriever is None:
            retriever = self.retriever
        return retriever.invoke(query)
        # else:
        #     return "Vector store not found"

    def get_relevant_documents(self,query: str,retriever = None):
        """
        Function to get relevant documents
        Args:
            query: query to search
        Returns:
            results
        """
        return self.retriever.get_relevant_documents(query)
    
    def generate_rag_chain(self,context_prompt: str = None,retriever = None,llm= None):
        """
        Function to start a conversation chain with a rag data. Call this to load a rag_chain module.
        Args:
            context_prompt: prompt to context
            retriever: retriever. Default is None.
            llm: language model. Default is openai. Need chat model llm. "ChatOpenAI", "ChatAnthropic" etc. like chatbot
        Returns:
            rag_chain_model.
        """
        if context_prompt is None:
            context_prompt = ("You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. "
                              "If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. "
                              "\n\n {context}")
        contextualize_q_system_prompt = ("Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood "
                                        "without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is.")
        contextualize_q_prompt = ChatPromptTemplate.from_messages([("system", contextualize_q_system_prompt),MessagesPlaceholder("chat_history"),("human", "{input}"),])

        if retriever is None:
            retriever = self.retriever
        if llm is None:
            llm = ChatOpenAI(model="gpt-4o")

        history_aware_retriever = create_history_aware_retriever(llm,retriever, contextualize_q_prompt)
        qa_prompt = ChatPromptTemplate.from_messages([("system", context_prompt),MessagesPlaceholder("chat_history"),("human", "{input}"),])
        question_answer_chain =  create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        return rag_chain

    def conversation_chain(self,query: str,rag_chain,file:str =None):
        """
        Function to create a conversation chain
        Args:
            query: query to search
            rag_chain : rag_chain model
            file: load a file and update it with the conversation. If None it will not be saved.
        Returns:
            results
        """
        if file is not None:
            try:
                chat_history = self.load_conversation(file,list_type=True)
                if len(chat_history) == 0:
                    chat_history = []
            except:
                chat_history = []
        else:
            chat_history = []
        query = "You : " + query 
        res = rag_chain.invoke({"input": query,"chat_history": chat_history})
        print(f"Response: {res['answer']}")
        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=res['answer']))
        if file is not None:
            self.save_conversation(chat_history,file)
        return chat_history

    def load_conversation(self,file: str,list_type: bool = False):
        """
        Function to load the conversation
        Args:
            file: file to load
            list_type: if True, return the chat_history as a list. Default is False.
        Returns:
            chat_history
        """
        if list_type:
            chat_history = []
            with open(file,'r') as f:
                for line in f:
                    # inner_list = [elt.strip() for elt in line.split(',')]
                    chat_history.append(line.strip())
        else:
            with open(file, "r") as f:
                chat_history = f.read()
        return chat_history

    def save_conversation(self,chat: str,file: str):
        """
        Function to save the conversation
        Args:
            chat: chat results
            file: file to save
        Returns:
            None
        """
        if isinstance(chat,str):
            with open(file, "a") as f:
                f.write(chat)
        elif isinstance(chat,list):
            with open(file, "a") as f:
                for i in chat[-2:]:
                    f.write("%s\n" % i)
        print(f"Saved file : {file}")

    def firecrawl_web(self, website, api_key: str = None, mode="scrape", file_to_save: str = './firecrawl_embeddings',**kwargs):
        """
        Function to get data from website. Use this to get data from a website and save it as embeddings/retriever. To ask questions from the website,
          use the load_retriever and query_embeddings function.
        Args:
            website : str - link to website.
            api_key : api key of firecrawl, if None environment variable "FIRECRAWL_API_KEY" will be used.
            mode(str) : 'scrape' default to just use the same page. Not the whole website.
            file_to_save: path to save the embeddings
            **kwargs: additional arguments
        Returns:
            retriever
        """
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
        db = Chroma.from_documents(
            split_docs, embeddings, persist_directory=file_to_save)        
        print(f"Retriever saved at {file_to_save}")
        return db


