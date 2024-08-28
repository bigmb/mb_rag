## Function to generate embeddings for the RAG model

import os
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
from ..chatbot.basic import get_chatbot_ollama,get_chatbot_google_generative_ai,get_chatbot_anthropic,get_chatbot_openai
load_env_file()

test_file = '/home/malav/Desktop/mb_packages/mb_rag/examples/test.txt'
test_db = '/home/malav/Desktop/mb_packages/mb_rag/examples/db/test.db'

__all__ = ['embedding_generator']

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
    """

    def __init__(self,model: str = 'openai',model_type: str = 'text-embedding-3-small',vector_store_type:str = 'chroma' ,logger= None,model_kwargs: dict = None, vector_store_kwargs: dict = None) -> None:
        self.logger = logger
        if model == 'openai':
            self.model = get_chatbot_openai(model_type, **(model_kwargs or {}))
        elif model == 'anthropic':
            self.model = get_chatbot_anthropic(model_type, **(model_kwargs or {}))
        elif model == 'google':
            self.model = get_chatbot_google_generative_ai(model_type, **(model_kwargs or {}))
        elif model == 'ollama':
            self.model = get_chatbot_ollama(model_type, **(model_kwargs or {}))
        else:
            raise ValueError(f"Model {model} not found")
        self.vector_store_type = vector_store_type
        self.vector_store = self.load_vectorstore(**(vector_store_kwargs or {}))

    def check_file(self, file_path):
        """
        Check if the file exists
        """
        if os.path.exists(file_path):
            return True
        else:
            return False

    def generate_text_embeddings(self,text_data_path: list = None,text_splitter_type: str = 'character',
                                 chunk_size: int = 1000,chunk_overlap: int = 5,folder_save_path: str = './text_embeddings',
                                 replace_existing: bool = False):
        """
        Function to generate text embeddings
        Args:
            text_data_path: list of text files
            # metadata: list of metadata for each text file. Dictionary format
            text_splitter_type: type of text splitter. Default is character
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
            os.remove(folder_save_path)

        if text_data_path is None:
            return "Please provide text data path"

        assert isinstance(text_data_path, list), "text_data_path should be a list"
        # if metadata is not None:
        #     assert isinstance(metadata, list), "metadata should be a list"
        #     assert len(text_data_path) == len(metadata), "Number of text files and metadata should be equal"

        if self.logger is not None:
            self.logger.info(f"Loading text data from {text_data_path}")

        doc_data = [] 
        for i in text_data_path:
            if self.check_file(i):
                text_loader = TextLoader(i)
                get_text = text_loader.load()
                print(get_text) ## testing - Need to remove
                metadata = {'source': i}
                if metadata is not None:
                    for j in get_text:
                        j.metadata = metadata
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

        print(docs) ## testing - Need to remove
        if self.logger is not None:
            self.logger.info(f"Generating embeddings for {len(docs)} documents")    
        self.vector_store.from_documents(docs, self.model,persist_directory=folder_save_path)
        if self.logger is not None:
            self.logger.info(f"Embeddings generated and saved at {folder_save_path}")

    def load_model(self,model: str,model_type: str):
        """
        Function to load model
        Args:
            model: type of model
            model_type: type of model
        Returns:
            model
        """
        if model == 'openai':
            model_emb = OpenAIEmbeddings(model = model_type)
            if self.logger is not None:
                self.logger.info(f"Loaded model {model_type}")
            return model_emb
        else:
            return "Model not found"

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
        else:
            return "Vector store not found"
        
    def generate_rag_chain(self,context_prompt: str = None,retriever = None,llm= None):
        """
        Function to start a conversation chain with a rag data. Call this to load a rag_chain module.
        Args:
            context_prompt: prompt to context
            retriever: retriever
            llm: language model
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
            chat_history = self.load_conversation(file)
        else:
            chat_history = []
        query = "You : " + query 
        res = rag_chain.invoke({"question": query,"chat_history": chat_history})
        print(f"Response: {res['answer']}")
        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=res['answer']))
        if file is not None:
            self.save_conversation(chat_history,file)

    def load_conversation(self,file: str):
        """
        Function to load the conversation
        Args:
            file: file to load
        Returns:
            chat_history
        """
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
        with open(file, "a") as f:
            f.write(chat)
        print(f"Saved file : {file}")

    def firecrawl_web(self, website, api_key: str = None, mode="scrape", file_to_save: str = './firecrawl_embeddings',**kwargs):
        """
        Function to get data from website. Use this to get data from a website and save it as embeddings/retriever. To ask questions from the website,
          use the load_retriever and query_embeddings function.
        Args:
            website : str - link to wevsite.
            api_key : api key of firecrawl, if None environment variable "FIRECRAWL_API_KEY" will be used.
            mode(str) : 'scrape' default to just use the same page. Not the whole website.
            file_to_save: path to save the embeddings
            **kwargs: additional arguments
        Returns:
            None
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



