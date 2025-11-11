## file for loading all models for chat/rag

import os
from langchain_core.messages import HumanMessage
from mb_rag.utils.extra import check_package
import base64
from typing import Any
from .utils.all_data_extract import DocumentExtractor
import pandas as pd
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

__all__ = [
    'ModelFactory',
]

class ModelFactory:
    """Factory class for creating different types of chatbot models"""
    
    def __init__(self, model_type: str = 'openai', model_name: str = "gpt-4o", **kwargs) -> Any:
        """
        Factory method to create any type of model
        Args:
            model_type (str): Type of model to create. Default is OpenAI. Options are openai, anthropic, google, ollama , groq
            model_name (str): Name of the model
            **kwargs: Additional arguments
        Returns:
            Any: Chatbot model
        """
        creators = {
            'openai': self.create_openai,
            'anthropic': self.create_anthropic,
            'google': self.create_google,
            'ollama': self.create_ollama,
            'groq': self.create_groq,
            'deepseek': self.create_deepseek,
            'qwen' : self.create_qwen,
            'hugging_face': self.create_hugging_face
        }
        
        self.model_type = model_type
        self.model_name = model_name
        model_data = creators[model_type] if model_type in creators else None
        if not model_data:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        try:
            self.model = model_data(model_name, **kwargs)
        except Exception as e:
            raise ValueError(f"Error creating {model_type} model: {str(e)}")
        
    @classmethod
    def create_openai(cls, model_name: str = "gpt-4o", **kwargs) -> Any:
        """
        Create OpenAI chatbot model
        Args:
            model_name (str): Name of the model
            **kwargs: Additional arguments
        Returns:
            ChatOpenAI: Chatbot model
        """
        if not check_package("openai"):
            raise ImportError("OpenAI package not found. Please install it using: pip install openai langchain-openai")
        
        from langchain_openai import ChatOpenAI
        kwargs["model_name"] = model_name
        return ChatOpenAI(**kwargs)

    @classmethod
    def create_anthropic(cls, model_name: str = "claude-3-opus-20240229", **kwargs) -> Any:
        """
        Create Anthropic chatbot model
        Args:
            model_name (str): Name of the model
            **kwargs: Additional arguments
        Returns:
            ChatAnthropic: Chatbot model
        """
        if not check_package("anthropic"):
            raise ImportError("Anthropic package not found. Please install it using: pip install anthropic langchain-anthropic")
        
        from langchain_anthropic import ChatAnthropic
        kwargs["model_name"] = model_name
        return ChatAnthropic(**kwargs)

    @classmethod
    def create_google(cls, model_name: str = "gemini-2.0-flash", **kwargs) -> Any:
        """
        Create Google chatbot model
        Args:
            model_name (str): Name of the model
            **kwargs: Additional arguments
        Returns:
            ChatGoogleGenerativeAI: Chatbot model
        """
        if not check_package("langchain_google_genai"):
            raise ImportError("langchain_google_genai package not found. Please install it using: pip install google-generativeai")
        
        from langchain_google_genai import ChatGoogleGenerativeAI
        kwargs["model"] = model_name
        return ChatGoogleGenerativeAI(**kwargs)

    @classmethod
    def create_ollama(cls, model_name: str = "llama3", **kwargs) -> Any:
        """
        Create Ollama chatbot model
        Args:
            model_name (str): Name of the model
            **kwargs: Additional arguments
        Returns:
            Ollama: Chatbot model
        """
        if not check_package("langchain_ollama"):
            raise ImportError("Langchain Community package not found. Please install it using: pip install langchain_ollama")
        
        from langchain_ollama import ChatOllama

        print(f"Current Ollama serve model is {os.system('ollama ps')}")
        kwargs["model"] = model_name
        return ChatOllama(**kwargs)

    @classmethod
    def create_groq(cls, model_name: str = "llama-3.3-70b-versatile", **kwargs) -> Any:
        """
        Create Groq chatbot model
        Args:
            model_name (str): Name of the model
            **kwargs: Additional arguments. Options are: temperature, groq_api_key, model_name
        Returns:
            ChatGroq: Chatbot model
        """
        if not check_package("langchain_groq"):
            raise ImportError("Langchain Groq package not found. Please install it using: pip install langchain-groq")

        from langchain_groq import ChatGroq
        kwargs["model"] = model_name
        return ChatGroq(**kwargs)

    @classmethod
    def create_deepseek(cls, model_name: str = "deepseek-chat", **kwargs) -> Any:
        """
        Create Deepseek chatbot model
        Args:
            model_name (str): Name of the model
            **kwargs: Additional arguments
        Returns:     
            ChatDeepseek: Chatbot model
        """
        if not check_package("langchain_deepseek"):
            raise ImportError("Langchain Deepseek package not found. Please install it using: pip install langchain-deepseek")

        from langchain_deepseek import ChatDeepSeek
        kwargs["model"] = model_name
        return ChatDeepSeek(**kwargs)

    @classmethod
    def create_qwen(cls, model_name: str = "qwen", **kwargs) -> Any:
        """
        Create Qwen chatbot model
        Args:
            model_name (str): Name of the model
            **kwargs: Additional arguments
        Returns:
            ChatQwen: Chatbot model
        """
        if not check_package("langchain_community"):
            raise ImportError("Langchain Qwen package not found. Please install it using: pip install langchain_community")

        from langchain_community.chat_models.tongyi import ChatTongyi
        kwargs["model"] = model_name
        return ChatTongyi(streaming=True,**kwargs)

    @classmethod
    def create_hugging_face(cls, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",model_function: str = "image-text-to-text",
                            device='cpu',**kwargs) -> Any:
        """
        Create and load hugging face model.
        Args:
            model_name (str): Name of the model
            model_function (str): model function. Default is image-text-to-text.
            device (str): Device to use. Default is cpu
            **kwargs: Additional arguments
        Returns:
            ChatHuggingFace: Chatbot model
        """
        if not check_package("transformers"):
            raise ImportError("Transformers package not found. Please install it using: pip install transformers")
        if not check_package("langchain_huggingface"):
            raise ImportError("langchain_huggingface package not found. Please install it using: pip install langchain_huggingface")
        if not check_package("torch"):
            raise ImportError("Torch package not found. Please install it using: pip install torch")

        from langchain_huggingface import HuggingFacePipeline
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForImageTextToText,AutoProcessor
        import torch
        
        device = torch.device("cuda" if torch.cuda.is_available() and device == "cuda" else "cpu")
        
        temperature = kwargs.pop("temperature", 0.7)
        max_length = kwargs.pop("max_length", 1024)
        
        if model_function == "image-text-to-text":
            tokenizer = AutoProcessor.from_pretrained(model_name,trust_remote_code=True)
            model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device,
                trust_remote_code=True,
                **kwargs
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device,
                trust_remote_code=True,
                **kwargs)
        
        # Create pipeline
        pipe = pipeline(
            model_function,
            model=model,
            tokenizer=tokenizer,
            max_length=max_length,
            temperature=temperature
        )
        
        # Create and return LangChain HuggingFacePipeline
        return HuggingFacePipeline(pipeline=pipe)

    def _reset_model(self):
        """Reset the model"""
        self.model = self.model.reset()

    def invoke_query(self,query: str,file_path: str = None,get_content_only: bool = True,images: list = None,pydantic_model = None) -> str:
        """
        Invoke the model
        Args:
            query (str): Query to send to the model
            file_path (str): Path to text file to load. Default is None
            get_content_only (bool): Whether to return only content
            images (list): List of images to send to the model
            pydantic_model: Pydantic model for structured output
        Returns:
            str: Response from the model
        """
        if file_path:
            loader = DocumentExtractor()
            data = loader.get_data(file_path)
            query = query + "\n\n" + data

        structured_model = None
        if pydantic_model is not None:
            try:
                structured_model = self.model.with_structured_output(pydantic_model)
            except Exception as e:
                raise ValueError(f"Error with pydantic_model: {e}")
        if structured_model is None:
            structured_model = self.model
        else:
            print("Using structured model with pydantic schema. So get_content_only is set to False.")
            get_content_only = False  # Override to get full response when using structured model
        if images:
            message = self._model_invoke_images(
                images=images,
                prompt=query)
            res = structured_model.invoke([message])
        else:
            res = structured_model.invoke(query)
        if get_content_only:
            try:
                return res.content
            except Exception:
                return res
        return res

    def invoke_query_threads(self,query_list: list,get_content_only: bool = True,input_data: list = None,n_workers: int = 4,pydantic_model=None) -> pd.DataFrame:
        """
        Invoke the model with multiple threads (parallel queries).

        Args:
            query_list (list): List of queries to send to the model
            get_content_only (bool): Whether to return only content
            input_data (list): List of input data for the model
            n_workers (int): Number of workers to use for threading
            pydantic_model: Pydantic model for structured output

        Returns:
            pandas.DataFrame: Response from the model
        """
        if not isinstance(query_list, list):
            raise ValueError("query_list must be a list of strings")
        if input_data is not None and not isinstance(input_data, list):
            raise ValueError("input_data should be a list of messages")
        if input_data is not None and len(input_data) != len(query_list):
            raise ValueError("Length of input_data should equal length of query_list")

        print("Length of query_list:", len(query_list))

        df = pd.DataFrame(query_list, columns=["query"])
        df["response"] = None
        df["input_data"] = None if input_data is None else input_data

        structured_model = None
        if pydantic_model is not None:
            try:
                structured_model = self.model.with_structured_output(pydantic_model)
                print("Using structured model with Pydantic schema. Setting get_content_only=False.")
                get_content_only = False
            except Exception as e:
                raise ValueError(f"Error initializing pydantic_model: {e}")
        else:
            structured_model = self.model

        def process_one(i, query_data):
            try:
                if input_data is not None and len(input_data) > 0:
                    image_data = input_data[i]
                    message = self._model_invoke_images(images=image_data, prompt=query_data)
                    res = structured_model.invoke([message])
                else:
                    res = structured_model.invoke(query_data)

                if get_content_only:
                    try:
                        res = res.content
                    except Exception:
                        pass
                return i, query_data, res
            except Exception as e:
                return i, query_data, f"Error: {e}"

        # Run all queries concurrently
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(process_one, i, q) for i, q in enumerate(query_list)]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing queries"):
                i, query_data, res = future.result()
                df.at[i, "query"] = query_data
                df.at[i, "response"] = res

        df.sort_index(inplace=True)
        return df

    def _image_to_base64(self,image):
        with open(image, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def _model_invoke_images(self, images: list, prompt: str) -> str:
        """
        Function to invoke the model with images
        Args:
            images (list): List of images
            prompt (str): Prompt
        Returns:
            str: Output from the model
        """
        if not isinstance(images, list):
            images = [images]
        base64_images = [self._image_to_base64(image) for image in images]
        image_prompt_create = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_images[i]}"}} for i in range(len(images))]
        prompt_new = [{"type": "text", "text": prompt}, *image_prompt_create]

        message = HumanMessage(content=prompt_new)
        return message
        
    def _get_llm_metadata(self):
        """
        Returns Basic metadata about the LLM
        """
        print("Model Name: ", self.model)
        print("Model Temperature: ", self.model.temperature)
        print("Model Max Tokens: ", self.model.max_output_tokens)
        print("Model Top P: ", self.model.top_p)
        print("Model Top K: ", self.model.top_k)
        print("Model Input Schema:",self.model.input_schema)

