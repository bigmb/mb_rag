# MB-RAG: Modular Building Blocks for Retrieval-Augmented Generation

MB-RAG is a flexible Python package that provides modular building blocks for creating RAG (Retrieval-Augmented Generation) applications. It integrates multiple LLM providers, embedding models, and utility functions to help you build powerful AI applications.

## Features

- **Multiple LLM Support**: 
  - OpenAI (GPT-4, GPT-3.5)
  - Anthropic (Claude)
  - Google (Gemini)
  - Ollama (Local models)
  - Groq

- **RAG Capabilities**:
  - Text splitting and chunking
  - Multiple embedding models
  - Vector store integration
  - Conversation history management
  - Context-aware retrieval

- **Image Processing**:
  - Bounding box generation with Gemini Vision
  - Custom image annotations
  - Multiple output formats
  - Batch processing capabilities

## Installation

1. Basic Installation:
```bash
pip install mb_rag
```

## Quick Start

### Basic Chat Examples
## check example_llm.ipynb for more details
```python
from mb_rag.basic import ModelFactory

# 1. Simple Query with ModelFactory
model = ModelFactory(model_type="openai", model_name="gpt-4o")
response = model.invoke_query("What is artificial intelligence?")
print(response)

# 2. Image Analysis
model = ModelFactory(model_type="openai", model_name="gpt-4o")
response = model.invoke_query(
    "What's in these images?",
    images=["image1.jpg", "image2.jpg"]
)
print(response)

## other models
# Anthropic Claude
claude_model = ModelFactory(
    model_type="anthropic",
    model_name="claude-3-opus-20240229"
)
response = claude_model.invoke_query("Explain quantum computing")

# Google Gemini
gemini_model = ModelFactory(
    model_type="google",
    model_name="gemini-1.5-pro-latest"
)
response = gemini_model.invoke_query("Describe the solar system")

# Local Ollama
ollama_model = ModelFactory(
    model_type="ollama",
    model_name="llama3.1"
)
response = ollama_model.invoke_query("What is the meaning of life?")

## Running in threads 
response = model.invoke_query_threads(query_list=['q1','q2'],input_data=[[images_data],[images_data]],n_workers=4)


## check example_conversation.ipynb for more details

from mb_rag.chatbot.conversation import ConversationModel
# 3. Conversation with Context : if file_path/message_list is not provided, it will create a new conversation
conversation = ConversationModel(llm=ModelFactory(model_type="openai", model_name="gpt-4o"),
                                file_path=None,
                                message_list=None)

conversation.initialize_conversation()

# Continue the conversation
response = conversation.add_message("How is it different from deep learning?")
print(response)

# Access conversation history
print("\nAll messages:")
for message in conversation.all_messages_content:
    print(message)

# Save conversation
conversation.save_conversation("chat_history.txt")

```

### Embeddings and RAG Example
```python
from mb_rag.rag.embeddings import embedding_generator

# Initialize embedding generator
em_gen = embedding_generator(
    model="openai",
    model_type="text-embedding-3-small",
    vector_store_type="chroma"
)

# Generate embeddings from text files
em_gen.generate_text_embeddings(
    text_data_path=['./data.txt'],
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

# Generate RAG chain for conversation
rag_chain = em_gen.generate_rag_chain(retriever=em_retriever)

# Have a conversation with context
response = em_gen.conversation_chain(
    "What is this document about?",
    rag_chain,
    file='conversation_history.txt'  # Optional: Save conversation
)

# Query specific information
results = em_gen.query_embeddings(
    "What are the key points discussed?",
    em_retriever
)

# Add new data to existing embeddings
em_gen.add_data(
    './embeddings',
    ['new_data.txt'],
    chunk_size=500
)

# Web scraping and embedding
db = em_gen.firecrawl_web(
    website="https://github.com",
    mode="scrape",
    file_to_save='./web_embeddings'
)
```

### Image Processing with Bounding Boxes
```python
from mb_rag.utils.bounding_box import BoundingBoxProcessor, BoundingBoxConfig

# Initialize processor with configuration
config = BoundingBoxConfig(
    model_name="gemini-1.5-pro-latest",
    api_key="your-api-key"  # Or use environment variable GOOGLE_API_KEY
)
processor = BoundingBoxProcessor(config)

# Generate bounding boxes
boxes = processor.generate_bounding_boxes(
    "image.jpg",
    prompt="Return bounding boxes of objects"
)

# Add boxes to image with custom styling
processed_img = processor.add_bounding_boxes(
    "image.jpg",
    boxes,
    color=(0, 255, 0),  # Green color
    thickness=2,
    font_scale=0.5,
    show=True  # Display result
)

# Save processed image
processor.save_image(processed_img, "output.jpg")

# Complete processing pipeline
result = processor.process_image(
    "image.jpg",
    output_path="result.jpg",
    show=True
)

# Batch processing
def batch_process_images(processor, image_paths, output_dir, **kwargs):
    """Process multiple images with same settings."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    for img_path in image_paths:
        try:
            output_path = os.path.join(
                output_dir,
                f"processed_{os.path.basename(img_path)}"
            )
            result = processor.process_image(
                img_path,
                output_path=output_path,
                **kwargs
            )
            results.append((img_path, output_path, True))
        except Exception as e:
            results.append((img_path, None, False))
            print(f"Error processing {img_path}: {e}")
    return results

# Example batch processing
images = ["image1.jpg", "image2.jpg", "image3.jpg"]
results = batch_process_images(
    processor,
    images,
    "./batch_output",
    show=False
)
```

## Package Structure

```
mb_rag/
├── rag/
│   └── embeddings.py      # RAG and embedding functionality
├── chatbot/
    └── conversation.py         # Conversation functionality
│   └── chains.py         # LangChain integration
├── agents/
│   ├── run_agent.py      # Agent execution
│   └── web_browser_agent.py  # Web browsing capabilities, Added WebAgent with langgraph
└── utils/
    ├── bounding_box.py   # Image processing utilities
    └── extra.py          # Additional utilities
└── basic.py          # Basic chatbot implementations
```

## Dependencies

Core dependencies:
- langchain-core
- langchain-community
- langchain
- python-dotenv

Optional dependencies by feature:
- Language Models: langchain-openai, langchain-anthropic, langchain-google-genai, langchain-ollama
- Image Processing: Pillow, opencv-python, google-generativeai
- Vector Stores: chromadb
- Web Tools: firecrawl

See `requirements.txt` for a complete list.

## Environment Setup

Create a `.env` file in your project root:
```env
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
```