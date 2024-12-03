# MB-RAG: Modular Building Blocks for Retrieval-Augmented Generation

MB-RAG is a flexible Python package that provides modular building blocks for creating RAG (Retrieval-Augmented Generation) applications. It integrates multiple LLM providers, embedding models, and utility functions to help you build powerful AI applications.

## Features

- **Multiple LLM Support**: 
  - OpenAI (GPT-4, GPT-3.5)
  - Anthropic (Claude)
  - Google (Gemini)
  - Ollama (Local models)

- **RAG Capabilities**:
  - Text splitting and chunking
  - Multiple embedding models
  - Vector store integration
  - Conversation history management
  - Context-aware retrieval

- **Web Tools**:
  - Web browsing agent
  - Content scraping
  - Link extraction
  - Search functionality

- **Image Processing**:
  - Bounding box generation
  - Image analysis with Gemini Vision
  - OpenCV integration

## Installation

1. Basic Installation:
```bash
pip install mb_rag
```


## Quick Start

### Basic Model load and ask question
```
from mb_chatbot.basic import load_model,model_invoke

model = load_model(model_name: str = "gpt-4o", model_type: str = 'openai')
response = model_invoke(model,question='What is AI?')
response = model_invoke(model,question='what is there in the all the images?',images=['path1','path2']) ## running with images
```

### Basic RAG Example
```python
from mb_rag.rag.embeddings import embedding_generator
from mb_rag.chatbot.basic import get_chatbot_openai

# Initialize embeddings
embedder = embedding_generator(model='openai')

# Generate embeddings from text files
embedder.generate_text_embeddings(
    text_data_path=['data.txt'],
    folder_save_path='./embeddings'
)

# Load retriever
retriever = embedder.load_retriever('embeddings')

# Create RAG chain
chatbot = get_chatbot_openai()
rag_chain = embedder.generate_rag_chain(retriever=retriever, llm=chatbot)

# Query your documents
response = embedder.conversation_chain(
    "What are the key points in the document?",
    rag_chain
)
print(response)
```

### Web Browsing Example
```python
from mb_rag.agents.web_browser_agent import WebBrowserAgent

# Initialize web browser agent
agent = WebBrowserAgent()

# Browse and extract content
content = agent.browse("https://example.com")
links = agent._extract_links("https://example.com")
```

### Image Processing Example
```python
from mb_rag.utils.bounding_box import google_model, generate_bounding_box

# Initialize Gemini model
model = google_model()

# Generate bounding boxes
boxes = generate_bounding_box(
    model,
    "image.jpg",
    "Return bounding boxes of objects"
)
```

## Package Structure

```
mb_rag/
├── rag/
│   └── embeddings.py      # Core RAG functionality
├── chatbot/
│   ├── basic.py          # Basic chatbot implementations
│   └── chains.py         # LangChain integration
├── agents/
│   ├── run_agent.py      # Agent execution
│   └── web_browser_agent.py  # Web browsing capabilities
└── utils/
    ├── bounding_box.py   # Image processing utilities
    └── extra.py          # Additional utilities
```

## Dependencies

Core dependencies:
- langchain-core
- langchain-community
- langchain
- python-dotenv

Optional dependencies are organized by feature:
- Language Models (OpenAI, Anthropic, Google, Ollama)
- Web Tools (BeautifulSoup, Requests)
- Image Processing (Pillow, OpenCV)
- Vector Stores (Chroma, FAISS)
- Cloud Services (AWS, Google Cloud)

See `requirements.txt` for a complete list of optional dependencies.

## Environment Setup

Create a `.env` file in your project root:
```env
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
```

## Error Handling

The package includes comprehensive error checking:
- Dependency verification before operations
- Clear error messages with installation instructions
- Helpful debugging information
- Fallbacks when possible

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## Acknowledgments

Built with [LangChain](https://github.com/langchain-ai/langchain) and other amazing open-source projects.
