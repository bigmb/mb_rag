"""
PDF Extraction and Chroma DB Example

This example demonstrates how to:
1. Extract text from PDF files using the PDFExtractor
2. Generate embeddings from the extracted text
3. Store the embeddings in a Chroma vector database
4. Perform semantic search queries on the embedded content

Requirements:
- PDF files to process
- Required packages: pypdf, chromadb, langchain

Usage:
1. Place PDF files in a directory
2. Update the PDF_DIR path in this script
3. Run the script: python pdf_chroma_example.py
"""

import os
import sys
from typing import List, Dict, Optional

# Add the project root to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mb_rag.utils.pdf_extract import PDFExtractor
from mb_rag.rag.embeddings import embedding_generator

# Configuration
PDF_DIR = "./pdf_files"  # Directory containing PDF files
EMBEDDINGS_DIR = "./pdf_embeddings"  # Directory to save embeddings
CHUNK_SIZE = 500  # Size of text chunks for embeddings
CHUNK_OVERLAP = 50  # Overlap between chunks

def get_pdf_files(directory: str) -> List[str]:
    """
    Get all PDF files in a directory.
    
    Args:
        directory (str): Directory path
        
    Returns:
        List[str]: List of PDF file paths
    """
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return []
    
    pdf_files = []
    for file in os.listdir(directory):
        if file.lower().endswith('.pdf'):
            pdf_files.append(os.path.join(directory, file))
    
    return pdf_files

def extract_pdf_content(pdf_files: List[str], extraction_method: str = "pypdf") -> List:
    """
    Extract content from PDF files.
    
    Args:
        pdf_files (List[str]): List of PDF file paths
        extraction_method (str): Method to use for extraction
        
    Returns:
        List: List of Document objects
    """
    extractor = PDFExtractor()
    all_documents = []
    
    for pdf_file in pdf_files:
        try:
            print(f"Extracting content from {pdf_file}...")
            documents = extractor.extract_pdf(pdf_file, extraction_method=extraction_method)
            all_documents.extend(documents)
            print(f"  Extracted {len(documents)} pages")
        except Exception as e:
            print(f"  Error extracting from {pdf_file}: {str(e)}")
    
    return all_documents

def create_embeddings(documents, embeddings_dir: str):
    """
    Create embeddings from documents and store in Chroma DB.
    
    Args:
        documents: List of Document objects
        embeddings_dir (str): Directory to save embeddings
        
    Returns:
        embedding_generator: Configured embedding generator
    """
    # Initialize embedding generator with OpenAI embeddings and Chroma vector store
    em_gen = embedding_generator(
        model="openai",  # Can be changed to "ollama", "google", etc.
        model_type="text-embedding-3-small",
        vector_store_type="chroma"
    )
    
    # Create temporary text file with document content
    temp_file_path = "temp_pdf_content.txt"
    with open(temp_file_path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(f"--- Page {doc.metadata.get('page', 'unknown')} of {doc.metadata.get('source', 'unknown')} ---\n")
            f.write(doc.page_content)
            f.write("\n\n")
    
    # Generate embeddings
    print(f"Generating embeddings and storing in {embeddings_dir}...")
    em_gen.generate_text_embeddings(
        text_data_path=[temp_file_path],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        folder_save_path=embeddings_dir,
        replace_existing=True
    )
    
    # Clean up temporary file
    os.remove(temp_file_path)
    
    return em_gen

def perform_search(em_gen, embeddings_dir: str, query: str, k: int = 3):
    """
    Perform semantic search on the embeddings.
    
    Args:
        em_gen: Embedding generator
        embeddings_dir (str): Directory with embeddings
        query (str): Search query
        k (int): Number of results to return
    """
    # Load embeddings and create retriever
    print(f"Loading embeddings from {embeddings_dir}...")
    em_retriever = em_gen.load_retriever(
        embeddings_dir,
        search_params=[{"k": k, "score_threshold": 0.1}]
    )
    
    # Perform search
    print(f"\nSearching for: '{query}'")
    results = em_gen.query_embeddings(query, em_retriever)
    
    # Display results
    print(f"\nFound {len(results)} results:")
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        source = doc.metadata.get('source', 'Unknown source')
        if isinstance(source, str) and source.endswith('.txt'):
            # For the temporary text file, extract the original source from content
            source_line = doc.page_content.split('\n')[0]
            if '---' in source_line:
                source = source_line.strip('- ')
        print(f"Source: {source}")
        print(f"Content: {doc.page_content[:300]}...")  # Show first 300 chars
    
    return results

def create_rag_conversation(em_gen, embeddings_dir: str):
    """
    Create a RAG conversation chain.
    
    Args:
        em_gen: Embedding generator
        embeddings_dir (str): Directory with embeddings
        
    Returns:
        Any: RAG chain for conversation
    """
    # Load embeddings and create retriever
    em_retriever = em_gen.load_retriever(
        embeddings_dir,
        search_params=[{"k": 3, "score_threshold": 0.1}]
    )
    
    # Generate RAG chain
    print("\nCreating RAG conversation chain...")
    rag_chain = em_gen.generate_rag_chain(retriever=em_retriever)
    
    return rag_chain

def interactive_conversation(em_gen, rag_chain):
    """
    Interactive conversation with RAG.
    
    Args:
        em_gen: Embedding generator
        rag_chain: RAG chain for conversation
    """
    print("\n=== Interactive Conversation ===")
    print("Type 'exit' to end the conversation")
    
    conversation_file = "pdf_conversation_history.txt"
    
    while True:
        query = input("\nYou: ")
        if query.lower() == 'exit':
            break
        
        # Get response
        response = em_gen.conversation_chain(
            query,
            rag_chain,
            file=conversation_file
        )
        
        # Response is printed by the conversation_chain method

def main():
    """Main function to run the example."""
    # Create directories if they don't exist
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    
    # Get PDF files
    pdf_files = get_pdf_files(PDF_DIR)
    if not pdf_files:
        print(f"No PDF files found in {PDF_DIR}")
        print("Please add PDF files to this directory and run the script again.")
        return
    
    print(f"Found {len(pdf_files)} PDF files: {[os.path.basename(f) for f in pdf_files]}")
    
    # Extract content from PDFs
    documents = extract_pdf_content(pdf_files)
    if not documents:
        print("No content extracted from PDF files.")
        return
    
    print(f"Extracted {len(documents)} pages from all PDFs.")
    
    # Create embeddings
    em_gen = create_embeddings(documents, EMBEDDINGS_DIR)
    
    # Example search
    sample_query = "What are the main topics discussed in these documents?"
    perform_search(em_gen, EMBEDDINGS_DIR, sample_query)
    
    # Create RAG conversation chain
    rag_chain = create_rag_conversation(em_gen, EMBEDDINGS_DIR)
    
    # Interactive conversation
    interactive_conversation(em_gen, rag_chain)

if __name__ == "__main__":
    main()
