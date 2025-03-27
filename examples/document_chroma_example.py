"""
CSV and PowerPoint Extraction with Chroma DB Example

This example demonstrates how to:
1. Extract data from CSV files using the CSVExtractor
2. Extract content from PowerPoint files using the PowerPointExtractor
3. Generate embeddings from the extracted content
4. Store the embeddings in a Chroma vector database
5. Perform semantic search queries on the embedded content

Requirements:
- CSV and/or PowerPoint files to process
- Required packages: pandas, python-pptx, chromadb, langchain

Usage:
1. Place CSV files in a directory called 'csv_files'
2. Place PowerPoint files in a directory called 'ppt_files'
3. Run the script: python document_chroma_example.py
"""

import os
import sys
from typing import List, Dict, Optional

# Add the project root to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mb_rag.utils.document_extract import CSVExtractor, PowerPointExtractor
from mb_rag.rag.embeddings import embedding_generator

# Configuration
CSV_DIR = "./csv_files"  # Directory containing CSV files
PPT_DIR = "./ppt_files"  # Directory containing PowerPoint files
EMBEDDINGS_DIR = "./document_embeddings"  # Directory to save embeddings
CHUNK_SIZE = 500  # Size of text chunks for embeddings
CHUNK_OVERLAP = 50  # Overlap between chunks

def get_csv_files(directory: str) -> List[str]:
    """
    Get all CSV files in a directory.
    
    Args:
        directory (str): Directory path
        
    Returns:
        List[str]: List of CSV file paths
    """
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return []
    
    csv_files = []
    for file in os.listdir(directory):
        if file.lower().endswith('.csv'):
            csv_files.append(os.path.join(directory, file))
    
    return csv_files

def get_ppt_files(directory: str) -> List[str]:
    """
    Get all PowerPoint files in a directory.
    
    Args:
        directory (str): Directory path
        
    Returns:
        List[str]: List of PowerPoint file paths
    """
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return []
    
    ppt_files = []
    for file in os.listdir(directory):
        if file.lower().endswith(('.ppt', '.pptx')):
            ppt_files.append(os.path.join(directory, file))
    
    return ppt_files

def extract_csv_content(csv_files: List[str]) -> List:
    """
    Extract content from CSV files.
    
    Args:
        csv_files (List[str]): List of CSV file paths
        
    Returns:
        List: List of Document objects
    """
    extractor = CSVExtractor()
    all_documents = []
    
    for csv_file in csv_files:
        try:
            print(f"Extracting content from {csv_file}...")
            documents = extractor.extract_csv(
                csv_file, 
                include_stats=True,
                chunk_by_row=True,
                rows_per_chunk=20
            )
            all_documents.extend(documents)
            print(f"  Extracted {len(documents)} chunks")
        except Exception as e:
            print(f"  Error extracting from {csv_file}: {str(e)}")
    
    return all_documents

def extract_ppt_content(ppt_files: List[str]) -> List:
    """
    Extract content from PowerPoint files.
    
    Args:
        ppt_files (List[str]): List of PowerPoint file paths
        
    Returns:
        List: List of Document objects
    """
    extractor = PowerPointExtractor()
    all_documents = []
    
    for ppt_file in ppt_files:
        try:
            print(f"Extracting content from {ppt_file}...")
            documents = extractor.extract_ppt(
                ppt_file,
                include_notes=True,
                include_hidden_slides=False
            )
            all_documents.extend(documents)
            print(f"  Extracted {len(documents)} slides")
        except Exception as e:
            print(f"  Error extracting from {ppt_file}: {str(e)}")
    
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
    temp_file_path = "temp_document_content.txt"
    with open(temp_file_path, "w", encoding="utf-8") as f:
        for i, doc in enumerate(documents):
            source = doc.metadata.get('source', 'Unknown source')
            file_type = doc.metadata.get('file_type', 'Unknown type')
            
            # Write header based on file type
            if file_type == 'csv':
                rows = doc.metadata.get('rows', 'Unknown')
                columns = doc.metadata.get('columns', [])
                f.write(f"--- CSV File: {os.path.basename(source)} ---\n")
                f.write(f"Rows: {rows}, Columns: {', '.join(columns)}\n\n")
            elif file_type in ['ppt', 'pptx']:
                slide_number = doc.metadata.get('slide_number', 'Unknown')
                total_slides = doc.metadata.get('total_slides', 'Unknown')
                title = doc.metadata.get('title', 'Untitled')
                f.write(f"--- PowerPoint File: {os.path.basename(source)} ---\n")
                f.write(f"Slide {slide_number}/{total_slides}: {title}\n\n")
            else:
                f.write(f"--- Document {i+1} ---\n\n")
            
            # Write content
            f.write(doc.page_content)
            f.write("\n\n" + "="*50 + "\n\n")
    
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
    
    conversation_file = "document_conversation_history.txt"
    
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
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(PPT_DIR, exist_ok=True)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    
    # Get CSV files
    csv_files = get_csv_files(CSV_DIR)
    print(f"Found {len(csv_files)} CSV files: {[os.path.basename(f) for f in csv_files]}")
    
    # Get PowerPoint files
    ppt_files = get_ppt_files(PPT_DIR)
    print(f"Found {len(ppt_files)} PowerPoint files: {[os.path.basename(f) for f in ppt_files]}")
    
    if not csv_files and not ppt_files:
        print("No CSV or PowerPoint files found.")
        print("Please add files to the respective directories and run the script again.")
        return
    
    # Extract content
    all_documents = []
    
    if csv_files:
        csv_documents = extract_csv_content(csv_files)
        all_documents.extend(csv_documents)
        print(f"Extracted {len(csv_documents)} documents from CSV files.")
    
    if ppt_files:
        ppt_documents = extract_ppt_content(ppt_files)
        all_documents.extend(ppt_documents)
        print(f"Extracted {len(ppt_documents)} documents from PowerPoint files.")
    
    if not all_documents:
        print("No content extracted from files.")
        return
    
    # Create embeddings
    em_gen = create_embeddings(all_documents, EMBEDDINGS_DIR)
    
    # Example search
    sample_query = "What are the main topics discussed in these documents?"
    perform_search(em_gen, EMBEDDINGS_DIR, sample_query)
    
    # Create RAG conversation chain
    rag_chain = create_rag_conversation(em_gen, EMBEDDINGS_DIR)
    
    # Interactive conversation
    interactive_conversation(em_gen, rag_chain)

if __name__ == "__main__":
    main()
