"""
PDF Table Extraction to CSV and Chroma DB Example

This example demonstrates how to:
1. Extract tables from PDF files and convert them to CSV format
2. Process the CSV data
3. Generate embeddings from the CSV data
4. Store the embeddings in a Chroma vector database
5. Perform semantic search queries on the embedded content

Requirements:
- PDF files with tables to process
- Required packages: pypdf, tabula-py, pandas, chromadb, langchain

Usage:
1. Place PDF files with tables in a directory
2. Update the PDF_DIR path in this script
3. Run the script: python pdf_csv_chroma_example.py
"""

import os
import sys
import pandas as pd
from typing import List, Dict, Optional

# Add the project root to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mb_rag.utils.pdf_extract import PDFExtractor, PDFToCSV
from mb_rag.rag.embeddings import embedding_generator

# Configuration
PDF_DIR = "./pdf_files"  # Directory containing PDF files
CSV_DIR = "./csv_files"  # Directory to save CSV files
EMBEDDINGS_DIR = "./csv_embeddings"  # Directory to save embeddings
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

def extract_tables_to_csv(pdf_files: List[str], output_dir: str) -> List[str]:
    """
    Extract tables from PDF files and save as CSV.
    
    Args:
        pdf_files (List[str]): List of PDF file paths
        output_dir (str): Directory to save CSV files
        
    Returns:
        List[str]: List of CSV file paths
    """
    converter = PDFToCSV()
    all_csv_files = []
    
    for pdf_file in pdf_files:
        try:
            print(f"Extracting tables from {pdf_file}...")
            csv_files = converter.convert_pdf_tables_to_csv(pdf_file, output_dir)
            all_csv_files.extend(csv_files)
            print(f"  Extracted {len(csv_files)} tables to CSV")
        except Exception as e:
            print(f"  Error extracting tables from {pdf_file}: {str(e)}")
    
    return all_csv_files

def process_csv_files(csv_files: List[str]) -> str:
    """
    Process CSV files and combine into a single text file for embedding.
    
    Args:
        csv_files (List[str]): List of CSV file paths
        
    Returns:
        str: Path to the combined text file
    """
    combined_text_path = "combined_csv_data.txt"
    
    with open(combined_text_path, "w", encoding="utf-8") as f:
        for csv_file in csv_files:
            try:
                # Read CSV file
                df = pd.read_csv(csv_file)
                
                # Write CSV metadata and content to text file
                f.write(f"--- CSV File: {os.path.basename(csv_file)} ---\n")
                f.write(f"Columns: {', '.join(df.columns)}\n")
                f.write(f"Rows: {len(df)}\n\n")
                
                # Write CSV content in a readable format
                f.write(df.to_string(index=False))
                f.write("\n\n")
                
                # Write some basic statistics
                f.write("Basic Statistics:\n")
                for column in df.columns:
                    if pd.api.types.is_numeric_dtype(df[column]):
                        f.write(f"{column} - Min: {df[column].min()}, Max: {df[column].max()}, "
                               f"Mean: {df[column].mean():.2f}, Median: {df[column].median()}\n")
                
                f.write("\n" + "="*50 + "\n\n")
                
            except Exception as e:
                f.write(f"Error processing {csv_file}: {str(e)}\n\n")
    
    print(f"Combined CSV data saved to {combined_text_path}")
    return combined_text_path

def create_embeddings(text_file_path: str, embeddings_dir: str):
    """
    Create embeddings from text file and store in Chroma DB.
    
    Args:
        text_file_path (str): Path to text file
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
    
    # Generate embeddings
    print(f"Generating embeddings and storing in {embeddings_dir}...")
    em_gen.generate_text_embeddings(
        text_data_path=[text_file_path],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        folder_save_path=embeddings_dir,
        replace_existing=True
    )
    
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
    
    conversation_file = "csv_conversation_history.txt"
    
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
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    
    # Get PDF files
    pdf_files = get_pdf_files(PDF_DIR)
    if not pdf_files:
        print(f"No PDF files found in {PDF_DIR}")
        print("Please add PDF files to this directory and run the script again.")
        return
    
    print(f"Found {len(pdf_files)} PDF files: {[os.path.basename(f) for f in pdf_files]}")
    
    # Extract tables from PDFs to CSV
    csv_files = extract_tables_to_csv(pdf_files, CSV_DIR)
    if not csv_files:
        print("No tables extracted from PDF files.")
        return
    
    print(f"Extracted {len(csv_files)} CSV files: {[os.path.basename(f) for f in csv_files]}")
    
    # Process CSV files
    combined_text_file = process_csv_files(csv_files)
    
    # Create embeddings
    em_gen = create_embeddings(combined_text_file, EMBEDDINGS_DIR)
    
    # Example search
    sample_query = "What are the key statistics in the data?"
    perform_search(em_gen, EMBEDDINGS_DIR, sample_query)
    
    # Create RAG conversation chain
    rag_chain = create_rag_conversation(em_gen, EMBEDDINGS_DIR)
    
    # Interactive conversation
    interactive_conversation(em_gen, rag_chain)
    
    # Clean up temporary file
    os.remove(combined_text_file)

if __name__ == "__main__":
    main()
