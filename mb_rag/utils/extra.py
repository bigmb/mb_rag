## extra functions for RAG's

import os
from dotenv import load_dotenv
import PyPDF2

__all__ = ["load_env_file"]

def load_env_file(file_path='.env'):
    """
    Load environment variables from a .env file.

    Args:
        file_path (str): Path to the .env file. Defaults to '.env'.

    Returns:
        None
    """
    load_dotenv(file_path)

    # Get the loaded environment variables
    env_vars = os.environ

    return env_vars


def pdf_to_text(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except PyPDF2.errors.PdfReadError as e:
        print(f"Error reading {pdf_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred with {pdf_path}: {e}")
    return text

def convert_pdfs_in_folder(folder_path):
    """
    Convert all PDF files in the given folder to text files.
    Args:
        folder_path (str): Path to the folder containing the PDF files.
    Returns:
        None
    Example : convert_pdfs_in_folder('/folder_path') # folder_path is the path to the folder containing the PDF files.
    The converted PDF files and text files will be created in the same folder
    """
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            text = pdf_to_text(pdf_path)
            if text:  # Only write to file if text is not empty
                text_filename = os.path.splitext(filename)[0] + '.txt'
                text_path = os.path.join(folder_path, text_filename)
                with open(text_path, 'w', encoding='utf-8') as text_file:
                    text_file.write(text)
                print(f"Converted: {filename} to {text_filename}")
            else:
                print(f"No text extracted from {filename}")