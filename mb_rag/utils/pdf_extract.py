"""
PDF Extraction Module

This module provides functionality for extracting text and metadata from PDF files.
It supports various extraction methods and includes features for handling different
PDF structures, including tables and images.

Example Usage:
    ```python
    # Initialize PDF extractor
    extractor = PDFExtractor()
    
    # Extract text from a PDF file
    docs = extractor.extract_pdf("document.pdf")
    
    # Extract with specific options
    docs = extractor.extract_pdf(
        "document.pdf",
        extraction_method="pdfplumber",
        extract_images=True
    )
    
    # Extract from multiple PDFs
    docs = extractor.extract_multiple_pdfs(
        ["doc1.pdf", "doc2.pdf"],
        extraction_method="pymupdf"
    )
    ```

Features:
    - Multiple extraction methods (PyPDF2, PDFPlumber, PyMuPDF)
    - Text and metadata extraction
    - Optional image extraction
    - Table detection and extraction
    - Batch processing for multiple PDFs
"""

import os
import tempfile
from typing import List, Dict, Optional, Union, Any, Tuple
import importlib.util
from langchain_core.documents import Document

class PDFExtractor:
    """
    Class for extracting text and metadata from PDF files.
    
    This class provides methods for extracting content from PDF files using
    different extraction methods and processing options.
    
    Args:
        logger: Optional logger instance for logging operations
        
    Example:
        ```python
        extractor = PDFExtractor()
        docs = extractor.extract_pdf("document.pdf")
        ```
    """
    
    def __init__(self, logger=None):
        """Initialize the PDF extractor."""
        self.logger = logger
        
    @staticmethod
    def check_package(package_name: str) -> bool:
        """
        Check if a Python package is installed.
        
        Args:
            package_name (str): Name of the package to check
            
        Returns:
            bool: True if package is installed, False otherwise
        """
        return importlib.util.find_spec(package_name) is not None
    
    def check_file(self, file_path: str) -> bool:
        """
        Check if file exists.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            bool: True if file exists, False otherwise
        """
        return os.path.exists(file_path)
    
    def extract_pdf(self, pdf_path: str, extraction_method: str = "pypdf", 
                   extract_images: bool = False, extract_tables: bool = False,
                   **kwargs) -> List[Document]:
        """
        Extract text and metadata from a PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            extraction_method (str): Method to use for extraction 
                                    ("pypdf", "pdfplumber", or "pymupdf")
            extract_images (bool): Whether to extract images
            extract_tables (bool): Whether to extract tables
            **kwargs: Additional arguments for the extraction method
            
        Returns:
            List[Document]: List of Document objects containing extracted content
            
        Raises:
            ValueError: If the file doesn't exist or extraction method is invalid
            ImportError: If required packages are not installed
        """
        if not self.check_file(pdf_path):
            raise ValueError(f"File {pdf_path} not found")
        
        if extraction_method == "pypdf":
            return self._extract_with_pypdf(pdf_path, **kwargs)
        elif extraction_method == "pdfplumber":
            return self._extract_with_pdfplumber(pdf_path, extract_tables, **kwargs)
        elif extraction_method == "pymupdf":
            return self._extract_with_pymupdf(pdf_path, extract_images, **kwargs)
        else:
            raise ValueError(f"Invalid extraction method: {extraction_method}")
    
    def extract_multiple_pdfs(self, pdf_paths: List[str], extraction_method: str = "pypdf",
                             extract_images: bool = False, extract_tables: bool = False,
                             **kwargs) -> List[Document]:
        """
        Extract text and metadata from multiple PDF files.
        
        Args:
            pdf_paths (List[str]): List of paths to PDF files
            extraction_method (str): Method to use for extraction
            extract_images (bool): Whether to extract images
            extract_tables (bool): Whether to extract tables
            **kwargs: Additional arguments for the extraction method
            
        Returns:
            List[Document]: List of Document objects containing extracted content
        """
        all_docs = []
        for pdf_path in pdf_paths:
            try:
                docs = self.extract_pdf(
                    pdf_path, 
                    extraction_method=extraction_method,
                    extract_images=extract_images,
                    extract_tables=extract_tables,
                    **kwargs
                )
                all_docs.extend(docs)
                if self.logger:
                    self.logger.info(f"Successfully extracted content from {pdf_path}")
                else:
                    print(f"Successfully extracted content from {pdf_path}")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error extracting from {pdf_path}: {str(e)}")
                else:
                    print(f"Error extracting from {pdf_path}: {str(e)}")
        
        return all_docs
    
    def _extract_with_pypdf(self, pdf_path: str, **kwargs) -> List[Document]:
        """
        Extract text using PyPDF2.
        
        Args:
            pdf_path (str): Path to the PDF file
            **kwargs: Additional arguments for PyPDF2
            
        Returns:
            List[Document]: List of Document objects
            
        Raises:
            ImportError: If PyPDF2 is not installed
        """
        if not self.check_package("pypdf"):
            raise ImportError("PyPDF2 package not found. Please install: pip install pypdf")
        
        from langchain_community.document_loaders import PyPDFLoader
        
        loader = PyPDFLoader(pdf_path, **kwargs)
        documents = loader.load()
        
        if self.logger:
            self.logger.info(f"Extracted {len(documents)} pages from {pdf_path} using PyPDF2")
        else:
            print(f"Extracted {len(documents)} pages from {pdf_path} using PyPDF2")
            
        return documents
    
    def _extract_with_pdfplumber(self, pdf_path: str, extract_tables: bool = False, **kwargs) -> List[Document]:
        """
        Extract text using PDFPlumber.
        
        Args:
            pdf_path (str): Path to the PDF file
            extract_tables (bool): Whether to extract tables
            **kwargs: Additional arguments for PDFPlumber
            
        Returns:
            List[Document]: List of Document objects
            
        Raises:
            ImportError: If PDFPlumber is not installed
        """
        if not self.check_package("pdfplumber"):
            raise ImportError("PDFPlumber package not found. Please install: pip install pdfplumber")
        
        import pdfplumber
        
        documents = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                
                metadata = {
                    "source": pdf_path,
                    "page": i + 1,
                    "total_pages": len(pdf.pages)
                }
                
                if extract_tables:
                    tables = page.extract_tables()
                    if tables:
                        table_text = []
                        for table in tables:
                            table_rows = []
                            for row in table:
                                # Filter out None values and convert to strings
                                row_text = [str(cell) if cell is not None else "" for cell in row]
                                table_rows.append(" | ".join(row_text))
                            table_text.append("\n".join(table_rows))
                        
                        metadata["tables"] = table_text
                        # Append table text to the main text
                        text += "\n\nTABLES:\n" + "\n\n".join(table_text)
                
                documents.append(Document(page_content=text, metadata=metadata))
        
        if self.logger:
            self.logger.info(f"Extracted {len(documents)} pages from {pdf_path} using PDFPlumber")
        else:
            print(f"Extracted {len(documents)} pages from {pdf_path} using PDFPlumber")
            
        return documents
    
    def _extract_with_pymupdf(self, pdf_path: str, extract_images: bool = False, **kwargs) -> List[Document]:
        """
        Extract text using PyMuPDF (fitz).
        
        Args:
            pdf_path (str): Path to the PDF file
            extract_images (bool): Whether to extract images
            **kwargs: Additional arguments for PyMuPDF
            
        Returns:
            List[Document]: List of Document objects
            
        Raises:
            ImportError: If PyMuPDF is not installed
        """
        if not self.check_package("fitz"):
            raise ImportError("PyMuPDF package not found. Please install: pip install pymupdf")
        
        import fitz
        
        documents = []
        temp_dir = None
        
        try:
            if extract_images:
                temp_dir = tempfile.mkdtemp()
            
            with fitz.open(pdf_path) as doc:
                for i, page in enumerate(doc):
                    text = page.get_text()
                    
                    metadata = {
                        "source": pdf_path,
                        "page": i + 1,
                        "total_pages": len(doc),
                        "title": doc.metadata.get("title", ""),
                        "author": doc.metadata.get("author", ""),
                        "subject": doc.metadata.get("subject", ""),
                        "keywords": doc.metadata.get("keywords", "")
                    }
                    
                    if extract_images and temp_dir:
                        image_list = page.get_images(full=True)
                        image_paths = []
                        
                        for img_index, img in enumerate(image_list):
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            
                            image_path = os.path.join(
                                temp_dir, 
                                f"page{i+1}_img{img_index+1}.{base_image['ext']}"
                            )
                            
                            with open(image_path, "wb") as img_file:
                                img_file.write(image_bytes)
                            
                            image_paths.append(image_path)
                        
                        if image_paths:
                            metadata["images"] = image_paths
                    
                    documents.append(Document(page_content=text, metadata=metadata))
        
        finally:
            # Clean up temporary directory if it was created
            if extract_images and temp_dir and os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir)
        
        if self.logger:
            self.logger.info(f"Extracted {len(documents)} pages from {pdf_path} using PyMuPDF")
        else:
            print(f"Extracted {len(documents)} pages from {pdf_path} using PyMuPDF")
            
        return documents


class PDFToCSV:
    """
    Class for converting PDF tables to CSV format.
    
    This class provides methods for extracting tables from PDF files
    and converting them to CSV format.
    
    Args:
        logger: Optional logger instance for logging operations
        
    Example:
        ```python
        converter = PDFToCSV()
        csv_paths = converter.convert_pdf_tables_to_csv("document.pdf", "output_dir")
        ```
    """
    
    def __init__(self, logger=None):
        """Initialize the PDF to CSV converter."""
        self.logger = logger
        
    @staticmethod
    def check_package(package_name: str) -> bool:
        """
        Check if a Python package is installed.
        
        Args:
            package_name (str): Name of the package to check
            
        Returns:
            bool: True if package is installed, False otherwise
        """
        return importlib.util.find_spec(package_name) is not None
    
    def convert_pdf_tables_to_csv(self, pdf_path: str, output_dir: str = None, 
                                 pages: List[int] = None) -> List[str]:
        """
        Extract tables from PDF and convert to CSV.
        
        Args:
            pdf_path (str): Path to the PDF file
            output_dir (str): Directory to save CSV files (default: same as PDF)
            pages (List[int]): Specific pages to extract tables from (default: all)
            
        Returns:
            List[str]: Paths to the created CSV files
            
        Raises:
            ImportError: If required packages are not installed
            ValueError: If the PDF file doesn't exist
        """
        if not os.path.exists(pdf_path):
            raise ValueError(f"PDF file not found: {pdf_path}")
        
        if not self.check_package("tabula"):
            raise ImportError("Tabula-py package not found. Please install: pip install tabula-py")
        
        import tabula
        import pandas as pd
        
        # Determine output directory
        if output_dir is None:
            output_dir = os.path.dirname(pdf_path)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract tables
        try:
            if pages:
                dfs = tabula.read_pdf(pdf_path, pages=pages, multiple_tables=True)
            else:
                dfs = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error extracting tables: {str(e)}")
            else:
                print(f"Error extracting tables: {str(e)}")
            return []
        
        if not dfs:
            if self.logger:
                self.logger.warning(f"No tables found in {pdf_path}")
            else:
                print(f"No tables found in {pdf_path}")
            return []
        
        # Save tables to CSV
        csv_paths = []
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        for i, df in enumerate(dfs):
            if not df.empty:
                csv_path = os.path.join(output_dir, f"{pdf_name}_table_{i+1}.csv")
                df.to_csv(csv_path, index=False)
                csv_paths.append(csv_path)
                
                if self.logger:
                    self.logger.info(f"Saved table {i+1} to {csv_path}")
                else:
                    print(f"Saved table {i+1} to {csv_path}")
        
        return csv_paths
