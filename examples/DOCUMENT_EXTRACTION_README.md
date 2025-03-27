# Document Extraction and RAG with Chroma DB

This document provides instructions on how to use the document extraction functionality in the MB-RAG package along with examples of using Chroma DB for RAG (Retrieval-Augmented Generation).

## Document Extraction Features

### PDF Extraction

The `PDFExtractor` class in `mb_rag.utils.pdf_extract` provides the following features:

- Extract text and metadata from PDF files
- Support for multiple extraction methods:
  - PyPDF2 (default): Fast and reliable for most PDFs
  - PDFPlumber: Better for PDFs with complex layouts and tables
  - PyMuPDF: Advanced features including image extraction
- Extract tables from PDFs and convert to CSV format
- Batch processing for multiple PDFs

### CSV Extraction

The `CSVExtractor` class in `mb_rag.utils.document_extract` provides the following features:

- Extract data from CSV files with metadata
- Include basic statistics for numeric columns
- Chunk data by rows for better embedding
- Batch processing for multiple CSV files

### PowerPoint Extraction

The `PowerPointExtractor` class in `mb_rag.utils.document_extract` provides the following features:

- Extract text and metadata from PowerPoint (PPT/PPTX) files
- Extract speaker notes
- Include slide titles and numbers
- Option to include or exclude hidden slides
- Batch processing for multiple PowerPoint files

## Installation Requirements

To use the document extraction functionality, you need to install the required dependencies:

```bash
# PDF Extraction
pip install pypdf
pip install pdfplumber  # For PDFPlumber extraction method
pip install pymupdf  # For PyMuPDF extraction method
pip install tabula-py  # For table extraction to CSV

# CSV Extraction
pip install pandas

# PowerPoint Extraction
pip install python-pptx

# For RAG with Chroma DB
pip install chromadb
```

## Basic Usage

### Extracting Text from PDFs

```python
from mb_rag.utils.pdf_extract import PDFExtractor

# Initialize extractor
extractor = PDFExtractor()

# Extract text from a PDF using the default method (PyPDF2)
documents = extractor.extract_pdf("path/to/document.pdf")

# Extract text using PDFPlumber with table extraction
documents = extractor.extract_pdf(
    "path/to/document.pdf",
    extraction_method="pdfplumber",
    extract_tables=True
)

# Extract text using PyMuPDF with image extraction
documents = extractor.extract_pdf(
    "path/to/document.pdf",
    extraction_method="pymupdf",
    extract_images=True
)

# Extract from multiple PDFs
documents = extractor.extract_multiple_pdfs(
    ["path/to/doc1.pdf", "path/to/doc2.pdf"],
    extraction_method="pypdf"
)
```

### Extracting Tables to CSV

```python
from mb_rag.utils.pdf_extract import PDFToCSV

# Initialize converter
converter = PDFToCSV()

# Extract tables from a PDF and save as CSV files
csv_files = converter.convert_pdf_tables_to_csv(
    "path/to/document.pdf",
    output_dir="path/to/output_directory"
)

# Extract tables from specific pages
csv_files = converter.convert_pdf_tables_to_csv(
    "path/to/document.pdf",
    output_dir="path/to/output_directory",
    pages=[1, 3, 5]  # Extract from pages 1, 3, and 5
)
```

### Extracting Data from CSV Files

```python
from mb_rag.utils.document_extract import CSVExtractor

# Initialize extractor
extractor = CSVExtractor()

# Extract data from a CSV file
documents = extractor.extract_csv("path/to/data.csv")

# Extract with statistics and chunking by rows
documents = extractor.extract_csv(
    "path/to/data.csv",
    include_stats=True,
    chunk_by_row=True,
    rows_per_chunk=20
)

# Extract from multiple CSV files
documents = extractor.extract_multiple_csvs(
    ["path/to/data1.csv", "path/to/data2.csv"],
    include_stats=True
)
```

### Extracting Content from PowerPoint Files

```python
from mb_rag.utils.document_extract import PowerPointExtractor

# Initialize extractor
extractor = PowerPointExtractor()

# Extract content from a PowerPoint file
documents = extractor.extract_ppt("path/to/presentation.pptx")

# Extract with specific options
documents = extractor.extract_ppt(
    "path/to/presentation.pptx",
    include_notes=True,
    include_hidden_slides=False
)

# Extract from multiple PowerPoint files
documents = extractor.extract_multiple_ppts(
    ["path/to/pres1.pptx", "path/to/pres2.pptx"],
    include_notes=True
)
```

## RAG with Chroma DB Examples

The examples directory contains several example scripts that demonstrate how to use the document extraction functionality with Chroma DB for RAG:

### 1. PDF Text Extraction and RAG

The `pdf_chroma_example.py` script demonstrates:
- Extracting text from PDF files
- Generating embeddings from the extracted text
- Storing the embeddings in a Chroma vector database
- Performing semantic search queries on the embedded content
- Interactive conversation with RAG

To run the example:

1. Create a directory called `pdf_files` and add your PDF files to it
2. Run the script:
   ```bash
   python examples/pdf_chroma_example.py
   ```

### 2. PDF Table Extraction to CSV and RAG

The `pdf_csv_chroma_example.py` script demonstrates:
- Extracting tables from PDF files and converting them to CSV format
- Processing the CSV data
- Generating embeddings from the CSV data
- Storing the embeddings in a Chroma vector database
- Performing semantic search queries on the embedded content
- Interactive conversation with RAG

To run the example:

1. Create a directory called `pdf_files` and add your PDF files with tables to it
2. Run the script:
   ```bash
   python examples/pdf_csv_chroma_example.py
   ```

### 3. CSV and PowerPoint Extraction with RAG

The `document_chroma_example.py` script demonstrates:
- Extracting data from CSV files using the CSVExtractor
- Extracting content from PowerPoint files using the PowerPointExtractor
- Generating embeddings from the extracted content
- Storing the embeddings in a Chroma vector database
- Performing semantic search queries on the embedded content
- Interactive conversation with RAG

To run the example:

1. Create directories called `csv_files` and `ppt_files`
2. Add your CSV files to the `csv_files` directory
3. Add your PowerPoint files to the `ppt_files` directory
4. Run the script:
   ```bash
   python examples/document_chroma_example.py
   ```

## Customizing the Examples

You can customize the examples by modifying the following variables at the top of each script:

```python
# PDF Example Configuration
PDF_DIR = "./pdf_files"  # Directory containing PDF files
EMBEDDINGS_DIR = "./pdf_embeddings"  # Directory to save embeddings
CHUNK_SIZE = 500  # Size of text chunks for embeddings
CHUNK_OVERLAP = 50  # Overlap between chunks

# PDF CSV Example Configuration
PDF_DIR = "./pdf_files"  # Directory containing PDF files
CSV_DIR = "./csv_files"  # Directory to save CSV files
EMBEDDINGS_DIR = "./csv_embeddings"  # Directory to save embeddings

# Document Example Configuration
CSV_DIR = "./csv_files"  # Directory containing CSV files
PPT_DIR = "./ppt_files"  # Directory containing PowerPoint files
EMBEDDINGS_DIR = "./document_embeddings"  # Directory to save embeddings
```

## Using Different Embedding Models

By default, the examples use OpenAI's embedding model. You can change this by modifying the `model` and `model_type` parameters in the `embedding_generator` initialization:

```python
# Initialize embedding generator with a different model
em_gen = embedding_generator(
    model="ollama",  # Options: "openai", "ollama", "google", "anthropic"
    model_type="llama3",  # Model-specific identifier
    vector_store_type="chroma"
)
```

## Troubleshooting

### Common Issues

1. **Missing dependencies**: Make sure you have installed all required packages for the extraction method you're using.

2. **Java requirement for tabula-py**: The `tabula-py` package requires Java to be installed on your system. If you encounter errors with table extraction, ensure Java is installed and properly configured.

3. **PDF extraction errors**: Some PDFs may have security features or complex structures that make extraction difficult. Try different extraction methods if one fails.

4. **Memory issues with large files**: Processing large files can be memory-intensive. Consider extracting specific pages/rows or reducing the chunk size for embeddings.

5. **PowerPoint format issues**: The `python-pptx` package may have limitations with certain PowerPoint features or older file formats. If you encounter issues, try saving the presentation in the newer .pptx format.

### Getting Help

If you encounter issues not covered here, please check the main MB-RAG documentation or open an issue on the project repository.
