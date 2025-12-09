# Resume RAG System with LangChain and watsonx.ai

This project implements a Retrieval Augmented Generation (RAG) system to extract information from PDF resumes and answer questions about them using LangChain and IBM watsonx.ai.

## Overview

The system:
1. **Extracts text** from a PDF resume
2. **Creates a vector store** from the extracted text
3. **Retrieves relevant information** based on questions
4. **Generates answers** using an LLM with the retrieved context

## Prerequisites

1. **IBM Cloud Account**: You need an IBM Cloud account to use watsonx.ai
2. **watsonx.ai Project**: Create a project in watsonx.ai
3. **watsonx.ai Runtime Service**: Create a Runtime service instance (Lite plan is free)
4. **API Key**: Generate an API key from the Runtime service
5. **Project ID**: Your watsonx.ai project ID

## Installation

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Download NLTK Data

The script will automatically download required NLTK data, but you can also do it manually:

```python
import nltk
nltk.download('averaged_perceptron_tagger_eng')
```

## Setup Instructions

### Step 1: Get Your Credentials

1. Log in to [watsonx.ai](https://dataplatform.cloud.ibm.com/)
2. Create a watsonx.ai project (if you haven't already)
3. Create a watsonx.ai Runtime service instance:
   - Go to the IBM Cloud catalog
   - Search for "watsonx.ai Runtime"
   - Select the Lite (free) plan
   - Create the service
4. Generate an API Key:
   - Go to your Runtime service instance
   - Navigate to "Service credentials"
   - Create a new credential
   - Copy the API key
5. Get your Project ID:
   - In your watsonx.ai project
   - The Project ID is shown in the project details

### Step 2: Set Environment Variables (Optional)

You can set the PROJECT_ID as an environment variable:

```bash
export PROJECT_ID="your-project-id-here"
```

Or you can enter it when prompted by the script.

### Step 3: Automatic Credential Loading

The script looks for credentials in this order:

1. Environment variables `WML_APIKEY` and `PROJECT_ID`
2. JSON file at `/Users/sanikachavan/Downloads/apikey.json` (expects an `apikey` field and optionally `project_id`)
3. Interactive prompts (only used if the first two options are missing)

This lets you keep secrets out of the repository while still running the script without retyping them every time.

## Usage

### Basic Usage

1. Place your resume PDF file in a location accessible to the script
2. Run the script:

```bash
python resume_rag.py
```

3. Provide the path to your resume PDF file when prompted (credentials are loaded automatically unless missing)

4. The system will:
   - Load and process your resume
   - Create a vector store
   - Test the system with a sample query
   - Start an interactive Q&A session

### Example Questions

Once the system is ready, you can ask questions like:

- "What are the key skills mentioned in the resume?"
- "What is the candidate's work experience?"
- "What education does the candidate have?"
- "What programming languages are mentioned?"
- "What is the candidate's current position?"
- "What certifications does the candidate have?"

Type `quit` or `exit` to stop the session.

## How It Works

### Step-by-Step Process

1. **PDF Loading**: Uses `PyPDFLoader` from LangChain to extract text from PDF pages

2. **Text Cleaning**: Removes excessive whitespace and newlines from the extracted text

3. **Document Storage**: Keeps every PDF page intact as a LangChain `Document` (and only splits a page if the embedding model's 512-token limit would be exceeded)

4. **Embeddings**: Uses IBM's `slate-125m-english-rtrvr-v2` embedding model to create vector embeddings for each page

5. **Vector Store**: Stores embeddings in a Chroma vector database for fast similarity search

6. **Retrieval**: When you ask a question, the system:
   - Converts your question to an embedding
   - Finds the most similar resume pages in the vector store
   - Retrieves the top 3 most relevant pages

7. **Generation**: The LLM (IBM Granite) generates an answer based on:
   - The retrieved page-level context
   - Your question
   - A carefully crafted prompt template

## Code Structure

```
Resume_Extractor/
├── resume_rag.py          # Main script with all functionality
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Functions Explained

### `load_resume_pdf(pdf_path)`
- Loads PDF using PyPDFLoader
- Cleans text by removing extra whitespace
- Adds metadata (ID, source, page number)

### `enforce_embedding_limit(documents, max_tokens, chars_per_token)`
- Ensures each resume page stays within the embedding model's token limit
- Automatically splits a page into sequential slices only when the model requires it

### `create_vector_store(documents, embeddings, collection_name)`
- Creates a Chroma vector store
- Stores resume pages with their embeddings
- Enables fast similarity search

### `setup_embeddings(credentials, project_id)`
- Initializes IBM Slate embedding model
- Used to convert text to vector representations

### `setup_llm(credentials, project_id, model_id)`
- Sets up IBM Granite LLM
- Configures model parameters (temperature, top_p, etc.)

### `create_rag_chain(vectorstore, llm)`
- Creates the complete RAG pipeline
- Combines retrieval and generation
- Returns a chain that can answer questions

## Customization

### Change Number of Retrieved Documents

In `create_rag_chain`, modify the `search_kwargs`:

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 instead of 3
```

### Modify LLM Parameters

In `setup_llm`, adjust the parameters dictionary:

```python
parameters = {
    GenParams.TEMPERATURE: 0.5,  # Higher = more creative
    GenParams.MAX_NEW_TOKENS: 1024,  # Longer responses
    # ... other parameters
}
```

### Change the Prompt Template

Modify the `template` variable in `create_rag_chain` to change how the LLM responds:

```python
template = """Your custom prompt here...
{context}
Question: {question}
Answer:"""
```

## Troubleshooting

### Issue: "File not found"
- Make sure the PDF path is correct
- Use absolute path if relative path doesn't work
- Check file permissions

### Issue: "API key invalid"
- Verify your API key is correct
- Make sure the Runtime service is active
- Check that the service is associated with your project

### Issue: "Project ID not found"
- Verify your project ID
- Make sure the Runtime service is associated with the project

### Issue: "No content extracted from PDF"
- The PDF might be image-based (scanned)
- Try using OCR tools first
- Check if the PDF is password-protected

### Issue: Poor retrieval results
- Increase the number of retrieved documents (k parameter)
- Check if the resume text was extracted correctly
- Confirm the PDF text is being fully captured (no missing pages)

## Next Steps

- Add support for multiple resume formats (DOCX, TXT)
- Implement persistent vector store (save to disk)
- Add web interface for easier interaction
- Support batch question processing
- Add resume analysis features (skill extraction, experience summary, etc.)

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [IBM watsonx.ai Documentation](https://www.ibm.com/products/watsonx-ai)
- [Chroma Vector Database](https://www.trychroma.com/)

## License

This project is for educational purposes.

