# Resume RAG System — LangChain + IBM watsonx.ai

> Extract text from a resume PDF, store full-page documents in a vector store (no chunking), then answer user questions by retrieving relevant pages and calling a watsonx LLM.

## Problem statement

Many teams want a quick Q&A system over resumes (or other PDFs) that:

1. Extracts text from an uploaded PDF,
2. Stores the extracted text in a vector store for retrieval, and
3. Uses an LLM to answer user questions **only** from retrieved context.

Constraints for this repository:

* **No chunking** — each PDF page is stored as a single `Document`.
* **Only LangChain primitives** are used.
* **Only IBM watson / watsonx models** are used for embeddings and LLM inference (no OpenAI, etc.).

## Short solution summary

This project uses `PyPDFLoader` to extract **page-level** text from a resume PDF, injects each page as a single `Document` into a Chroma vector store using `WatsonxEmbeddings`, and answers questions by retrieving the top-`k` pages and passing the concatenated context to a `WatsonxLLM` via a LangChain prompt. No chunking is performed — each page remains intact.

## Architecture (high-level)

![AricheturalDiagram](https://github.com/user-attachments/assets/05d12528-068c-4665-a17a-1115759a9036)

## Files provided (key ones)

* `main.py` — runnable script with the full pipeline (credential setup, PDF load, embeddings, vector store creation, test search, LLM chain loop).
* `README.md` — (this file) usage & explanation.
* `requirements.txt` — Python dependencies (recommended).

## Key design decisions (called out)

* **`PyPDFLoader`**: Produces one LangChain `Document` per page, preserves metadata, and integrates nicely with LangChain.
* **No chunking**: Each page is indexed as a single unit to satisfy the `NO CHUNKING` requirement. (Trade-off: very long pages can risk LLM context limits.)
* **Chroma vector store**: Local, simple, and integrates with LangChain — ideal for experiments and single-repo setups.
* **IBM watsonx.ai**: `WatsonxEmbeddings` and `WatsonxLLM` (e.g., `ibm/slate-125m-english-rtrvr-v2` for embeddings and `ibm/granite-3-8b-instruct` for LLM) are used to comply with the constraint to use only watson/watsonx models. These provide enterprise-grade governance and on-cloud/on-prem options.
* **GPT-2 tokenizer**: Used only to estimate token counts (fast, lightweight, model-agnostic).

## Prerequisites

* Python 3.10+ recommended
* An IBM watsonx.ai instance (API key + optionally `project_id`)
* Local filesystem or environment with sample PDF(s) to test

Install dependencies (example):

```bash
pip install -r requirements.txt
```

`requirements.txt` should include (example):

```
langchain
langchain_community
langchain_ibm
ibm_watson_machine_learning
ibm_watsonx_ai
chromadb
transformers
pypdf
```

> Adjust package names/versions for your environment and IBM SDK compatibility.

## Setup

1. Create an API key for IBM watsonx.ai and save it to a JSON file, or set it as an environment variable `WML_APIKEY` (the code reads both). Example `apikey.json` format:

```json
{ "apikey": "<YOUR_API_KEY>", "project_id": "<OPTIONAL_PROJECT_ID>" }
```

2. Optionally export `PROJECT_ID` if you want to override the default project id.

3. Edit `DEFAULT_CREDENTIALS_PATH` or store your key at the path used in the script.

## How it works (step-by-step)

1. **Load credentials** — `setup_credentials()` reads API key and project id.
2. **Load PDF** — `load_resume_pdf(pdf_path)` uses `PyPDFLoader` to create a list of `Document` objects (one per page). Each document is sanitized and annotated with metadata (`id`, `source`, `page`).
3. **(Optional) Tokenization check** — `split_text_into_chunks()` exists in the code for token-based chunking, but to honor the `NO CHUNKING` requirement you should **bypass** it and index `documents` directly.
4. **Create embeddings** — `setup_embeddings()` creates a `WatsonxEmbeddings` instance.
5. **Create vector store** — `Chroma.from_documents(documents=documents, embedding=embeddings, collection_name=...)`.
6. **Test retrieval** — `test_vector_store()` runs a similarity search (k=3) and prints results with scores and metadata.
7. **Set up LLM** — `setup_llm()` configures `WatsonxLLM` with `GenParams` tuned for low temperature and instruction-following.
8. **Create RAG chain** — `create_rag_chain()` builds a LangChain runnable that: retrieves top-k docs → formats them into a single context string → passes into the Chat Prompt Template → calls the LLM → returns parsed string.
9. **Interactive loop** — in `main()` a simple REPL accepts questions and returns answers.

## How to run (example)

```bash
python main.py
# When prompted, paste the path to your resume PDF, e.g. /path/to/resume.pdf
# Then ask questions like "What are the key skills?"
```

Sample output sequence you can expect:

1. Credentials loaded (source shown).
2. `Loaded N pages from the resume`
3. `Vector store created successfully!`
4. Test similarity search results printed (top-3).
5. Interactive question → answer loop.

## Code highlights & important snippets

**No-chunking vector store creation** (recommended change):

```python
# Use this to index full pages directly
vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, collection_name=collection_name)
```

**RAG prompt template**

```text
You are a helpful assistant that answers questions about a resume based on the provided context.

Context from the resume:
{context}

Question: {question}

Answer the question based only on the context provided. If the information is not in the context, say "I don't have that information in the resume." Be concise and accurate.

Answer:
```

**Watsonx LLM parameters** — tuned for factual answers and low hallucination:

```python
parameters = {
    GenParams.DECODING_METHOD: 'greedy',
    GenParams.TEMPERATURE: 0.2,
    GenParams.TOP_P: 0.9,
    GenParams.TOP_K: 50,
    GenParams.MIN_NEW_TOKENS: 10,
    GenParams.MAX_NEW_TOKENS: 512,
    GenParams.REPETITION_PENALTY: 1.2,
    GenParams.STOP_SEQUENCES: ['\n\n'],
}
```
## Testing & validation

* Test with 1–10 page resumes first.
* Use `test_vector_store()` to inspect retrieved pages and ensure relevance.
* If a page is massive and triggers a context overflow on the LLM, either use a larger-context watsonx model or split that page manually (outside of the enforced "no chunking" rule).

## Security & governance

* Keep API keys secret. Use environment variables or a secrets manager in production.
* watsonx.ai provides enterprise governance and audit features — leverage these for production deployments.

## Limitations & trade-offs

* **No chunking** can cause context overflow if pages are extremely large.
* Chroma is local — for high scale, consider a managed vector DB (Pinecone, Milvus, etc.).
* Accuracy depends on the embedding model selection; test alternative watsonx retriever models for best retrieval quality.

## Next steps / enhancements

* Add an admin UI to upload/manage PDFs and inspect vector store.
* Add a pre-check for page token size and raise a warning or automated split (if allowed).
* Add filtering by metadata (page, source) in retrieval.
* Add automated tests, CI/CD packaging, and containerization.

