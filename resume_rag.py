import json
import os
import getpass
from pathlib import Path
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes
from langchain_ibm import WatsonxEmbeddings, WatsonxLLM
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from transformers import AutoTokenizer

DEFAULT_CREDENTIALS_PATH = Path("/Users/sanikachavan/Desktop/Resume_Extractor/apikey.json")
DEFAULT_PROJECT_ID = "ee41bd3b-fe85-4c82-979b-98336f24c8f5"
MAX_TOKENS_PER_CHUNK = 500

def _load_credentials_file(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Warning: Could not read credentials file at {path}: {exc}")
        return {}

def setup_credentials():
    config = _load_credentials_file(DEFAULT_CREDENTIALS_PATH)
    api_key = (os.environ.get("WML_APIKEY") or config.get("apikey", "")).strip()
    project_id = (os.environ.get("PROJECT_ID") or config.get("project_id", "")).strip()
    credentials_source = None
    project_source = None
    if api_key:
        credentials_source = ("environment variable" if os.environ.get("WML_APIKEY") else f"file {DEFAULT_CREDENTIALS_PATH}")
    else:
        api_key = getpass.getpass("Please enter your WML api key (hit enter): ").strip()
        credentials_source = "interactive prompt"
    if not project_id:
        if DEFAULT_PROJECT_ID:
            project_id = DEFAULT_PROJECT_ID
            project_source = "default value in script"
        else:
            project_id = input("Please enter your project_id (hit enter): ").strip()
            project_source = "interactive prompt"
    else:
        project_source = ("environment variable" if os.environ.get("PROJECT_ID") else f"file {DEFAULT_CREDENTIALS_PATH}")
    if not is_project_id_associated(project_id):
        print("Error: Project ID is not associated with a WML instance.")
        return None, None
    credentials = {"url": "https://us-south.ml.cloud.ibm.com", "apikey": api_key}
    print("\nLoaded IBM watsonx.ai credentials:")
    print(f"  • API key source: {credentials_source}")
    print(f"  • Project ID source: {project_source}")
    return credentials, project_id

def is_project_id_associated(project_id: str) -> bool:
    return True

def load_resume_pdf(pdf_path: str) -> list:
    print(f"Loading resume from: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from the resume")
    doc_id = 0
    for doc in documents:
        doc.page_content = " ".join(doc.page_content.split())
        doc.metadata["id"] = doc_id
        doc.metadata["source"] = pdf_path
        doc.metadata["page"] = doc.metadata.get("page", doc_id)
        doc_id += 1
    if documents:
        print(f"\nSample document (first 200 chars):")
        print(documents[0].page_content[:200] + "...")
        print(f"\nDocument metadata: {documents[0].metadata}")
    return documents

def split_text_into_chunks(documents: list, max_tokens: int = MAX_TOKENS_PER_CHUNK) -> list:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    all_chunks = []
    for doc in documents:
        tokens = tokenizer.encode(doc.page_content)
        for i in range(0, len(tokens), max_tokens):
            chunk_ids = tokens[i:i+max_tokens]
            chunk_text = tokenizer.decode(chunk_ids)
            new_metadata = dict(doc.metadata)
            new_metadata["chunk_index"] = i // max_tokens
            all_chunks.append(Document(page_content=chunk_text, metadata=new_metadata))
    return all_chunks

def create_vector_store(documents: list, embeddings, collection_name: str = "resume_rag") -> Chroma:
    print(f"\nCreating vector store with collection name: {collection_name}...")
    chunked_docs = split_text_into_chunks(documents)
    vectorstore = Chroma.from_documents(documents=chunked_docs, embedding=embeddings, collection_name=collection_name)
    print("Vector store created successfully!")
    return vectorstore

def setup_embeddings(credentials: dict, project_id: str, model_id: str = "ibm/slate-125m-english-rtrvr-v2") -> WatsonxEmbeddings:
    print("\nSetting up embeddings model...")
    embeddings = WatsonxEmbeddings(model_id=model_id, url=credentials["url"], apikey=credentials["apikey"], project_id=project_id)
    print("Embeddings model ready!")
    return embeddings

def setup_llm(credentials: dict, project_id: str, model_id: str = "ibm/granite-3-8b-instruct") -> WatsonxLLM:
    print(f"\nSetting up LLM: {model_id}...")
    parameters = {
        GenParams.DECODING_METHOD: 'greedy',
        GenParams.TEMPERATURE: 0.2,
        GenParams.TOP_P: 0.9,
        GenParams.TOP_K: 50,
        GenParams.MIN_NEW_TOKENS: 10,
        GenParams.MAX_NEW_TOKENS: 512,
        GenParams.REPETITION_PENALTY: 1.2,
        GenParams.STOP_SEQUENCES: ['\n\n'],
        GenParams.RETURN_OPTIONS: {
            'input_tokens': True,
            'generated_tokens': True,
            'token_logprobs': True,
            'token_ranks': True,
        }
    }
    llm = WatsonxLLM(model_id=model_id, url=credentials["url"], apikey=credentials["apikey"], project_id=project_id, params=parameters)
    print("LLM ready!")
    return llm

def create_rag_chain(vectorstore: Chroma, llm: WatsonxLLM):
    print("\nCreating RAG chain...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    template = """You are a helpful assistant that answers questions about a resume based on the provided context.

Context from the resume:
{context}

Question: {question}

Answer the question based only on the context provided. If the information is not in the context, say "I don't have that information in the resume." Be concise and accurate.

Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])
    chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    print("RAG chain created successfully!")
    return chain

def test_vector_store(vectorstore: Chroma, test_query: str = "What are the key skills?") -> None:
    print(f"\nTesting vector store with query: '{test_query}'")
    print("-" * 60)
    search_results = vectorstore.similarity_search_with_score(test_query, k=3)
    for i, (doc, score) in enumerate(search_results, 1):
        print(f"\nResult {i} (Score: {score:.4f}):")
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Source: {doc.metadata.get('source', 'N/A')}")
        print(f"Page: {doc.metadata.get('page', 'N/A')}")

def main():
    print("=" * 60)
    print("Resume RAG System - LangChain with watsonx.ai")
    print("=" * 60)
    credentials, project_id = setup_credentials()
    if not credentials or not project_id:
        print("Cannot continue without valid credentials and project ID.")
        return
    pdf_path = input("\nPlease enter the path to your resume PDF file: ").strip()
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
        return
    documents = load_resume_pdf(pdf_path)
    if not documents:
        print("Error: No content extracted from PDF")
        return
    embeddings = setup_embeddings(credentials, project_id)
    vectorstore = create_vector_store(documents, embeddings)
    test_vector_store(vectorstore)
    llm = setup_llm(credentials, project_id)
    chain = create_rag_chain(vectorstore, llm)
    print("\n" + "=" * 60)
    print("Resume Q&A System Ready!")
    print("=" * 60)
    print("Type 'quit' to exit.\n")
    while True:
        question = input("Your question: ").strip()
        if question.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        if not question:
            continue
        try:
            answer = chain.invoke(question)
            print(f"\nAnswer: {answer}\n" + "-" * 60)
        except Exception as exc:
            print(f"Error while answering: {exc}\n")

if __name__ == "__main__":
    main()
