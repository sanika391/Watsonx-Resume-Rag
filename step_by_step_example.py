import os
import getpass
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes
from langchain_ibm import WatsonxEmbeddings, WatsonxLLM
from langchain.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader

print("=" * 70)
print("STEP-BY-STEP RESUME RAG SYSTEM")
print("=" * 70)

print("\n[STEP 1] Setting up credentials...")
credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": getpass.getpass("Please enter your WML api key: ")
}

try:
    project_id = os.environ["PROJECT_ID"]
    print(f"Using PROJECT_ID from environment: {project_id}")
except KeyError:
    project_id = input("Please enter your project_id: ")
    print(f"Using PROJECT_ID: {project_id}")

print("\n[STEP 2] Loading PDF resume...")
pdf_path = input("Enter the path to your resume PDF: ").strip()

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

loader = PyPDFLoader(pdf_path)
documents = loader.load()
print(f"✓ Loaded {len(documents)} pages from PDF")

print(f"\nSample content (first 300 chars):")
print(documents[0].page_content[:300] + "...")

print("\n[STEP 3] Cleaning documents...")
doc_id = 0
for doc in documents:
    doc.page_content = " ".join(doc.page_content.split())
    doc.metadata["id"] = doc_id
    doc.metadata["source"] = pdf_path
    doc.metadata["page"] = doc.metadata.get("page", doc_id)
    doc_id += 1

print(f"✓ Cleaned {len(documents)} documents")
print(f"Sample metadata: {documents[0].metadata}")

MAX_EMBEDDING_TOKENS = 512
AVERAGE_CHARS_PER_TOKEN = 4


def enforce_embedding_limit(
    docs: list[Document],
    max_tokens: int = MAX_EMBEDDING_TOKENS,
    chars_per_token: int = AVERAGE_CHARS_PER_TOKEN,
) -> list[Document]:
    max_chars = max_tokens * chars_per_token
    safe_docs: list[Document] = []

    for doc in docs:
        if len(doc.page_content) <= max_chars:
            safe_docs.append(doc)
            continue

        print(
            f"Page {doc.metadata.get('page')} is long; "
            "splitting it to satisfy the embedding limit."
        )
        for segment_idx, start in enumerate(range(0, len(doc.page_content), max_chars)):
            segment_content = doc.page_content[start : start + max_chars]
            metadata = dict(doc.metadata)
            metadata["segment"] = segment_idx
            safe_docs.append(Document(page_content=segment_content, metadata=metadata))

    return safe_docs


documents = enforce_embedding_limit(documents)
print(f"✓ Prepared {len(documents)} embedding-safe documents")

print("\n[STEP 5] Setting up embeddings model...")
embeddings = WatsonxEmbeddings(
    model_id=EmbeddingTypes.IBM_SLATE_30M_ENG.value,
    url=credentials["url"],
    apikey=credentials["apikey"],
    project_id=project_id,
)
print("✓ Embeddings model ready (IBM Slate 30M)")

print("\n[STEP 6] Creating vector store...")
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="resume_rag_collection"
)
print("✓ Vector store created successfully!")

print("\n[TEST] Testing vector store retrieval...")
test_query = "What are the key skills?"
search_results = vectorstore.similarity_search_with_score(test_query, k=2)
print(f"Query: '{test_query}'")
for i, (doc, score) in enumerate(search_results, 1):
    print(f"  Result {i} (score: {score:.4f}): {doc.page_content[:150]}...")

print("\n[STEP 7] Setting up retriever...")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("✓ Retriever ready (will retrieve top 3 most relevant pages)")

print("\n[STEP 8] Setting up language model...")
model_id = "ibm/granite-3-8b-instruct"

parameters = {
    GenParams.DECODING_METHOD: 'greedy',
    GenParams.TEMPERATURE: 0.2,
    GenParams.TOP_P: 0.9,
    GenParams.TOP_K: 50,
    GenParams.MIN_NEW_TOKENS: 10,
    GenParams.MAX_NEW_TOKENS: 512,
    GenParams.REPETITION_PENALTY: 1.2,
}

llm = WatsonxLLM(
    model_id=model_id,
    url=credentials.get("url"),
    apikey=credentials.get("apikey"),
    project_id=project_id,
    params=parameters
)
print(f"✓ LLM ready ({model_id})")

print("\n[STEP 9] Creating prompt template...")
template = """You are a helpful assistant that answers questions about a resume based on the provided context.

Context from the resume:
{context}

Question: {question}

Answer the question based only on the context provided. If the information is not in the context, say "I don't have that information in the resume." Be concise and accurate.

Answer:"""

prompt = ChatPromptTemplate.from_template(template)
print("✓ Prompt template created")

print("\n[STEP 10] Creating document formatter...")
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])
print("✓ Formatter function ready")

print("\n[STEP 11] Creating RAG chain...")
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print("✓ RAG chain created successfully!")

print("\n" + "=" * 70)
print("SYSTEM READY! Testing with sample questions...")
print("=" * 70)

print("\n[TEST 1]")
question1 = "What are the key skills mentioned in the resume?"
print(f"Question: {question1}")
answer1 = chain.invoke(question1)
print(f"Answer: {answer1}")

print("\n[TEST 2]")
question2 = "What is the candidate's work experience?"
print(f"Question: {question2}")
answer2 = chain.invoke(question2)
print(f"Answer: {answer2}")

print("\n" + "=" * 70)
print("INTERACTIVE MODE")
print("=" * 70)
print("You can now ask questions about the resume.")
print("Type 'quit' or 'exit' to stop.\n")

while True:
    question = input("Your question: ").strip()
    
    if question.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        break
    
    if not question:
        continue
    
    try:
        print("\nProcessing...")
        answer = chain.invoke(question)
        print(f"\nAnswer: {answer}\n")
        print("-" * 70)
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please try again.\n")

print("\n" + "=" * 70)
print("Session ended. Thank you!")
print("=" * 70)

