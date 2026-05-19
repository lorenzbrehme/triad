
import os
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

from langchain_postgres import PGVector
import bs4
from langchain import hub
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
import time
from langchain_huggingface import HuggingFaceEmbeddings
# Load .env file and override existing environment variables
load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# llm = init_chat_model("gpt-5-nano", model_provider="openai")
# llm = init_chat_model("gemma-4-31b-it", model_provider="google_genai")
llm = init_chat_model("gemini-2.5-flash-lite", model_provider="google_genai")



# embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )
# embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store = PGVector(
    embeddings=embeddings,
    collection_name="my_docs",
    # connection="postgresql+psycopg://postgres:password@localhost:5432/hotpot",
    # connection="postgresql+psycopg://postgres:password@localhost:5432/musique",
    # connection="postgresql+psycopg://postgres:password@localhost:5432/hotpot_sentence_transformer",
    # connection="postgresql+psycopg://postgres:password@localhost:5432/musique_sentence_transformer",
    # connection="postgresql+psycopg://postgres:password@localhost:5432/musique_embedding_02"
    # connection="postgresql+psycopg://postgres:password@localhost:5432/hotpot_embedding_02"
    # connection="postgresql+psycopg://postgres:password@localhost:5432/musique_embedding_01"
    connection="postgresql+psycopg://postgres:password@localhost:5432/hotpot_embedding_01"
)





prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer the question based on the provided context and the previous chat history. If the context does not contain the answer, state that you cannot find the answer."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


def llm_invoke_with_retry(llm, messages, max_retries=3, wait_time=60):
    """Invoke LLM with retry logic for rate limiting"""
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            if '429' in str(e):
                if attempt < max_retries - 1:
                    print(f"Rate limit exceeded. Waiting for 60 seconds before evaluating next batch")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Max retries exceeded. Rate limit still active.")
                    raise e
                # try again
            elif '503' in str(e):
                if attempt < max_retries - 1:
                    print(f"Service Unavailable. Waiting for 60 seconds before evaluating next batch") 
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Max retries exceeded. Rate limit still active.")
                    raise e
                # try again
            else:
                # For non-rate-limit errors, don't retry
                raise e
    return None


# Define application steps
def retrieve(state: State):
    question = state["question"]

    retrieved_docs = vector_store.similarity_search(question, k=5)
    # retrieved_docs = vector_store.similarity_search(question, k=3)
    

    return {"context": retrieved_docs, "question": question}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages_for_qa = prompt.invoke({
        "context": docs_content,
        "question": state["question"] 
    })
    response = llm_invoke_with_retry(llm, messages_for_qa)
    return {"answer": response.content}

def get_rag_graph():
    # Compile application and test
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    return graph
