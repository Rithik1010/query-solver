import os
from dotenv import load_dotenv

import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain.memory import ConversationBufferMemory


from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    WebBaseLoader,
)
from langchain_community.vectorstores.qdrant import Qdrant
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain

from langchain.docstore.document import Document

from langchain_openai import OpenAIEmbeddings, OpenAI

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

import qdrant_client
from qdrant_client import QdrantClient
from qdrant_client.http import models


from graph import app as GraphApp, GraphState

from langchain_core.prompts.prompt import PromptTemplate

prompt_template = """
Your name is Novatr AI.
You've been created by a team of Engineers at Novatr,
You are a friendly and knowledgeable AI assistant of Novatr, a cutting-edge educational platform dedicated to the Architecture, Engineering, and Construction (AEC) industry. Equipped with extensive resources and content from Novatr, your mission is to provide invaluable assistance to users. 
Use the following pieces of context to answer the question into markdown format at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""


QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


load_dotenv()


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")

q_URL = os.getenv("QDRANT_URL")
q_API_KEY = os.getenv("QDRANT_API_KEY")

qdrant_client = QdrantClient(
    url=q_URL, port=None, api_key=q_API_KEY, timeout=10000)

doc_store = Qdrant(
    client=qdrant_client,
    embeddings=embeddings,
    collection_name="website_dump",
)

MyOpenAI = OpenAI(
    temperature=0.2,
    openai_api_key=OPENAI_API_KEY,
    model_name="gpt-3.5-turbo-instruct",
)

chat_history = {}


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# tools = load_tools(["serpapi", "llm-math"], llm=MyOpenAI)

# agent = initialize_agent(
#     tools, MyOpenAI, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True)


# Delete all existing schema
async def delete_schema():
    # client = await initialize_weaviate()
    vectorstore._client.schema.delete_all()
    return "Schema was deleted."

# Load pdfs from a folder


async def load_pdfs(folder_path):
    loader = DirectoryLoader(
        folder_path, glob="./*.pdf", loader_cls=PyPDFLoader)
    return loader


# Load website urls
async def load_website_urls(urls):
    loader = WebBaseLoader(urls)
    return loader


# Upload texts to vectorstore
async def upload_texts_to_vectorstore(loader, group_id):
    if not group_id:
        group_id = "GENERAL"
    texts = await split_documents(loader)
    await add_texts_to_class(texts, group_id)
    return f"{len(texts)} chunks are added."


# Split documents
async def split_documents(loader):
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)

    texts = text_splitter.split_documents(documents)
    return texts


async def split_texts(input_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)

    texts = text_splitter.split_text(input_text)
    return texts


# Add texts as objects to a class
async def add_texts_to_class(texts, group_id):
    text_meta_pair = [
        (text.page_content, {**text.metadata, "group_id": group_id}) for text in texts
    ]
    chunks, meta = list(zip(*text_meta_pair))
    print(meta)
    vectorstore.add_texts(chunks, meta)
    return "Texts are added."


async def get_question_relevance_score(question: str) -> float:
    return 0.0


# Get chat completion for a query
async def get_chat_completion(query, session_key, **kwargs):
    # Log session key access (optional)
    logger.debug(f"Accessing chat history for session key: {session_key}")

    if session_key not in chat_history:
        chat_history[session_key] = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key="answer"
        )
        logger.info(
            f"Created new conversation buffer for session key: {session_key}")

    course_id = kwargs.get("course_id", None)
    module_id = kwargs.get("module_id", None)
    topic_id = kwargs.get("topic_id", None)

    should_clause = []

    if course_id:
        should_clause.append(
            models.FieldCondition(
                key="metadata.course_id",
                match=models.MatchValue(value=course_id),
            )
        )
        logger.debug(f"Added course_id filter: {course_id}")

    if module_id:
        should_clause.append(
            models.FieldCondition(
                key="metadata.module_id",
                match=models.MatchValue(value=module_id),
            )
        )
        logger.debug(f"Added module_id filter: {module_id}")

    if topic_id:
        should_clause.append(
            models.FieldCondition(
                key="metadata.topic_id",
                match=models.MatchValue(value=topic_id),
            )
        )
        logger.debug(f"Added topic_id filter: {topic_id}")

    if len(should_clause) == 0:
        filter = models.Filter()
    else:
        filter = models.Filter(should=should_clause)

    # Log filter details
    logger.info(f"Filtering documents with criteria: {filter}")

    # Filter documents in the "website_dump" collection
    filtered_documents = qdrant_client.scroll(
        collection_name="website_dump",
        scroll_filter=filter,
    )

    # Process filtered documents
    langchain_documents = []
    for i in filtered_documents[0]:
        # Extract relevant information from Qdrant document
        doc = i.payload
        text = doc.get("page_content", "")
        metadata = doc.get("metadata", {})
        # Create Langchain document object
        langchain_documents.append(
            Document(
                page_content=text,
                metadata=metadata,
            )
        )
        logger.debug(f"Extracted document with metadata: {metadata}")

    doc_store = Qdrant.from_documents(
        url=q_URL,
        port=None,
        api_key=q_API_KEY,
        embedding=embeddings,
        collection_name="website_dump",
        documents=langchain_documents,
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=MyOpenAI,
        verbose=True,
        retriever=doc_store.as_retriever(),
        memory=chat_history[session_key],
        return_generated_question=True,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
    )

    # Log query and ConversationalRetrievalChain creation
    logger.info(f"Running ConversationalRetrievalChain with query: {query}")
    logger.debug(f"ConversationalRetrievalChain configuration: {qa.__dict__}")

    result = qa({"question": query})
    c = {
        **result,
        "source_documents": [
            doc.metadata for doc in result.get("source_documents", [])
        ],
        "question_relevance_score": await get_question_relevance_score(query),
    }

    logger.info(f"Chat completion result: {c}")

    return c


async def get_chat_title(query):
    input_chunks = await split_texts(query)
    output_chunks = []
    for chunk in input_chunks:
        # response = MyOpenAI(f"Please create heading for the following question:\n{chunk}?\n")
        response = MyOpenAI(
            f"Generate a simple and short heading that captures the essence of the follwing text: {chunk}")
        summary = response.strip().strip('"')
        output_chunks.append(str(summary))
    out = {
        "query": query,
        "title": " ".join(output_chunks),
    }
    return out


def load_pdf():
    print("Loading PDF")
