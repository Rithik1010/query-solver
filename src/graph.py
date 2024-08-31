from langgraph.graph import END, StateGraph
from typing import TypedDict, List

from langchain import hub
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_openai import OpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = OpenAI(
    temperature=0.2,
    openai_api_key=OPENAI_API_KEY,
    model_name="gpt-3.5-turbo-instruct",
)


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
    """

    query: str
    context_document: List[Document]
    response: any
    run_web_search: bool = False
    response_source: str


### Nodes ###
def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    local = state_dict["local"]
    documents = retriever.get_relevant_documents(question)
    return {"keys": {"documents": documents, "local": local, "question": question}}


def generate(state: GraphState) -> GraphState:
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains generation
    """
    print("---GENERATE---")
    question = state["query"]
    documents = state["context_document"]

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # Post-processing
    # def format_docs(docs):
    #     return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {**state, "response": generation}


def grade_documents(state: GraphState) -> GraphState:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with relevant documents
    """

    question = state["query"]
    documents = state["context_document"]

    print(
        "---CHECK RELEVANCE---\n Query: {}, Document: {}".format(
            question, len(documents)
        )
    )

    prompt = PromptTemplate(
        template="""
        You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keywords related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' and also score_relavance number between 0-1 that indicate how much document is relevant to the question. \n
        Provide the output as a JSON with a keys 'score' and 'score_relavance' and no premable or explaination.
        """,
        input_variables=["question", "context"],
    )

    chain = prompt | llm | JsonOutputParser()

    # Score
    filtered_docs = []

    prompt_context = [
        {
            "question": question,
            "context": d[0].page_content,
        }
        for d in documents
    ]

    s1 = chain.batch(prompt_context)

    print(s1)
    for index, val in enumerate(s1):
        if val['score_relavance'] > 0.49:
            filtered_docs.append(documents[index])

    if not len(documents) == 0:
        matching_threashold_reached = (len(filtered_docs) / len(documents)) > 0.59
    else:
        matching_threashold_reached = False

    return {
        **state,
        "context_document": filtered_docs,
        "query": question,
        "run_web_search": not matching_threashold_reached,
        "response_source": "vector"
    }


def transform_query(state: GraphState) -> GraphState:
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["query"]

    # Create a prompt template with format instructions and the query
    prompt = PromptTemplate(
        template="""You are generating questions that is well optimized for retrieval. \n 
        Look at the input and try to reason about the underlying sematic intent / meaning. \n 
        Here is the initial question:
        \n ------- \n
        {question} 
        \n ------- \n
        Provide an improved question without any premable, only respond with the updated question: """,
        input_variables=["question"],
    )

    # Prompt
    chain = prompt | llm | StrOutputParser()
    better_question = chain.invoke({"question": question})

    return {
        **state,
        "query": better_question,
    }


def web_search(state: GraphState) -> GraphState:
    """
    Web search based on the re-phrased question using Tavily API.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Web results appended to documents.
    """

    print("---WEB SEARCH---")
    question = state["query"]
    documents = state["context_document"]

    tool = TavilySearchResults()
    docs = tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)

    return {
        **state,
        "response_source": "web",
        "context_document": documents,
    }


### Edges
def decide_to_generate(state: GraphState) -> str:
    """
    Determines whether to generate an answer or re-generate a question for web search.

    Args:
        state (dict): The current state of the agent, including all keys.

    Returns:
        str: Next node to call
    """

    print("---DECIDE TO GENERATE---")

    if state["run_web_search"] == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: TRANSFORM QUERY and RUN WEB SEARCH---")
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


workflow = StateGraph(GraphState)

# Define the nodes
# retrieve
# workflow.add_node("retrieve", retrieve)

# grade documents
workflow.add_node("grade_documents", grade_documents)

# generatae
workflow.add_node("generate", generate)

# transform_query
workflow.add_node("transform_query", transform_query)

# web search
workflow.add_node("web_search", web_search)

# Build graph
workflow.set_entry_point("grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "web_search")
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()
