from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
# Updated import to use the standalone package you have installed
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
# Added imports for manual chain construction (LCEL)
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

# Retrieve Data
def get_docs():
    loader = WebBaseLoader('https://python.langchain.com/docs/expression_language/')
    docs = loader.load()

## This is a specific strategy for splitting text.
    # It tries to split text smartly by looking for natural separators in order
    # (like double newlines for paragraphs, then single newlines, then spaces).
    # chunk_size=200: This sets the target size for each piece of text to 200 characters.
    # The splitter will try to keep chunks around this length.
    # chunk_overlap=20:
    # This creates a "buffer" of 20 characters that are repeated between two consecutive chunks.
    # For example, the end of Chunk 1 will be the same as the beginning of Chunk 2.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )

    splitDocs = text_splitter.split_documents(docs)

    return splitDocs


## It takes your list of split text chunks (docs).
#  uses the embedding model to convert all those chunks into vectors.
#  stores these vectors in a way that allows for extremely fast "nearest neighbor" searches.
def create_vector_store(docs):
    embedding = OpenAIEmbeddings()
    vectorStore = FAISS.from_documents(docs, embedding=embedding)  ## Facebook AI Similarity Search
    return vectorStore


def create_chain(vectorStore):
    model = ChatOpenAI(
        temperature=0.4,
        model='gpt-4o-mini'
    )

    prompt = ChatPromptTemplate.from_template("""
    Answer the user's question.
    Context: {context}
    Question: {input}
    """)

    retriever = vectorStore.as_retriever()

    # Define a helper function to format retrieved documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Construct the chain using LCEL instead of convenience functions
    # This pipeline:
    # 1. Takes the input and retrieves documents ("context")
    # 2. Passes input and context to the prompt
    # 3. Sends prompt to model
    # 4. Parses output to string
    chain = (
        {
            "context": itemgetter("input") | retriever | format_docs,
            "input": itemgetter("input")
        }
        | prompt
        | model
        | StrOutputParser()
    )

    return chain


docs = get_docs()
vectorStore = create_vector_store(docs)
chain = create_chain(vectorStore)

response = chain.invoke({
    "input": "What is LangGraph?",
})

print(response)