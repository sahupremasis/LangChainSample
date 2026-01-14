from dotenv import load_dotenv
load_dotenv()

from operator import itemgetter

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
    RunnableBranch,
)

def get_documents_from_web(url: str):
    loader = WebBaseLoader(url)
    docs = loader.load()
    print("Loaded docs:", len(docs))
    print("First chunk preview:\n", docs[0].page_content[:500])

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
    return splitter.split_documents(docs)

def create_db(docs):
    embedding = OpenAIEmbeddings()
    return FAISS.from_documents(docs, embedding=embedding)

def _format_docs(docs) -> str:
    # "Stuff" documents into a single context string
    return "\n\n".join(
        f"[Source {i+1}]\n{d.page_content}"
        for i, d in enumerate(docs)
    )

# ---------------------------
# Build chain (LangChain-only LCEL)
# ---------------------------

def create_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.4)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 1) Prompt to generate a better search query when we have chat history
    query_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human", "Create a concise search query to retrieve relevant context for the user's last question."),
    ])

    query_chain = query_prompt | llm | StrOutputParser()

    # 2) History-aware retriever (no langchain-classic)
    # If chat_history is empty -> use input directly
    # Else -> use LLM to rewrite/generate a search query, then retrieve with it
    history_aware_retriever = RunnableBranch(
        # If chat history exists → generate query → retrieve
        (
            lambda x: bool(x.get("chat_history")),
            query_chain | retriever
        ),
        # Else → retrieve using the raw user input
        itemgetter("input") | retriever
    )

    # 3) Answer prompt uses {context} + chat_history + input
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based only on the context below.\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    # 4) Combine: retrieve docs -> format to context -> answer
    chain = (
        RunnablePassthrough.assign(docs=history_aware_retriever)
        .assign(context=RunnableLambda(lambda x: _format_docs(x["docs"])))
        | answer_prompt
        | llm
        | StrOutputParser()
    )

    return chain

def process_chat(chain, question, chat_history):
    return chain.invoke({"input": question, "chat_history": chat_history})

# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    docs = get_documents_from_web("https://python.langchain.com/docs/expression_language/")
    vectorstore = create_db(docs)
    chain = create_chain(vectorstore)

    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        response = process_chat(chain, user_input, chat_history)

        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))

        print("Assistant:", response)

            #
            # ┌───────────────────────────────────────────────────────────────────┐
            # │                           START / SETUP                            │
            # └───────────────────────────────────────────────────────────────────┘
            #                 │
            #                 v
            # ┌───────────────────────────────────────────────────────────────────┐
            # │ 1) Load Web Page                                                    │
            # │    WebBaseLoader(url) → docs (raw Documents)                        │
            # └───────────────────────────────────────────────────────────────────┘
            #                 │
            #                 v
            # ┌───────────────────────────────────────────────────────────────────┐
            # │ 2) Split into Chunks                                                │
            # │    RecursiveCharacterTextSplitter → split_docs (many chunks)         │
            # └───────────────────────────────────────────────────────────────────┘
            #                 │
            #                 v
            # ┌───────────────────────────────────────────────────────────────────┐
            # │ 3) Create Vector DB (FAISS)                                         │
            # │    OpenAIEmbeddings: chunk → embedding vector                        │
            # │    FAISS.from_documents(split_docs, embeddings)                      │
            # └───────────────────────────────────────────────────────────────────┘
            #
            #
            # ┌───────────────────────────────────────────────────────────────────┐
            # │                           CHAT LOOP                                 │
            # └───────────────────────────────────────────────────────────────────┘
            #                 │
            #                 v
            # ┌───────────────────────────────────────────────────────────────────┐
            # │ User asks a question                                                │
            # │ input = "what is LangGraph?"                                             │
            # │ chat_history = [HumanMessage, AIMessage, ...]                        │
            # └───────────────────────────────────────────────────────────────────┘
            #                 │
            #                 v
            # ┌─────────────────────────── Decision ───────────────────────────────┐
            # │ 4) Do we have chat_history?                                         │
            # └───────────────────────────────────────────────────────────────────┘
            #         │ Yes                                             │ No
            #         v                                                 v
            # ┌───────────────────────────────────────────┐   ┌────────────────────┐
            # │ 5A) Generate a search query with LLM       │   │ 5B) Use raw input   │
            # │ query_prompt(chat_history + input)         │   │ query = input       │
            # │   → LLM → "LangGraph  Expression..."   │   └────────────────────┘
            # └───────────────────────────────────────────┘                │
            #         │                                                    │
            #         └───────────────────────────────┬────────────────────┘
            #                                         v
            # ┌───────────────────────────────────────────────────────────────────┐
            # │ 6) Retrieve relevant chunks                                         │
            # │ retriever(query) → top-k matching Documents (chunks)                │
            # │ (FAISS similarity search using embeddings)                           │
            # └───────────────────────────────────────────────────────────────────┘
            #                 │
            #                 v
            # ┌───────────────────────────────────────────────────────────────────┐
            # │ 7) "Stuff" docs into a single context string                         │
            # │ context = format_docs(docs)                                          │
            # │ e.g. "[Source 1] ...\n\n[Source 2] ..."                              │
            # └───────────────────────────────────────────────────────────────────┘
            #                 │
            #                 v
            # ┌───────────────────────────────────────────────────────────────────┐
            # │ 8) Build final answer prompt                                         │
            # │ system: "Answer based on context: {context}"                         │
            # │ + chat_history                                                      │
            # │ + human: {input}                                                    │
            # └───────────────────────────────────────────────────────────────────┘
            #                 │
            #                 v
            # ┌───────────────────────────────────────────────────────────────────┐
            # │ 9) Call Chat Model (ChatOpenAI)                                      │
            # │ prompt → LLM → answer text                                           │
            # └───────────────────────────────────────────────────────────────────┘
            #                 │
            #                 v
            # ┌───────────────────────────────────────────────────────────────────┐
            # │ 10) Update chat history                                              │
            # │ chat_history += [HumanMessage(input), AIMessage(answer)]             │
            # └───────────────────────────────────────────────────────────────────┘
            #                 │
            #                 v
            # ┌───────────────────────────────────────────────────────────────────┐
            # │                           LOOP CONTINUES                             │
            # └───────────────────────────────────────────────────────────────────┘
