# --- Code cell 5 ---
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma

"""
Task (clarified): Load a DOCX file, split its text first by Markdown-like headers and then into
smaller character-based chunks, normalize whitespace in each chunk, and prepare an OpenAI
embedding model to embed those chunks later for vector storage / retrieval.
"""

# 1) Load the DOCX into LangChain Document objects
loader_docx = Docx2txtLoader("Data_Science_Readme.docx")
pages = loader_docx.load()

# 2) Split the first document's text by Markdown-style headers (e.g., "#", "##")
md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "Course Title"), ("##", "Lecture Title")]
)
pages_md_split = md_splitter.split_text(pages[0].page_content)

# 3) Normalize whitespace inside each split document (collapse newlines/tabs/multiple spaces)
for i in range(len(pages_md_split)):
    pages_md_split[i].page_content = " ".join(pages_md_split[i].page_content.split())

# 4) Further split header-based documents into smaller overlapping chunks using "." as a separator
char_splitter = CharacterTextSplitter(
    separator=".",
    chunk_size=500,
    chunk_overlap=50,
)
pages_char_split = char_splitter.split_documents(pages_md_split)

# 5) Create an embeddings client/model (used later to embed the chunks)
embedding = OpenAIEmbeddings(model="text-embedding-ada-002")

# --- Code cell 7 ---
len(pages_char_split)

print(
    "Creating a Chroma vectorstore from `pages_char_split` by computing embeddings with `embedding`, "
    "then persisting the resulting index/database to './local-database' for later reuse."
)

vectorstore = Chroma.from_documents(
    documents=pages_char_split,
    embedding=embedding,
    persist_directory="./local-database",
)

# --- Code cell 9 ---
vectorstore_from_directory = Chroma(persist_directory = "./local-database",
                                    embedding_function = embedding)

