# --- Markdown cell 1 ---
# # Indexing: Document Loading with PyPDF Loader

# --- Code cell 3 ---
# Run the line of code below to check the version of langchain in the current environment.
# Substitute "langchain" with any other package name to check their version.

# --- Code cell 4 ---

# --- Code cell 5 ---
from langchain_community.document_loaders import PyPDFLoader
import copy

# --- Code cell 6 ---
loader_pdf = PyPDFLoader("Introduction_to_Data_and_Data_Science.pdf")

# --- Code cell 7 ---
pages_pdf = loader_pdf.load()

# --- Code cell 8 ---
pages_pdf

# --- Code cell 9 ---
pages_pdf_cut = copy.deepcopy(pages_pdf)

# --- Code cell 10 ---
' '.join(pages_pdf_cut[0].page_content.split())

# --- Code cell 11 ---
for i in pages_pdf_cut:
    i.page_content = ' '.join(i.page_content.split())

# --- Code cell 12 ---
pages_pdf_cut

# --- Code cell 13 ---
pages_pdf[0].page_content, pages_pdf_cut[0].page_content

# --- Code cell 14 ---
# (empty)

# --- Code cell 15 ---
# (empty)
