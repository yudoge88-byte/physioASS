from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

PDF_DIR = "guidelines"  # your folder with PDFs

# Load PDFs WITH METADATA
print("Loading PDFs with metadata...")
documents = []
for file in os.listdir(PDF_DIR):
    if file.endswith(".pdf"):
        filepath = os.path.join(PDF_DIR, file)
        loader = PyPDFLoader(filepath)
        pages = loader.load()
        # ADD CPG METADATA to each page
        for page in pages:
            page.metadata.update({
                "cpg_title": file.replace('.pdf', ''),  # "shoulder_protocol"
                "source_file": filepath,
                "chunk_type": "cpg_guideline"
            })
        documents.extend(pages)
print(f"Loaded {len(documents)} pages with metadata")

# Split into chunks (CPG-optimized)
print("Splitting into CPG chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2500,
    chunk_overlap=400,
    separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],
    keep_separator=True
)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} CPG chunks")


# Create embeddings
print("Creating local embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("🔄 Clean rebuild...")
db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="db",
    collection_name="physio_docs"
)
print("✅ Rebuild complete!")


# 🚨 DB SIZE CHECK
import os
sqlite_size = os.path.getsize("db/chroma.sqlite3") / 1024 / 1024  # MB
print(f"SQLite size: {sqlite_size:.1f} MB")

# Test reload
try:
    test_db = Chroma(embedding_function=embeddings, persist_directory="db")
    count = test_db._collection.count()
    print(f"✅ Vector count: {count}")
    
    if count == 0:
        print("❌ EMPTY DB - reindex needed!")
    elif sqlite_size < 1:
        print("⚠️  DB too small - check indexing")
    else:
        print("✅ DB healthy!")
        
except Exception as e:
    print(f"❌ Load failed: {e}")
# Verify with EXACT match
test_db = Chroma(
    embedding_function=embeddings,
    persist_directory="db",
    collection_name="physio_docs"
)
print(f"✅ Vectors: {test_db._collection.count()}")
docs = test_db.similarity_search("physio", k=1)
print(f"✅ Sample: {docs[0].page_content[:100]}...")
