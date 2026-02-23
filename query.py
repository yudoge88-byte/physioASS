from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import pipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load DB
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(embedding_function=embeddings, persist_directory="db", collection_name="physio_docs")

llm = pipeline("text-generation", model="distilgpt2", max_new_tokens=120, truncation=True)

def advanced_rag(patient_case, k=4, max_distance=0.85):  # <-- Distance, not similarity!
    """❌ 'No info' if poor matches (distance > threshold)"""
    docs_scores = db.similarity_search_with_score(patient_case, k=k)
    
    # Filter good matches (low distance = high similarity)
    good_docs = [(doc, score) for doc, score in docs_scores if score < max_distance]
    
    if not good_docs:
        return f"❌ No relevant guidelines (best distance: {docs_scores[0][1]:.3f}). Try rephrasing.", []
    
    print(f"✅ Found {len(good_docs)} good matches (best: {good_docs[0][1]:.3f})")
    
    # Build safe context
    context = ""
    for doc, score in good_docs:
        context += f"[{score:.3f}] {doc.page_content[:800]}\n\n"
    
    prompt = f"""Patient: {patient_case}

Guidelines:
{context}

Recommendation:"""
    
    result = llm(prompt, max_new_tokens=100)[0]['generated_text']
    return result.split('Recommendation:')[-1].strip(), good_docs

# Test incontinence
case = "Patient can't hold pee in - urinary incontinence"
answer, sources = advanced_rag(case, max_distance=0.9)  # Relaxed threshold
print("\n🏥", answer)

# 4️⃣ Test complex cases
cases = [
    """Patient: 45yo male, acute shoulder pain after fall. Limited ROM, positive impingement test. No numbness/weakness.""",
    """28yo female runner, can't hold pee in""",
    """65yo post-op rotator cuff repair, week 6, pain with external rotation."""
]

for case in cases:
    answer, sources = advanced_rag(case)
    print(f"\n🏥 CASE: {case[:80]}...")
    print(f"   {answer}")
    print(f"   📚 {len(sources)} sources")

