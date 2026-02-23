import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(embedding_function=embeddings, persist_directory="db", collection_name="physio_docs")

db = load_db()
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1)

st.title("🏥 Physio CPG AI")
st.markdown("**Groq | Reasoning + 'From CPG [1]: quote'**")

patient_case = st.text_area("Patient Case:", 
    placeholder="lateral knee+ankle pain 3w post-run")

if st.button("🧠 Generate", type="primary") and patient_case:
    with st.spinner("⚡ Analysis..."):
        docs_scores = db.similarity_search_with_score(patient_case, k=8)
        top_score = docs_scores[0][1]
        
        if top_score > 0.90:
            st.warning("🔍 Refine")
        else:
            unique_docs = []
            seen = set()
            for doc, score in docs_scores:
                source = doc.metadata.get('source', '')
                if source not in seen:
                    unique_docs.append(doc)
                    seen.add(source)
            
            sources = [f"[{i+1}] {doc.metadata.get('title', 'No Title')} | {doc.metadata.get('authors', 'No Authors')} ({doc.metadata.get('year', 'No Year')})"
                      for i, doc in enumerate(unique_docs[:6])]
            
            # Numbered context chunks - SHORTER for concise quotes
            numbered_context = ""
            for i, doc in enumerate(unique_docs[:6]):
                numbered_context += f"\n[{i+1}] {doc.page_content[:300]}\n"
            
            prompt = ChatPromptTemplate.from_template("""
You are an expert physiotherapist specializing in differential diagnosis.

**Patient Case**: {question}

**CPG Instructions**: Select SHORTEST relevant phrases (<50 words) defining:
- Diagnostic criteria/symptoms matching patient exactly
- Clinical tests/findings that align
Ignore irrelevant sections.

**Excerpts (numbered by unique source)**:
{numbered_context}

**Sources**: {sources}

Format EXACTLY:

## 🧠 Clinical Reasoning (Synthesis)
Holistic analysis...

## 📋 Differential Diagnosis
1. **Diagnosis** (probability): Brief reasoning...
   From CPG [1]: "SHORT quote (<50 words)"
   *Relevance*: How this matches patient's specific symptoms/tests.

2. **Next Dx**...

## 📊 Prognosis (6w)
From CPG [X]: "SHORT quote"

## 🩺 Plan
**W1-2**: From CPG [Y]: "SHORT quote" + patient-specific adjustment

**Red Flags**: From CPG [Z]: "SHORT quote"
""")
            
            chain = ({"numbered_context": lambda x: numbered_context,
                     "sources": lambda x: "\n".join(sources),
                     "question": RunnablePassthrough()} 
                    | prompt | llm | StrOutputParser())
            
            answer = chain.invoke(patient_case)
            st.markdown(answer)
            
            with st.expander("📚 Full Sources"):
                for i, src in enumerate(sources):
                    st.write(f"**[{i+1}]** {src}")
            
            st.caption(f"**{len(unique_docs)} CPGs** | {top_score:.3f}")

with st.expander("🔍 Matches"):
    if 'docs_scores' in locals():
        for i, (doc, score) in enumerate(docs_scores[:4], 1):
            st.metric(f"#{i}", f"{score:.3f}")
            st.caption(doc.metadata.get('cpg_title', '?'))
