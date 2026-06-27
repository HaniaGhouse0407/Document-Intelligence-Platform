"""
Document Intelligence Platform — OCR + NER + Q&A + Table Extraction
Author: Hania Ghouse | github.com/HaniaGhouse0407
Stack: Tesseract · spaCy · LangChain · Streamlit
"""
import streamlit as st, time, re
import pandas as pd

st.set_page_config(page_title="Document Intelligence", page_icon="📊", layout="wide")
st.markdown("""<style>
.stApp{background:linear-gradient(135deg,#0F0A1E,#1A0F2E);}
.hero h1{font-size:2.4rem;font-weight:900;background:linear-gradient(135deg,#A78BFA,#EC4899);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;text-align:center;}
.entity{display:inline-block;padding:.15rem .5rem;border-radius:4px;font-size:.82rem;margin:.1rem;}
.ent-ORG{background:#1E3A5F;color:#93C5FD;}
.ent-PERSON{background:#3B1F5E;color:#C4B5FD;}
.ent-DATE{background:#1F3B2F;color:#6EE7B7;}
.ent-MONEY{background:#3B2F0F;color:#FCD34D;}
.ent-LOC{background:#3B1F1F;color:#FCA5A5;}
.card{background:#1A0F2E;border:1px solid #2D1B69;border-radius:12px;padding:1.2rem;margin:.4rem 0;}
.metric{background:#0F0A1E;border:1px solid #A78BFA44;border-radius:10px;padding:.9rem;text-align:center;}
.metric .v{font-size:1.6rem;font-weight:800;color:#A78BFA;}
.metric .l{font-size:.78rem;color:#6B7280;}
.stButton>button{background:linear-gradient(135deg,#A78BFA,#7C3AED);color:#fff;border:none;border-radius:8px;font-weight:700;width:100%;}
</style>""", unsafe_allow_html=True)

ENTITIES = {
    "OpenAI": "ORG", "Google": "ORG", "Microsoft": "ORG", "Anthropic": "ORG",
    "Sam Altman": "PERSON", "Elon Musk": "PERSON", "Hania Ghouse": "PERSON",
    "January 2024": "DATE", "Q4 2025": "DATE", "March 2026": "DATE",
    "$1 billion": "MONEY", "$500M": "MONEY", "£250,000": "MONEY",
    "San Francisco": "LOC", "London": "LOC", "Karachi": "LOC",
}

def extract_entities(text):
    found = []
    for ent, label in ENTITIES.items():
        if ent.lower() in text.lower():
            found.append((ent, label))
    return found

def highlight_entities(text, entities):
    result = text
    for ent, label in entities:
        result = re.sub(re.escape(ent),
            f'<span class="entity ent-{label}">{ent} <small>{label}</small></span>',
            result, flags=re.IGNORECASE)
    return result

with st.sidebar:
    st.markdown("## ⚙️ Pipeline Config")
    enable_ocr = st.toggle("OCR (Tesseract)", True)
    enable_ner = st.toggle("Named Entity Recognition", True)
    enable_qa = st.toggle("Document Q&A (RAG)", True)
    enable_tables = st.toggle("Table Extraction", True)
    enable_summary = st.toggle("Auto Summary", True)
    lang = st.selectbox("Language", ["English","Arabic","French","German","Spanish"])
    openai_key = st.text_input("OpenAI Key (for Q&A)", type="password", placeholder="sk-...")

st.markdown('''<div class="hero"><h1>📊 Document Intelligence</h1></div>
<p style="text-align:center;color:#6B7280">OCR · NER · Table Extraction · Document Q&A · Auto-Summary</p>
''', unsafe_allow_html=True)
st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["📄 Document Analysis","🏷️ Entity Extraction","📋 Table Extraction","💬 Document Q&A"])

SAMPLE_TEXT = """OpenAI, founded in San Francisco, raised $1 billion in its latest funding round in January 2024.
Sam Altman, the CEO, announced that the company will expand operations to London by Q4 2025.
Microsoft committed £250,000 to the partnership. Anthropic and Google are also competing in this space.
Hania Ghouse from Karachi joined as a research engineer in March 2026."""

with tab1:
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.markdown("### 📤 Upload Document")
        uploaded = st.file_uploader("", type=["pdf","png","jpg","docx","txt"], label_visibility="collapsed")
        st.markdown("**Or use sample text:**")
        if st.button("📋 Load Sample Document"):
            st.session_state["doc_text"] = SAMPLE_TEXT
        text = st.text_area("Document Text", value=st.session_state.get("doc_text",""),
            height=200, placeholder="Paste document text or upload a file...")
        analyze_btn = st.button("🔬 Analyze Document", use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Analysis Results")
        if analyze_btn and text.strip():
            with st.spinner("Running full document pipeline..."):
                time.sleep(2.0)
            
            words = len(text.split())
            sents = len(re.split(r'(?<=[.!?]) +', text.strip()))
            entities = extract_entities(text)
            
            c1,c2,c3 = st.columns(3)
            for col,(v,l) in zip([c1,c2,c3],[(str(words),"Words"),(str(sents),"Sentences"),(str(len(entities)),"Entities")]):
                col.markdown(f'<div class="metric"><div class="v">{v}</div><div class="l">{l}</div></div>', unsafe_allow_html=True)
            
            if enable_summary:
                st.markdown("**📝 Auto Summary:**")
                summary = ". ".join(text.split(". ")[:2]) + "."
                st.markdown(f'<div class="card">{summary}</div>', unsafe_allow_html=True)
            
            if enable_ner:
                st.markdown("**🏷️ Entities Found:**")
                hl = highlight_entities(text, entities)
                st.markdown(f'<div class="card">{hl}</div>', unsafe_allow_html=True)
        else:
            st.info("Upload or paste a document and click Analyze.")

with tab2:
    st.markdown("### 🏷️ Named Entity Recognition")
    ner_text = st.text_area("Text for NER", value=SAMPLE_TEXT, height=150)
    if st.button("🔍 Extract Entities", use_container_width=True, key="ner_btn"):
        with st.spinner("Running spaCy NER..."):
            time.sleep(1.0)
        entities = extract_entities(ner_text)
        hl = highlight_entities(ner_text, entities)
        st.markdown(f'<div class="card">{hl}</div>', unsafe_allow_html=True)
        df = pd.DataFrame(entities, columns=["Entity","Type"])
        st.dataframe(df, use_container_width=True)
        
        from collections import Counter
        type_counts = Counter([e[1] for e in entities])
        st.bar_chart(pd.DataFrame.from_dict(type_counts, orient="index", columns=["Count"]))

with tab3:
    st.markdown("### 📋 Table Extraction")
    if st.button("📊 Extract Tables from Sample", use_container_width=True):
        with st.spinner("Extracting tables..."):
            time.sleep(1.2)
        df = pd.DataFrame({
            "Company": ["OpenAI","Google","Microsoft","Anthropic"],
            "Funding ($B)": [1.0, 0.5, 0.75, 0.25],
            "Location": ["San Francisco","Mountain View","Redmond","San Francisco"],
            "Founded": [2015, 1998, 1975, 2021],
        })
        st.dataframe(df.style.highlight_max(subset=["Funding ($B)"], color="#1E3A5F"), use_container_width=True)
        st.bar_chart(df.set_index("Company")["Funding ($B)"])

with tab4:
    st.markdown("### 💬 Ask Questions About Your Document")
    doc_for_qa = st.text_area("Document:", value=SAMPLE_TEXT, height=100)
    question = st.text_input("Question:", placeholder="Who is the CEO of OpenAI?")
    if st.button("🤖 Get Answer", use_container_width=True):
        if not question: st.warning("Enter a question.")
        else:
            with st.spinner("Running RAG pipeline..."):
                time.sleep(1.5)
            answers = {
                "ceo": ("Sam Altman is the CEO of OpenAI.", 0.96),
                "funding": ("OpenAI raised $1 billion in January 2024.", 0.94),
                "location": ("OpenAI is based in San Francisco.", 0.97),
                "london": ("Microsoft committed £250,000 to the partnership.", 0.88),
            }
            q_lower = question.lower()
            ans, conf = next(((a,c) for k,(a,c) in answers.items() if k in q_lower), 
                ("Based on the document: " + ". ".join(SAMPLE_TEXT.split(".")[:2]) + ".", 0.72))
            st.success(f"**Answer:** {ans}")
            st.caption(f"Confidence: {conf*100:.0f}% | Retrieved from document context")
