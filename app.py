"""
app.py — CaseCounsel AI: Streamlit UI (OpenAI-powered)
Run with: streamlit run app.py
"""

import os
import glob
import time
import streamlit as st
from dotenv import load_dotenv
from src.document_processor import DocumentProcessor, structure_facts_with_llm
from src.legal_agent import LegalAgent
from src.llm_client import LLMClient
from src.report_builder import build_case_report

load_dotenv()  # local dev: reads OPENAI_API_KEY from a .env file

try:
    if "OPENAI_API_KEY" not in os.environ and "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass  # no secrets.toml present locally — .env already handled it

st.set_page_config(page_title="CaseCounsel AI", page_icon="⚖️", layout="wide", initial_sidebar_state="auto")

st.markdown("""
<style>
:root {
    --accent-red: #9a2b2b;
    --accent-blue: #1f4e8c;
    --ink: #1c1c1c;
    --muted: #6b6b6b;
}
.main { background-color: #faf9f7; }
h1, h2, h3 { font-family: 'Georgia', serif; color: var(--ink); }
.hero {
    padding: 26px 32px; border-radius: 14px; margin-bottom: 20px;
    background: linear-gradient(135deg, #1c2b45 0%, #2c3e5e 100%);
    color: #f4f1ea;
    display: flex; align-items: center; justify-content: space-between;
}
.hero h1 { color: #f4f1ea; margin-bottom: 4px; font-size: 1.7em; }
.hero p { color: #d9d4c8; margin: 0; font-size: 0.9em; max-width: 620px; }
.hero .badge {
    background: rgba(255,255,255,0.12); border-radius: 20px; padding: 6px 14px;
    font-size: 0.8em; white-space: nowrap;
}
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 2px solid #e5e0d5; }
.stTabs [data-baseweb="tab"] { font-weight: 600; padding: 10px 16px; }
.point-card {
    background: white; border-left: 5px solid var(--accent-red); border-radius: 8px;
    padding: 16px 20px; margin-bottom: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.point-card.cross { border-left-color: var(--accent-blue); }
.point-card .grounded {
    margin-top: 8px; font-size: 0.82em; color: var(--muted);
    border-top: 1px dashed #e5e0d5; padding-top: 8px;
}
.strength-pill {
    display: inline-block; padding: 2px 10px; border-radius: 12px;
    font-size: 0.75em; font-weight: 700; letter-spacing: 0.03em; text-transform: uppercase;
}
.strength-strong { background: #e3f2e6; color: #1a7a3c; }
.strength-moderate { background: #fdf1d9; color: #a06d00; }
.strength-weak { background: #f0f0f0; color: #888; }
.quote-box {
    background: #f4f1e8; border-radius: 6px; padding: 10px 14px; margin-top: 8px;
    font-style: italic; border-left: 3px solid var(--accent-blue);
}
.status-badge { padding: 3px 10px; border-radius: 10px; font-size: 0.8em; font-weight: 600; }
.status-compliant { background: #e3f2e6; color: #1a7a3c; }
.status-non-compliant { background: #fbe4e4; color: #b12525; }
.status-unclear { background: #f0f0f0; color: #888; }
.metric-box {
    background: white; border-radius: 8px; padding: 14px; text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.metric-box .num { font-size: 1.8em; font-weight: 700; color: var(--accent-red); }
.metric-box .label { font-size: 0.8em; color: var(--muted); text-transform: uppercase; }
.empty-step {
    background: white; border-radius: 10px; padding: 18px; height: 100%;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06); border-top: 3px solid var(--accent-blue);
}

/* ---------- Responsive breakpoints ---------- */

/* Tablets and small laptops */
@media (max-width: 900px) {
    .hero { padding: 20px 22px; }
    .hero h1 { font-size: 1.4em; }
    .hero p { font-size: 0.85em; }
    .metric-box .num { font-size: 1.5em; }
}

/* Phones */
@media (max-width: 640px) {
    .hero {
        flex-direction: column; align-items: flex-start; gap: 10px;
        padding: 18px 18px;
    }
    .hero .badge { align-self: flex-start; font-size: 0.75em; }
    .hero h1 { font-size: 1.25em; margin-bottom: 2px; }
    .hero p { font-size: 0.82em; max-width: 100%; }

    .point-card { padding: 12px 14px; margin-bottom: 10px; }
    .quote-box { padding: 8px 10px; font-size: 0.92em; }
    .grounded { font-size: 0.78em; }

    .metric-box { padding: 10px; }
    .metric-box .num { font-size: 1.3em; }
    .metric-box .label { font-size: 0.7em; }

    .empty-step { padding: 14px; margin-bottom: 10px; }

    .stTabs [data-baseweb="tab"] { padding: 8px 10px; font-size: 0.85em; }

    h1, h2, h3 { font-size: 90%; }
}

/* Very small phones */
@media (max-width: 400px) {
    .hero h1 { font-size: 1.1em; }
    .metric-box .num { font-size: 1.1em; }
    .strength-pill, .status-badge { font-size: 0.68em; padding: 2px 7px; }
}

/* Force Streamlit's horizontal column blocks to wrap into 2 per row on mobile
   instead of squeezing 4-5 columns into one unreadable line */
@media (max-width: 640px) {
    div[data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        min-width: 46% !important;
        flex: 1 1 46% !important;
        margin-bottom: 8px;
    }
}
</style>
""", unsafe_allow_html=True)

for key, default in [
    ("agent", None), ("analysis", None), ("case_doc", None),
    ("structured_facts", None), ("chat_history", []), ("file_name", None),
    ("history", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def corpus_is_populated() -> bool:
    return len(glob.glob("data/legal_corpus/*.json")) > 0 and os.path.isdir("data/vectorstore")


def load_from_history(idx: int):
    entry = st.session_state.history[idx]
    st.session_state.analysis = entry["analysis"]
    st.session_state.case_doc = entry["case_doc"]
    st.session_state.structured_facts = entry["structured_facts"]
    st.session_state.file_name = entry["file_name"]
    st.session_state.chat_history = []
    st.session_state.agent = entry["agent"]


with st.sidebar:
    st.markdown("### ⚖️ CaseCounsel AI")
    st.caption("Prosecution & cross-examination research assistant, powered by OpenAI")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        st.success("OpenAI API key loaded", icon="🔑")
    else:
        st.error("No OPENAI_API_KEY found. Add it to a `.env` file (local) or Secrets (cloud) and restart.", icon="🔑")

    model_choice = st.selectbox(
        "Model", ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"], index=0,
        help="gpt-4o: best quality. gpt-4o-mini: faster & cheaper, good for testing.",
    )

    st.divider()
    st.markdown("**Upload case document**")
    input_mode = st.radio("Input method", ["Upload file", "Use camera"], horizontal=True, label_visibility="collapsed")

    uploaded_file = None
    camera_photo = None

    if input_mode == "Upload file":
        uploaded_file = st.file_uploader(
            "FIR, Charge Sheet, or Witness Statement — PDF, Word (.docx), or photo (PNG/JPG)",
            type=["pdf", "docx", "png", "jpg", "jpeg"],
            label_visibility="collapsed",
        )
    else:
        camera_photo = st.camera_input("Take a photo of the document", label_visibility="collapsed")

    analyze_btn = st.button("🔍  Analyze Document", type="primary", use_container_width=True)

    if not corpus_is_populated():
        st.warning("Legal knowledge base not built yet. Run `python -m src.legal_kb_builder` after populating `data/legal_corpus/`.", icon="⚠️")

    if st.session_state.history:
        st.divider()
        st.markdown("**Recent cases**")
        for i, entry in enumerate(reversed(st.session_state.history)):
            real_idx = len(st.session_state.history) - 1 - i
            label = f"{entry['doc_type']} — {entry['file_name'][:24]}"
            if st.button(label, key=f"hist_{real_idx}", use_container_width=True):
                load_from_history(real_idx)
                st.rerun()

    st.divider()
    st.caption(
        "⚠️ Research aid only, for use by qualified advocates. "
        "All output must be independently verified against primary sources before filing or court use."
    )

badge_text = f"{len(st.session_state.history)} case(s) analyzed this session" if st.session_state.history else "Ready"
st.markdown(f"""
<div class="hero">
  <div>
    <h1>⚖️ CaseCounsel AI</h1>
    <p>Upload a case document to generate grounded prosecution points, cross-examination angles, and procedural compliance checks — backed by retrieval over your verified legal corpus.</p>
  </div>
  <div class="badge">{badge_text}</div>
</div>
""", unsafe_allow_html=True)

if analyze_btn:
    input_file = uploaded_file or camera_photo
    if not api_key:
        st.error("No OpenAI API key configured — set OPENAI_API_KEY in your .env file or Secrets.")
    elif not input_file:
        st.error("Upload a document or take a photo first.")
    elif not corpus_is_populated():
        st.error("Legal knowledge base is empty. Populate `data/legal_corpus/` and run the indexer first — see README.")
    else:
        start_time = time.time()
        progress = st.progress(0, text="Extracting document text...")

        os.makedirs("uploads", exist_ok=True)
        file_name = getattr(input_file, "name", None) or "camera_capture.jpg"
        save_path = f"uploads/{file_name}"
        with open(save_path, "wb") as f:
            f.write(input_file.getbuffer())

        try:
            processor = DocumentProcessor()
            case_doc = processor.process(save_path)
            progress.progress(25, text="Structuring case facts...")

            llm_client = LLMClient(model=model_choice)
            structured_facts = structure_facts_with_llm(llm_client, case_doc)
            progress.progress(50, text="Retrieving relevant law and precedents...")

            agent = LegalAgent(llm_client=llm_client)
            progress.progress(75, text="Generating prosecution & cross-examination analysis...")
            analysis = agent.analyze_case(case_doc, structured_facts)
            progress.progress(100, text="Done.")

            elapsed = time.time() - start_time

            st.session_state.agent = agent
            st.session_state.analysis = analysis
            st.session_state.case_doc = case_doc
            st.session_state.structured_facts = structured_facts
            st.session_state.chat_history = []
            st.session_state.file_name = file_name

            st.session_state.history.append({
                "file_name": file_name,
                "doc_type": case_doc.doc_type,
                "analysis": analysis,
                "case_doc": case_doc,
                "structured_facts": structured_facts,
                "agent": agent,
            })

            progress.empty()
            st.toast(f"Analysis complete in {elapsed:.1f}s", icon="✅")
            st.success(f"Analysis complete for **{case_doc.doc_type}** ({file_name}) in {elapsed:.1f}s.")
        except Exception as e:
            progress.empty()
            st.error(f"Something went wrong while processing this document: {e}")

if st.session_state.analysis:
    analysis = st.session_state.analysis
    case_doc = st.session_state.case_doc
    facts = st.session_state.structured_facts
    file_name = st.session_state.file_name or "document"

    if "error" in analysis:
        st.error(f"Analysis failed to parse: {analysis.get('error')}")
        with st.expander("Raw model output"):
            st.text(analysis.get("raw_output", ""))
    else:
        m1, m2, m3, m4, m5 = st.columns([1, 1, 1, 1, 1.3])
        with m1:
            st.markdown(f'<div class="metric-box"><div class="num">{len(analysis.get("prosecution_points", []))}</div><div class="label">Prosecution</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-box"><div class="num">{len(analysis.get("cross_examination_points", []))}</div><div class="label">Cross-Exam</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-box"><div class="num">{len(analysis.get("procedural_checks", []))}</div><div class="label">Procedural</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-box"><div class="num">{len(case_doc.sections_cited)}</div><div class="label">Sections</div></div>', unsafe_allow_html=True)
        with m5:
            pdf_bytes = build_case_report(case_doc, facts, analysis, file_name)
            st.download_button(
                "📄 Download PDF Report", data=pdf_bytes,
                file_name=f"{os.path.splitext(file_name)[0]}_analysis.pdf",
                mime="application/pdf", use_container_width=True,
            )

        st.write("")
        tab_overview, tab_prosecution, tab_cross, tab_procedure, tab_chat = st.tabs(
            ["📄  Overview", "🔴  Prosecution Points", "🔵  Cross-Examination", "✅  Procedural Checks", "💬  Ask Follow-up"]
        )

        with tab_overview:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Document Summary")
                st.write(f"**Type:** {case_doc.doc_type}")
                st.write(f"**FIR Number:** {case_doc.fir_number or 'Not detected'}")
                st.write(f"**Police Station:** {case_doc.police_station or 'Not detected'}")
                st.write(f"**Sections Cited:** {', '.join(case_doc.sections_cited) or 'None detected'}")
            with col2:
                st.subheader("Extracted Facts")
                st.write(f"**Complainant:** {facts.get('complainant', 'N/A')}")
                st.write(f"**Accused:** {', '.join(facts.get('accused') or []) or 'N/A'}")
                st.write(f"**Date of Incident:** {facts.get('date_of_incident', 'N/A')}")
                st.write(f"**Investigating Officer:** {facts.get('investigating_officer', 'N/A')}")

            st.subheader("Allegation Summary")
            st.info(facts.get("allegation_summary", "Not available"))

            st.subheader("Applicable Law")
            st.write(analysis.get("applicable_law_summary", "N/A"))

            if analysis.get("gaps_in_retrieved_context"):
                with st.expander("⚠️ Gaps flagged by the model — verify manually"):
                    for gap in analysis["gaps_in_retrieved_context"]:
                        st.write(f"- {gap}")

            with st.expander("View extracted raw text"):
                st.text_area("Raw text", case_doc.raw_text, height=250, label_visibility="collapsed")

        with tab_prosecution:
            st.subheader("Prosecution Points")
            points = analysis.get("prosecution_points", [])
            if not points:
                st.caption("No prosecution points generated.")
            for p in points:
                strength = p.get("strength", "moderate")
                st.markdown(f"""
                <div class="point-card">
                    <div>{p.get('point')}</div>
                    <div class="grounded">
                        Grounded in: {p.get('grounded_in')} &nbsp;·&nbsp;
                        <span class="strength-pill strength-{strength}">{strength}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with tab_cross:
            st.subheader("Cross-Examination Points")
            points = analysis.get("cross_examination_points", [])
            if not points:
                st.caption("No cross-examination points generated.")
            for p in points:
                st.markdown(f"""
                <div class="point-card cross">
                    <div><b>Target:</b> {p.get('target')}</div>
                    <div style="margin-top:4px;">{p.get('point')}</div>
                    <div class="quote-box">"{p.get('suggested_question')}"</div>
                    <div class="grounded">Grounded in: {p.get('grounded_in')}</div>
                </div>
                """, unsafe_allow_html=True)

        with tab_procedure:
            st.subheader("Procedural Compliance Checks")
            checks = analysis.get("procedural_checks", [])
            if not checks:
                st.caption("No procedural checks generated.")
            for c in checks:
                status = c.get("status", "unclear")
                badge_class = f"status-{status}"
                st.markdown(f"**{c.get('check')}**  <span class='status-badge {badge_class}'>{status}</span>", unsafe_allow_html=True)
                st.caption(c.get("note", ""))
                st.write("")

        with tab_chat:
            st.subheader("Ask about this case")
            st.caption("e.g. \"Draft a cross-examination question about the FIR delay\" or \"What if the incident predates BNS?\"")
            for turn in st.session_state.chat_history:
                with st.chat_message(turn["role"]):
                    st.write(turn["content"])

            question = st.chat_input("Ask a follow-up question...")
            if question:
                st.session_state.chat_history.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.write(question)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        answer = st.session_state.agent.ask_followup(question, st.session_state.chat_history)
                        st.write(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

else:
    st.markdown("### How it works")
    c1, c2, c3, c4 = st.columns(4)
    steps = [
        ("1️⃣", "Upload", "FIR, charge sheet, or witness statement — PDF, Word, photo, or camera capture."),
        ("2️⃣", "Extract", "Text, sections cited, and structured facts are pulled out automatically."),
        ("3️⃣", "Retrieve", "Relevant statute sections, amendments, and case law come from your legal corpus."),
        ("4️⃣", "Analyze", "Grounded prosecution & cross-examination points, plus a chat to dig deeper."),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3, c4], steps):
        with col:
            st.markdown(f'<div class="empty-step"><b>{icon} {title}</b><br/><span style="color:#6b6b6b; font-size:0.88em;">{desc}</span></div>', unsafe_allow_html=True)

    st.write("")
    st.caption("Upload a document in the sidebar to get started. Past analyses this session will appear in **Recent cases** for quick recall.")
