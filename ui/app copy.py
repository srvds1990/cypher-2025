# ui/app.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from db.vector_store_old import list_sessions
from agents.graph import build_summary_graph, build_qa_graph

st.set_page_config(page_title="Cypher25 AI Companion", layout="wide")
st.title("🤖 Cypher25 — AI Knowledge Companion")

sessions = list_sessions()
if "selected_session" not in st.session_state:
    st.session_state.selected_session = None

st.markdown("### 🔎 Select a Session")
cols = st.columns(min(len(sessions), 3))
for i, sess in enumerate(sessions):
    with cols[i % len(cols)]:
        if st.button(f"📌 {sess}", use_container_width=True):
            st.session_state.selected_session = sess

if st.session_state.selected_session:
    st.markdown(f"""
    <div style="background-color:#0E1117; padding:15px; border-radius:10px; margin:15px 0;">
        <h2 style="color:#FAFAFA; text-align:center;">🎯 {st.session_state.selected_session}</h2>
    </div>
    """, unsafe_allow_html=True)

    question = st.text_input("Ask something about this session:", key="query_input")

    # Build graphs ONCE
    summary_graph = build_summary_graph()
    qa_graph = build_qa_graph()

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("📑 Summarize Session"):
            result = summary_graph.invoke({
                "session": st.session_state.selected_session,
                "query": "summary"
            })

            st.subheader("📄 Context")
            st.markdown(result.get("context", "_No context returned_"))

            st.subheader("📝 Summary")
            st.markdown(result.get("summary", result.get("answer", "_No summary returned_")))

            st.subheader("📊 Visualization")
            st.markdown(result.get("visualization", "_No visualization returned_"))

    with col2:
        if st.button("💬 Run Agent on Query") and question:
            result = qa_graph.invoke({
                "session": st.session_state.selected_session,
                "query": question,
            })
            st.subheader("🤖 Agent Answer")
            st.markdown(result.get("answer", "_No answer returned_"))
else:
    st.info("Please select a session from above to start exploring.")
