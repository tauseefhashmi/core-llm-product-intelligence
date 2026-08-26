import requests
import streamlit as st

st.set_page_config(page_title="LLM Product Intelligence", layout="wide")
st.title("🧠 LLM Product Intelligence")
st.caption("RAG + local/frontier LLMs + evaluation-ready architecture")
question = st.text_area("Ask a product research question", "Which laptop is the best value for a data engineer under €1,000?")
if st.button("Analyze"):
    with st.spinner("Retrieving evidence and generating answer..."):
        r = requests.post("http://api:8000/ask", json={"question": question}, timeout=180)
        r.raise_for_status()
        data = r.json()
    st.markdown(data["answer"])
    st.subheader("Retrieved evidence")
    for s in data["sources"]:
        st.write(f"**{s['source']} / chunk {s['chunk']}**")
        st.write(s["text"])
