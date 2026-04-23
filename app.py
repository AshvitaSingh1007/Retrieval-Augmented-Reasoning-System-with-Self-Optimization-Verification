import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

from retriever import get_retriever, keyword_search
from generator import generate_answer
from verifier import verify_answer
from evaluator import evaluate

st.set_page_config(page_title="RAG Reasoning Demo", layout="wide")

st.title("Retrieval-Augmented Reasoning System")
st.write("Enter a query and see how retrieval, generation, and verification work together.")

query = st.text_input("Enter your question:")

if st.button("Run") and query:
    retriever, docs = get_retriever()

    dense_docs = retriever.invoke(query)
    dense_context = " ".join([doc.page_content for doc in dense_docs])

    keyword_docs = keyword_search(query, docs)
    keyword_context = " ".join(keyword_docs)

    context = dense_context + " " + keyword_context

    # Without verification
    answer_no_verification = generate_answer(query, context)

    # With verification
    answer = generate_answer(query, context)
    verification = verify_answer(answer, context)
    score = evaluate(answer, context)

    st.subheader("Results")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Without Verification")
        st.write(answer_no_verification)

    with col2:
        st.markdown("### With Verification")
        st.write(answer)
        st.write(f"**Verification:** {verification}")
        st.write(f"**Score:** {score}")

    st.markdown("### Retrieved Context (for transparency)")
    st.write(context)