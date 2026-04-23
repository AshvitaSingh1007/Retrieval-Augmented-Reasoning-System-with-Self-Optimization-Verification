from dotenv import load_dotenv
import os

load_dotenv()

from retriever import get_retriever
from generator import generate_answer
from verifier import verify_answer
from evaluator import evaluate

def run(query):
    retriever, docs = get_retriever()

    dense_docs = retriever.invoke(query)
    dense_context = " ".join([doc.page_content for doc in dense_docs])

    context = dense_context

    # WITHOUT verification
    answer_no_verification = generate_answer(query, context)

    # WITH verification
    answer = generate_answer(query, context)
    score, verification = evaluate(answer, context)

    print("\n--- RESULT ---")
    print("Question:", query)

    print("\n[WITHOUT VERIFICATION]")
    print("Answer:", answer_no_verification)

    print("\n[WITH VERIFICATION]")
    print("Answer:", answer)
    print("Verification:", verification)
    print("Score:", score)


if __name__ == "__main__":
    query = input("Enter your question: ")
    run(query)