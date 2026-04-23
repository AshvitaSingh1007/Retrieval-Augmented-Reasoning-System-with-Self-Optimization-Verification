from langchain_openai import ChatOpenAI

def generate_answer(query, context):
    llm = ChatOpenAI(model="gpt-4o-mini")

    prompt = f"""
    You are an AI assistant.

    Use the context below as your primary source. If needed, explain clearly using it.
    If the answer is not present, say "I don't know".

    Context:
    {context}

    Question:
    {query}
    """

    response = llm.invoke(prompt)
    return response.content