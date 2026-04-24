from langchain_openai import ChatOpenAI

def verify_answer(query, answer, context):
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    prompt = f"""
    Question: {query}
    Answer: {answer}
    Context: {context}

    Determine whether the answer is supported by the context.
    Respond with one of:
    Supported
    Partially Supported
    Not Supported
    """

    response = llm.invoke(prompt)

    return response.content