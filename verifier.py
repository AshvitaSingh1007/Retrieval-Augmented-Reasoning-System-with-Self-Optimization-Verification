from langchain_openai import ChatOpenAI

def verify_answer(answer, context):
    llm = ChatOpenAI(model="gpt-4o-mini")

    prompt = f"""
    Context:
    {context}

    Answer:
    {answer}

    Is the answer fully supported by the context?

    Respond ONLY with:
    Supported
    Partially Supported
    Not Supported
    """

    response = llm.invoke(prompt)
    return response.content.strip()

