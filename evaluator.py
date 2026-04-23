from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import ChatOpenAI

# Load embedding model (lightweight + fast)
model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_score(answer, context):
    answer_embedding = model.encode([answer])
    context_embedding = model.encode([context])

    similarity = cosine_similarity(answer_embedding, context_embedding)[0][0]
    return float(similarity)

def llm_verification(answer, context):
    llm = ChatOpenAI(model="gpt-4o-mini")

    prompt = f"""
    Context:
    {context}

    Answer:
    {answer}

    Is the answer supported by the context?

    Respond ONLY with:
    Supported
    Partially Supported
    Not Supported
    """

    response = llm.invoke(prompt)
    return response.content.strip()

def evaluate(answer, context):
    sim_score = semantic_score(answer, context)
    verification = llm_verification(answer, context)

    # Convert verification to numeric weight
    if verification == "Supported":
        weight = 1.0
    elif verification == "Partially Supported":
        weight = 0.6
    else:
        weight = 0.2

    final_score = sim_score * weight

    return round(final_score, 2), verification