import os
from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize LangSmith client
# Uses LANGCHAIN_API_KEY from .env automatically
client = Client()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)


# --- Faithfulness Evaluator ---
# Checks if the answer is grounded in the retrieved context
# Prevents hallucination — answer must come from the document
def faithfulness_evaluator(run, example) -> dict:
    prompt = f"""You are evaluating if an answer is faithful to the provided context.

Context: {example.inputs.get('context', '')}
Answer: {run.outputs.get('answer', '')}

Score from 0.0 to 1.0 where:
1.0 = answer is completely grounded in the context
0.0 = answer contains information not found in the context

Respond with only a number between 0.0 and 1.0."""

    response = llm.invoke(prompt)
    try:
        score = float(response.content.strip())
    except:
        score = 0.0

    return {"key": "faithfulness", "score": score}


# --- Relevance Evaluator ---
# Checks if the retrieved chunks are relevant to the question
# Measures retrieval quality, not just answer quality
def relevance_evaluator(run, example) -> dict:
    prompt = f"""You are evaluating if the retrieved context is relevant to the question.

Question: {example.inputs.get('question', '')}
Retrieved Context: {run.outputs.get('context', '')}

Score from 0.0 to 1.0 where:
1.0 = context is highly relevant to the question
0.0 = context is completely irrelevant

Respond with only a number between 0.0 and 1.0."""

    response = llm.invoke(prompt)
    try:
        score = float(response.content.strip())
    except:
        score = 0.0

    return {"key": "relevance", "score": score}


# --- Correctness Evaluator ---
# Compares the answer against a reference answer
# Requires a ground truth dataset in LangSmith
def correctness_evaluator(run, example) -> dict:
    prompt = f"""You are evaluating if an answer is correct compared to the reference answer.

Question: {example.inputs.get('question', '')}
Reference Answer: {example.outputs.get('answer', '')}
Generated Answer: {run.outputs.get('answer', '')}

Score from 0.0 to 1.0 where:
1.0 = generated answer is factually identical to reference
0.0 = generated answer is completely wrong

Respond with only a number between 0.0 and 1.0."""

    response = llm.invoke(prompt)
    try:
        score = float(response.content.strip())
    except:
        score = 0.0

    return {"key": "correctness", "score": score}


# --- Create evaluation dataset in LangSmith ---
# Builds a dataset of question/answer pairs for evaluation
# You can expand this with real financial document QA pairs
def create_eval_dataset(dataset_name: str = "financial-doc-eval"):
    examples = [
        {
            "inputs": {
                "question": "What was the net revenue growth compared to last quarter?",
                "context": "Net revenue for Q3 was $4.2M, compared to $3.7M in Q2, representing a 13.5% increase."
            },
            "outputs": {
                "answer": "Net revenue grew 13.5% compared to last quarter."
            }
        },
        {
            "inputs": {
                "question": "What is the current debt-to-equity ratio?",
                "context": "Total debt stands at $12M with shareholder equity of $48M, resulting in a debt-to-equity ratio of 0.25."
            },
            "outputs": {
                "answer": "The current debt-to-equity ratio is 0.25."
            }
        },
        {
            "inputs": {
                "question": "What are the main risk factors mentioned?",
                "context": "Key risk factors include market volatility, regulatory changes in the fintech sector, and exposure to credit default risk in the SME segment."
            },
            "outputs": {
                "answer": "Main risk factors are market volatility, regulatory changes, and credit default risk in SMEs."
            }
        }
    ]

    # Create or update the dataset in LangSmith
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Evaluation dataset for financial document Q&A agent"
    )

    client.create_examples(
        inputs=[e["inputs"] for e in examples],
        outputs=[e["outputs"] for e in examples],
        dataset_id=dataset.id
    )

    return dataset


# --- Run full evaluation ---
# Runs all evaluators against the dataset and logs results to LangSmith
def run_evaluation(agent_function, dataset_name: str = "financial-doc-eval"):
    results = evaluate(
        agent_function,
        data=dataset_name,
        evaluators=[
            faithfulness_evaluator,
            relevance_evaluator,
            correctness_evaluator,
        ],
        experiment_prefix="financial-doc-agent-eval",
        metadata={"model": "gpt-4o-mini", "embedding": "all-MiniLM-L6-v2"}
    )
    return results