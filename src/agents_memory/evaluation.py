"""LLM-as-judge evaluation for memory retrieval systems."""

import json
import os
import re

from openai import OpenAI

JUDGE_PROMPT = """You are evaluating a memory retrieval system.

Query: {query}
Expected: {expected}
Retrieved: {retrieved}

For each dimension, answer PASS (1) or FAIL (0):
1. Relevant: Are retrieved items relevant to the query?
2. Complete: Does retrieval cover the key expected information?
3. Accurate: Are the retrieved facts correct?

Return JSON:
{{"relevant": 0 or 1, "complete": 0 or 1, "accurate": 0 or 1, "explanation": "..."}}
"""


def evaluate_retrieval(query: str, expected: str, retrieved: str) -> dict:
    """Use LLM to judge retrieval quality.

    Args:
        query: The user query or context
        expected: What information should be retrieved
        retrieved: What was actually retrieved

    Returns:
        dict with relevant, complete, accurate scores (0 or 1) and explanation
    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=os.environ.get("JUDGE_MODEL", "gpt-5.2"),
        messages=[
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    query=query, expected=expected, retrieved=retrieved
                ),
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    return json.loads(response.choices[0].message.content)


def evaluate_batch(results: list[dict]) -> dict:
    """Evaluate multiple results and return summary.

    Args:
        results: List of dicts with query, expected, retrieved keys

    Returns:
        dict with pass rates for each dimension (0.0 to 1.0)
    """
    scores = {"relevant": [], "complete": [], "accurate": []}

    for r in results:
        eval_result = evaluate_retrieval(
            query=r["query"], expected=r["expected"], retrieved=r["retrieved"]
        )
        for key in scores:
            scores[key].append(eval_result[key])

    n = len(results)
    return {
        "relevant_rate": sum(scores["relevant"]) / n,
        "complete_rate": sum(scores["complete"]) / n,
        "accurate_rate": sum(scores["accurate"]) / n,
        "pass_rate": sum(sum(v) for v in scores.values()) / (n * 3),
    }


# ------------------------------------------------------------------
# Shared metrics used by benchmark scripts
# ------------------------------------------------------------------

REFUSAL_MARKERS = [
    "no info",
    "not specified",
    "not mentioned",
    "no direct",
    "not available",
    "no evidence",
    "none",
    "not found",
    "no relevant",
    "no data",
    "cannot be determined",
    "not provided",
    "no specific",
    "unknown",
    "i don't",
    "no memory",
    "no record",
    "not addressed",
]


def _is_refusal(text: str) -> bool:
    """Check if the predicted answer is a refusal to answer."""
    return any(marker in text.lower() for marker in REFUSAL_MARKERS)


def compute_f1(predicted: str, ground_truth: str | int) -> float:
    """Compute token-level F1 score between predicted and ground truth answers."""
    predicted = str(predicted) if not isinstance(predicted, str) else predicted
    ground_truth = (
        str(ground_truth) if not isinstance(ground_truth, str) else ground_truth
    )

    pred_tokens = set(re.findall(r"\w+", predicted.lower()))
    truth_tokens = set(re.findall(r"\w+", ground_truth.lower()))

    # Adversarial questions have empty ground truth — a refusal is correct
    if not truth_tokens:
        return 1.0 if (not pred_tokens or _is_refusal(predicted)) else 0.0

    if not pred_tokens:
        return 0.0

    common = pred_tokens & truth_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)

    return 2 * precision * recall / (precision + recall)


def evaluate_longmemeval(
    question: str,
    expected: str,
    predicted: str,
    category: str = "",
    question_id: str = "",
) -> dict:
    """LongMemEval native judge — binary accuracy via GPT-4o.

    Matches the exact prompts from the LongMemEval paper (ICLR 2025) so
    results are directly comparable to published numbers.

    Returns:
        dict with longmemeval_correct (0 or 1).
    """
    abstention = "_abs" in question_id

    if abstention:
        template = (
            "I will give you an unanswerable question, an explanation, and "
            "a response from a model. Please answer yes if the model correctly "
            "identifies the question as unanswerable. The model could say that "
            "the information is incomplete, or some other information is given "
            "but the asked information is not.\n\n"
            "Question: {q}\n\nExplanation: {a}\n\nModel Response: {r}\n\n"
            "Does the model correctly identify the question as unanswerable? "
            "Answer yes or no only."
        )
    elif category == "temporal-reasoning":
        template = (
            "I will give you a question, a correct answer, and a response from "
            "a model. Please answer yes if the response contains the correct "
            "answer. Otherwise, answer no. If the response is equivalent to the "
            "correct answer or contains all the intermediate steps to get the "
            "correct answer, you should also answer yes. If the response only "
            "contains a subset of the information required by the answer, answer "
            "no. In addition, do not penalize off-by-one errors for the number "
            "of days. If the question asks for the number of days/weeks/months, "
            "etc., and the model makes off-by-one errors (e.g., predicting 19 "
            "days when the answer is 18), the model's response is still "
            "correct. \n\nQuestion: {q}\n\nCorrect Answer: {a}\n\n"
            "Model Response: {r}\n\nIs the model response correct? "
            "Answer yes or no only."
        )
    elif category == "knowledge-update":
        template = (
            "I will give you a question, a correct answer, and a response from "
            "a model. Please answer yes if the response contains the correct "
            "answer. Otherwise, answer no. If the response contains some "
            "previous information along with an updated answer, the response "
            "should be considered as correct as long as the updated answer is "
            "the required answer.\n\nQuestion: {q}\n\nCorrect Answer: {a}\n\n"
            "Model Response: {r}\n\nIs the model response correct? "
            "Answer yes or no only."
        )
    elif category == "single-session-preference":
        template = (
            "I will give you a question, a rubric for desired personalized "
            "response, and a response from a model. Please answer yes if the "
            "response satisfies the desired response. Otherwise, answer no. "
            "The model does not need to reflect all the points in the rubric. "
            "The response is correct as long as it recalls and utilizes the "
            "user's personal information correctly.\n\n"
            "Question: {q}\n\nRubric: {a}\n\n"
            "Model Response: {r}\n\nIs the model response correct? "
            "Answer yes or no only."
        )
    else:
        # single-session-user, single-session-assistant, multi-session
        template = (
            "I will give you a question, a correct answer, and a response from "
            "a model. Please answer yes if the response contains the correct "
            "answer. Otherwise, answer no. If the response is equivalent to the "
            "correct answer or contains all the intermediate steps to get the "
            "correct answer, you should also answer yes. If the response only "
            "contains a subset of the information required by the answer, answer "
            "no. \n\nQuestion: {q}\n\nCorrect Answer: {a}\n\n"
            "Model Response: {r}\n\nIs the model response correct? "
            "Answer yes or no only."
        )

    prompt = template.format(q=question, a=expected, r=predicted)
    judge_model = os.environ.get("LONGMEMEVAL_JUDGE_MODEL", "gpt-4o")

    try:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        eval_text = response.choices[0].message.content.strip().lower()
        correct = 1 if "yes" in eval_text else 0
    except Exception as e:
        print(f"  LongMemEval judge error: {e}")
        correct = 0

    return {"longmemeval_correct": correct}


def evaluate_with_judge(question: str, expected: str, predicted: str) -> dict:
    """3-dimension LLM-as-judge evaluation (relevant / complete / accurate).

    Uses the same JUDGE_PROMPT as evaluate_retrieval so every benchmark
    (SimpleMem, Mem0, Memory-R1) produces comparable scores.

    Returns:
        dict with judge_relevant, judge_complete, judge_accurate (0 or 1)
        and judge_explanation.
    """
    # Adversarial questions have empty ground truth — a refusal is correct
    if not str(expected).strip() and _is_refusal(str(predicted)):
        return {
            "judge_relevant": 1,
            "judge_complete": 1,
            "judge_accurate": 1,
            "judge_explanation": "Correctly refused to answer unanswerable question",
        }

    try:
        result = evaluate_retrieval(question, expected, predicted)
        return {
            "judge_relevant": result.get("relevant", 0),
            "judge_complete": result.get("complete", 0),
            "judge_accurate": result.get("accurate", 0),
            "judge_explanation": result.get("explanation", ""),
        }
    except Exception as e:
        print(f"  Judge error: {e}")
        return {
            "judge_relevant": 0,
            "judge_complete": 0,
            "judge_accurate": 0,
            "judge_explanation": str(e),
        }
