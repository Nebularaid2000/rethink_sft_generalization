QA_INSTURCTIONS = """I want you act as an answer judge. Given a question and an answer, your objective is to determine if the provided answer contains non-factual or hallucinated information. You SHOULD give your judgement based on the following hallucination types and the world knowledge.
You should try your best to determine if the answer contains non-factual or hallucinated information. The answer you give MUST be \"Yes\" or \"No\""."""

DIALOGUE_INSTRUCTIONS = """I want you act as a response judge. Given a dialogue history and a response, your objective is to determine if the provided response contains non-factual or hallucinated information. You SHOULD give your judgement based on the following hallucination types and the world knowledge.
You should try your best to determine if the response contains non-factual or hallucinated information. The answer you give MUST be \"Yes\" or \"No\""."""

SUMMARIZATION_INSTRUCTIONS = """I want you act as a summary judge. Given a document and a summary, your objective is to determine if the provided summary contains non-factual or hallucinated information. You SHOULD give your judgement based on the following hallucination types and the world knowledge.
You should try your best to determine if the summary contains non-factual or hallucinated information. The answer you give MUST be \"Yes\" or \"No\""."""


def doc_to_text_qa(doc: dict[str, str]) -> str:
    doc_text = QA_INSTURCTIONS + "\n\n#Knowledge: " + doc["knowledge"] + "\n#Question#: " + doc["question"] + "\n#Answer#: " + doc["answer"] + "\n#Your Judgement#:"
    return doc_text


def doc_to_text_dialogue(doc: dict[str, str]) -> str:
    doc_text = DIALOGUE_INSTRUCTIONS + "\n\n#Knowledge: " + doc["knowledge"] + "\n#Dialogue History#: " + doc["dialogue_history"] + "\n#Response#: " + doc["response"] + "\n#Your Judgement#:"
    return doc_text


def doc_to_text_summarization(doc: dict[str, str]) -> str:
    doc_text_1 = SUMMARIZATION_INSTRUCTIONS + "\n\n#Document#: " + doc["document"]
    doc_text_2 = "\n#Summary#: " + doc["summary"] + "\n#Your Judgement#:"
    doc_text = doc_text_1 + doc_text_2
    return doc_text


def doc_to_target(doc: dict[str, str]) -> str:
    return doc['hallucination']


def compute_metrics(gold_answer: str, prediction: str) -> dict[str, float]:
    is_correct = True

    if ("yes" in prediction and "no" in prediction) or ("yes" not in prediction and "no" not in prediction):
        is_correct = False
    elif "yes" in prediction:
        prediction = "yes"
    elif "no" in prediction:
        prediction = "no"

    is_exact = gold_answer == prediction

    res = {"acc": 1.0 if (is_correct and is_exact) else 0.0}

    return res


import re

def process_results(doc: dict[str, str], results: list[str]):
    gold_list = doc_to_target(doc)
    response = results[0]
    
    # Remove all <think>...</think> sections.
    response_clean = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL).strip()
    # Note: `.*` is greedy and matches until the last closing think tag.
    response_clean = re.sub(r"^.*</think>", "", response_clean, flags=re.DOTALL).lstrip()
    response_clean = re.sub(r"<think>.*?<\|/think\|>\s*", "", response_clean, flags=re.DOTALL).strip()
    response_clean = re.sub(r"^.*<\|/think\|>", "", response_clean, flags=re.DOTALL).lstrip()
    response_clean = response_clean.strip()
    
    # Extract all Yes/No tokens.
    matches = re.findall(r'\b(Yes|No)\b', response_clean, re.IGNORECASE)
    
    if matches:
        prediction = matches[-1].lower()
    else:
        # Fall back to the original behavior: use the first line.
        prediction = response.strip().split("\n")[0].lower()
    
    scores = compute_metrics(gold_list, prediction)
    return scores
