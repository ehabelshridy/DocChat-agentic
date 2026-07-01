"""
llm.py

All calls to HuggingFace's Inference API live here: the embedding call
(shared with ingest_pipeline.py) and the chat completion call used by
every LLM-powered node in the graph (grade_relevance, rewrite_query,
generate_answer, check_groundedness).

Centralizing this means graph.py never touches HF_TOKEN or the HTTP
client directly -- it just calls chat_completion(...) and embed_query(...).
"""

import json
import os
import re
import sys
import time
from typing import List, Optional

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

CHAT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

_client: Optional[InferenceClient] = None


def get_client() -> InferenceClient:
    """Lazily build a single shared InferenceClient for the process."""
    global _client
    if _client is not None:
        return _client

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print(
            "ERROR: HF_TOKEN not found. Create a .env file in the backend/ "
            "directory with:\n  HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        )
        sys.exit(1)

    _client = InferenceClient(token=hf_token)
    return _client


def embed_query(text: str, model_name: str, max_retries: int = 3) -> List[float]:
    """Embed a single query string. Same model/endpoint used during
    ingestion, so the query lands in the same vector space as the
    stored chunks."""
    client = get_client()
    for attempt in range(1, max_retries + 1):
        try:
            result = client.feature_extraction([text], model=model_name)
            return result[0].tolist()
        except Exception as e:
            if attempt == max_retries:
                raise RuntimeError(f"Embedding failed after {max_retries} attempts: {e}") from e
            time.sleep(2 ** attempt)


def chat_completion(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.1,
    max_retries: int = 3,
) -> str:
    """Single chat call via HF Inference API's OpenAI-compatible chat
    endpoint. Low temperature by default since every caller in this
    project (grading, rewriting, generation, groundedness checks) wants
    consistent, non-creative output rather than varied phrasing.
    """
    client = get_client()
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat_completion(
                model=CHAT_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt == max_retries:
                raise RuntimeError(f"Chat completion failed after {max_retries} attempts: {e}") from e
            time.sleep(2 ** attempt)


def parse_json_response(raw: str) -> dict:
    """LLMs (even when told to return only JSON) sometimes wrap it in
    markdown fences or add a sentence before/after. This pulls out the
    first {...} block and parses that, instead of assuming raw is
    already clean JSON.
    """
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Last resort: treat the presence of "true"/"yes" as a positive signal.
    # This keeps the graph moving instead of crashing on a malformed
    # judge response, at the cost of being a blunter signal.
    lowered = cleaned.lower()
    return {"decision": "true" in lowered or "yes" in lowered, "reasoning": raw}
