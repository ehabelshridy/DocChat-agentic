"""
llm.py

All calls to OpenAI's API live here: the embedding call
(shared with ingest_pipeline.py) and the chat completion call used by
every LLM-powered node in the graph (grade_relevance, rewrite_query,
generate_answer, check_groundedness).

Models used:
  - Chat:       gpt-4o-mini
  - Embeddings: text-embedding-3-small (1536 dimensions)

Prompt Caching
--------------
OpenAI automatically caches the PREFIX of prompts longer than 1024
tokens, saving 50% on cached input token costs. The cache key is the
exact byte sequence of the prompt prefix, so:

  1. The system prompt is placed in a dedicated "system" message and
     kept IDENTICAL across every call that uses the same node prompt
     (RELEVANCE_SYSTEM_PROMPT, GROUNDEDNESS_SYSTEM_PROMPT, etc.).
     This is already true because these constants live in graph.py and
     are never modified at runtime.

  2. The system message is always sent FIRST so it forms the cacheable
     prefix. The user message (which changes every call) comes after.

  3. We use the `cached_tokens` field in usage_metadata to log how
     many tokens were served from cache -- useful for verifying that
     caching is active.

No extra API parameters are needed: caching is automatic on
gpt-4o-mini as long as the prefix is >= 1024 tokens and identical
across calls. For shorter system prompts (< 1024 tokens), OpenAI will
not cache them, but the code is still correct -- it just won't log
cache hits.

Centralizing this means graph.py never touches OPENAI_API_KEY or the
HTTP client directly -- it just calls chat_completion(...) and embed_query(...).
"""

import json
import logging
import os
import re
import sys
import time
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger(__name__)

CHAT_MODEL_NAME      = "gpt-4o-mini"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    """Lazily build a single shared OpenAI client for the process."""
    global _client
    if _client is not None:
        return _client

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(
            "ERROR: OPENAI_API_KEY not found. Create a .env file with:\n"
            "  OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
            "Get a key at https://platform.openai.com/api-keys"
        )
        sys.exit(1)

    _client = OpenAI(api_key=api_key)
    return _client


def embed_query(
    text: str,
    model_name: str = EMBEDDING_MODEL_NAME,
    max_retries: int = 3,
) -> List[float]:
    """Embed a single query string using OpenAI's embedding API.
    Same model used during ingestion so the query lands in the same
    vector space as the stored chunks.
    """
    client = get_client()
    for attempt in range(1, max_retries + 1):
        try:
            response = client.embeddings.create(
                model=model_name,
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            if attempt == max_retries:
                raise RuntimeError(
                    f"Embedding failed after {max_retries} attempts: {e}"
                ) from e
            time.sleep(2 ** attempt)


def chat_completion(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.1,
    max_retries: int = 3,
) -> str:
    """Single chat call via OpenAI API with automatic Prompt Caching.

    Prompt Caching works automatically when:
      - The model is gpt-4o-mini (or gpt-4o / o-series).
      - The prompt prefix is >= 1024 tokens.
      - The system message content is IDENTICAL across calls.

    The system prompt is always placed first (as the "system" role
    message) so it forms the cacheable prefix. The user message, which
    changes every call, is appended after and does NOT affect the cache
    key for the system prefix.

    Cache hits are logged at DEBUG level showing how many input tokens
    were served from cache (50% cheaper than normal input tokens).
    """
    client = get_client()
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=CHAT_MODEL_NAME,
                messages=[
                    # System message first -- this is the cacheable prefix.
                    # Keep its content identical across calls that share the
                    # same logical prompt (already guaranteed because callers
                    # pass the module-level constants from graph.py).
                    {"role": "system", "content": system_prompt},
                    # User message after -- changes every call, not cached.
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )

            # Log cache usage when available (gpt-4o-mini returns this
            # in response.usage.prompt_tokens_details).
            usage = getattr(response, "usage", None)
            if usage:
                details = getattr(usage, "prompt_tokens_details", None)
                cached  = getattr(details, "cached_tokens", 0) if details else 0
                total   = getattr(usage, "prompt_tokens", "?")
                if cached:
                    logger.debug(
                        "Prompt cache HIT: %d/%s input tokens served from cache "
                        "(saved ~50%% on those tokens).",
                        cached, total,
                    )
                else:
                    logger.debug(
                        "Prompt cache MISS: %s input tokens (prefix will be "
                        "cached for subsequent identical calls).",
                        total,
                    )

            return response.choices[0].message.content.strip()

        except Exception as e:
            if attempt == max_retries:
                raise RuntimeError(
                    f"Chat completion failed after {max_retries} attempts: {e}"
                ) from e
            wait = 2 ** attempt
            logger.warning("OpenAI call failed (attempt %d/%d): %s — retrying in %ds",
                           attempt, max_retries, e, wait)
            time.sleep(wait)


def parse_json_response(raw: str) -> dict:
    """LLMs (even when told to return only JSON) sometimes wrap it in
    markdown fences or add a sentence before/after. This pulls out the
    first {...} block and parses that, instead of assuming raw is
    already clean JSON.
    """
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$",          "", cleaned).strip()

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
    lowered = cleaned.lower()
    return {"decision": "true" in lowered or "yes" in lowered, "reasoning": raw}
