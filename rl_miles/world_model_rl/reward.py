"""LLM-as-a-judge reward for the single-turn language world model.

This keeps the same 6-dimension judge format used by the existing world-model
evaluation stack:

* predicted observation vs. ground-truth observation;
* six dimensions: format, factuality, consistency, realism, quality, efficiency;
* remote OpenAI-compatible judge endpoint, default DeepSeek-V4-Flash-0731;
* module/shared-launcher defaults, also used by the 4B, 9B, 27B, and
  35B-A3B 128-GPU wrappers: judge weight 1.0, similarity weight 0.0,
  and similarity disabled;
* the 122B-A10B 160-GPU wrapper enables similarity and sets weights 0.9/0.1;
  with both scorers successful, its default base reward is
  ``0.9 * judge_reward + 0.1 * similarity_score``;
* ``WM_JUDGE_WEIGHT`` / ``WM_SIMILARITY_WEIGHT`` override the weights, and
  ``WM_SIMILARITY_ENABLED`` controls whether similarity is used. Weights
  are individually clamped to [0, 1], not normalized to sum to 1;
* with remote judging and similarity enabled, a failed scorer falls back
  to the other scorer's unweighted score; if neither succeeds, local
  rule-based scoring is used. The thinking-length penalty applies afterward.

The miles rollout code invokes this module through::

    --custom-rm-path world_model_rl.reward.custom_rm

Configuration is passed through environment variables so it survives the
Ray job runtime environment.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import weakref
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _disable_http_proxy_env() -> None:
    """Prevent corporate proxy variables from being used by HTTP clients.

    Judge and similarity traffic should always connect directly to the
    in-cluster OpenAI-compatible services.  httpx clients below are also
    created with ``trust_env=False``; this explicit scrub makes the behavior
    obvious and protects any other HTTP stack used in this process.
    """
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"


_disable_http_proxy_env()

DEFAULT_BASE_URL = "http://<JUDGE_HOST>:<JUDGE_PORT>/v1"
DEFAULT_MODEL = "DeepSeek-V4-Flash-0731"
DEFAULT_API_KEY = "EMPTY"
DEFAULT_MAX_TOKENS = 8192
DEFAULT_CONCURRENCY = 128
DEFAULT_TIMEOUT = 300.0
DEFAULT_RETRIES = 3
DEFAULT_JUDGE_ENABLED = True
DEFAULT_SIMILARITY_URL = "http://<SIMILARITY_HOST>:<SIMILARITY_PORT>/similarity"
DEFAULT_SIMILARITY_TIMEOUT = 60.0
DEFAULT_JUDGE_WEIGHT = 1.0
DEFAULT_SIMILARITY_WEIGHT = 0.0
DEFAULT_SIMILARITY_ENABLED = False
DEFAULT_THINK_MODE = False

CORE_DIMENSIONS = ["format", "factuality", "consistency", "realism", "quality", "efficiency"]
DEFAULT_WEIGHTS = {"format": 1, "factuality": 2, "consistency": 2, "realism": 2, "quality": 2, "efficiency": 1}
OBS_MARKER = "###[Observation]"
THINK_LENGTH_PENALTY_CHARS = 10000
THINK_LENGTH_PENALTY_FACTOR = 0.9

_RL_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_PROMPT_ROOT = _RL_ROOT.parent / "prompts"
_PROMPT_ROOT = Path(os.environ.get("WM_JUDGE_PROMPT_DIR", str(_DEFAULT_PROMPT_ROOT)))
_JUDGE_SYSTEM_PROMPT_PATHS = {
    "android": _PROMPT_ROOT / "android" / "judge_system_prompt.txt",
    "mcp": _PROMPT_ROOT / "mcp" / "judge_system_prompt.txt",
    "os": _PROMPT_ROOT / "os" / "judge_system_prompt.txt",
    "search": _PROMPT_ROOT / "search" / "judge_system_prompt.txt",
    "swe": _PROMPT_ROOT / "swe" / "judge_system_prompt.txt",
    "terminal": _PROMPT_ROOT / "terminal" / "judge_system_prompt.txt",
    "web": _PROMPT_ROOT / "web" / "judge_system_prompt.txt",
}
_JUDGE_USER_TEMPLATE_PATH = _PROMPT_ROOT / "judge_user_prompt.txt"

_JUDGE_RESOURCES: "weakref.WeakKeyDictionary[Any, Dict[str, tuple]]" = weakref.WeakKeyDictionary()
_JUDGE_SYSTEM_PROMPTS: Optional[Dict[str, str]] = None
_JUDGE_WEIGHTS: Dict[str, float] = DEFAULT_WEIGHTS
_SIMILARITY_RESOURCES: "weakref.WeakKeyDictionary[Any, tuple]" = weakref.WeakKeyDictionary()


def _remove_thinking_tags(text: str, response_tag: str = "") -> str:
    if not text:
        return text

    tags = []
    for match in re.finditer(r"<think>", text, re.IGNORECASE):
        tags.append((match.start(), "open"))
    for match in re.finditer(r"</think>", text, re.IGNORECASE):
        tags.append((match.start(), "close"))

    if not tags:
        return text

    tags.sort(key=lambda item: item[0])
    ranges_to_remove = []
    used_indices = set()

    for index, (position, tag_type) in enumerate(tags):
        if index in used_indices or tag_type != "open":
            continue
        for candidate_index in range(index + 1, len(tags)):
            if candidate_index in used_indices:
                continue
            candidate_position, candidate_type = tags[candidate_index]
            if candidate_type == "close":
                ranges_to_remove.append((position, candidate_position + len("</think>")))
                used_indices.add(index)
                used_indices.add(candidate_index)
                break

    for index, (position, tag_type) in enumerate(tags):
        if index in used_indices or tag_type != "close":
            continue
        ranges_to_remove.append((0, position + len("</think>")))
        used_indices.add(index)

    for index, (position, tag_type) in enumerate(tags):
        if index in used_indices or tag_type != "open":
            continue
        end_pos = len(text)
        if response_tag:
            response_pattern = rf"<{re.escape(response_tag)}>"
            response_match = re.search(response_pattern, text[position:], re.IGNORECASE)
            if response_match:
                end_pos = position + response_match.start()
        ranges_to_remove.append((position, end_pos))
        used_indices.add(index)

    if not ranges_to_remove:
        return text

    ranges_to_remove.sort()
    merged_ranges = []
    for start, end in ranges_to_remove:
        if merged_ranges and start <= merged_ranges[-1][1]:
            merged_ranges[-1] = (merged_ranges[-1][0], max(merged_ranges[-1][1], end))
        else:
            merged_ranges.append((start, end))

    result_parts = []
    previous_end = 0
    for start, end in merged_ranges:
        if start > previous_end:
            result_parts.append(text[previous_end:start])
        previous_end = end
    if previous_end < len(text):
        result_parts.append(text[previous_end:])
    return "".join(result_parts).strip()


def parse_model_output(raw_output: str, response_tag: str) -> str:
    if not raw_output:
        return "No output"

    cleaned_text = _remove_thinking_tags(raw_output, response_tag)
    start_pattern = rf"<{re.escape(response_tag)}>"
    start_matches = list(re.finditer(start_pattern, cleaned_text, re.IGNORECASE))
    if not start_matches:
        return cleaned_text.strip() if cleaned_text.strip() else "No output"

    last_start_match = start_matches[-1]
    start_pos = last_start_match.end()
    close_pattern = rf"</{re.escape(response_tag)}>"
    close_match = re.search(close_pattern, cleaned_text[start_pos:], re.IGNORECASE)
    if close_match:
        end_pos = start_pos + close_match.start()
        return cleaned_text[start_pos:end_pos].strip()
    return cleaned_text[start_pos:].strip()


def _extract_json_object(text: str) -> str | None:
    code_blocks = list(re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE))
    for match in reversed(code_blocks):
        candidate = match.group(1).strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            return candidate

    end = text.rfind("}")
    if end == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(end, -1, -1):
        char = text[index]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "}":
            depth += 1
        elif char == "{":
            depth -= 1
            if depth == 0:
                return text[index : end + 1]
    return None


def _coerce_score(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        score = int(value)
    elif isinstance(value, str):
        match = re.search(r"[1-5]", value)
        if not match:
            return None
        score = int(match.group(0))
    else:
        return None
    return score if 1 <= score <= 5 else None


def _parse_judge_output(raw_output: str) -> Dict[str, Any]:
    json_text = _extract_json_object(raw_output)
    if not json_text:
        return {"success": False, "parse_error": "no_json_object", "judge_raw_output": raw_output}

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as exc:
        return {"success": False, "parse_error": f"json_decode_error: {exc}", "judge_raw_output": raw_output}

    scores = parsed.get("scores") if isinstance(parsed, dict) else None
    if not isinstance(scores, dict):
        return {
            "success": False,
            "parse_error": "missing_or_invalid_scores",
            "parsed_json": parsed,
            "judge_raw_output": raw_output,
        }

    coerced_scores = {
        dimension: score
        for dimension, score in ((key, _coerce_score(value)) for key, value in scores.items())
        if score is not None
    }
    if not coerced_scores:
        return {
            "success": False,
            "parse_error": "missing_or_invalid_scores",
            "parsed_json": parsed,
            "judge_raw_output": raw_output,
        }

    return {
        "success": True,
        "strengths": parsed.get("strengths", []) if isinstance(parsed, dict) else [],
        "weaknesses": parsed.get("weaknesses", []) if isinstance(parsed, dict) else [],
        "scores": coerced_scores,
        "judge_raw_output": raw_output,
    }


def _parse_weights(raw: Optional[str]) -> Dict[str, float]:
    if not raw:
        return dict(DEFAULT_WEIGHTS)
    if raw.strip().startswith("{"):
        parsed = json.loads(raw)
        return {k: float(v) for k, v in parsed.items() if k in CORE_DIMENSIONS}
    weights = dict(DEFAULT_WEIGHTS)
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        key, _, value = item.partition(":")
        key = key.strip().lower()
        if key in weights:
            weights[key] = float(value)
    return weights


def strip_observation_marker(text: str) -> str:
    if not text:
        return ""
    cleaned = parse_model_output(text, response_tag="").strip()
    cleaned = cleaned.lstrip()
    if cleaned.startswith(OBS_MARKER):
        cleaned = cleaned[len(OBS_MARKER):].lstrip(" \t\r\n")
    return cleaned


def strip_think_prefix(text: str) -> str:
    """Return only the content after the final ``</think>`` boundary.

    This is used for think-mode rollouts, where the generated response is
    ``<think>reasoning</think><observation>``. Ground truth does not contain
    the think wrapper, so only predicted responses are passed through this
    function.
    """
    if not text:
        return text
    matches = [match.start() for match in re.finditer(r"</think>", text, re.IGNORECASE)]
    if not matches:
        return text
    return text[matches[-1] + len("</think>"):].strip()


def _think_content(text: str) -> str:
    """Return the think-mode reasoning block from a generated response.

    Some rollouts emit only ``</think>`` before the observation while others
    include an explicit ``<think>`` opening tag. Prefer explicit tag bounds,
    otherwise treat the text before the final ``</think>`` as the reasoning.
    """
    if not text:
        return ""
    # Without thinking mode or a think boundary, this is an observation, not reasoning.
    if not _think_mode() and not re.search(r"</?think>", text, re.IGNORECASE):
        return ""
    start_match = re.search(r"<think>", text, re.IGNORECASE)
    start = start_match.end() if start_match else 0
    close_matches = [match.start() for match in re.finditer(r"</think>", text, re.IGNORECASE)]
    end = close_matches[-1] if close_matches else len(text)
    return text[start:end]


def _apply_think_length_penalty(reward: float, predicted: str, metadata: dict) -> float:
    """Scale down rewards whose reasoning block exceeds the char threshold."""
    think_length_chars = len(_think_content(predicted))
    penalty_applied = think_length_chars > THINK_LENGTH_PENALTY_CHARS
    adjusted_reward = (
        _clamp_unit(float(reward or 0.0) * THINK_LENGTH_PENALTY_FACTOR)
        if penalty_applied
        else _clamp_unit(reward)
    )
    _annotate_metadata(
        metadata,
        {
            "think_length_chars": think_length_chars,
            "think_length_penalty_applied": penalty_applied,
            "think_length_penalty_factor": THINK_LENGTH_PENALTY_FACTOR if penalty_applied else 1.0,
        },
    )
    return adjusted_reward


def _load_judge_system_prompts() -> Dict[str, str]:
    global _JUDGE_SYSTEM_PROMPTS
    if _JUDGE_SYSTEM_PROMPTS is not None:
        return _JUDGE_SYSTEM_PROMPTS
    prompts: Dict[str, str] = {}
    for domain, path in _JUDGE_SYSTEM_PROMPT_PATHS.items():
        try:
            prompts[domain] = path.read_text(encoding="utf-8").strip()
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Missing local judge prompt: {path}") from exc
    _JUDGE_SYSTEM_PROMPTS = prompts
    return _JUDGE_SYSTEM_PROMPTS


def _load_judge_user_template() -> str:
    return _JUDGE_USER_TEMPLATE_PATH.read_text(encoding="utf-8").strip()


def _parse_judge_output(raw_output: str) -> Dict[str, Any]:
    parsed = _parse_judge_output_minimal(raw_output)
    if parsed.get("success"):
        return parsed

    logger.warning("Judge parsing failed (%s); returning empty scores.", parsed.get("parse_error", "unknown"))
    return parsed


def _parse_judge_output_minimal(raw_output: str) -> Dict[str, Any]:
    blocks = re.findall(r"```json\s*(\{.*?\})\s*```", raw_output, flags=re.DOTALL | re.IGNORECASE)
    candidates = list(reversed(blocks))
    candidates.append(raw_output)
    for candidate in candidates:
        candidate = candidate.strip()
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            continue
        try:
            payload = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            continue
        scores = payload.get("scores")
        if isinstance(scores, dict):
            normalized: Dict[str, int] = {}
            for key in CORE_DIMENSIONS:
                if key in scores:
                    score = _coerce_score(scores[key])
                    if score is not None:
                        normalized[key] = score
            if normalized:
                return {"scores": normalized, "success": True, "judge_raw_output": raw_output}
    return {"scores": {}, "success": False, "parse_error": "no_valid_scores", "judge_raw_output": raw_output}


def _judge_enabled() -> bool:
    """Return whether the remote LLM judge should be used.

    Production RL uses the remote DeepSeek-V4-Flash-0731 judge by default.  Set
    ``WM_JUDGE_ENABLED=0`` or ``WM_JUDGE_SKIP=1`` only for local smoke tests
    when the judge endpoint is intentionally unavailable.
    """
    if os.environ.get("WM_JUDGE_SKIP", "").strip().lower() in {"1", "true", "yes"}:
        return False
    return os.environ.get("WM_JUDGE_ENABLED", str(int(DEFAULT_JUDGE_ENABLED))).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, str(int(default))).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _similarity_enabled() -> bool:
    return _env_flag("WM_SIMILARITY_ENABLED", DEFAULT_SIMILARITY_ENABLED)


def _think_mode() -> bool:
    return _env_flag("WM_THINK_MODE", DEFAULT_THINK_MODE)


def _annotate_metadata(metadata: Any, updates: Dict[str, Any]) -> None:
    if not isinstance(metadata, dict):
        return
    try:
        metadata.update(updates)
    except Exception:  # noqa: BLE001
        pass


def _ensure_sample_metadata(sample: Any) -> dict:
    metadata = getattr(sample, "metadata", None)
    if isinstance(metadata, dict):
        return metadata
    metadata = {}
    try:
        sample.metadata = metadata
    except Exception:  # noqa: BLE001
        pass
    return metadata


def _rule_based_reward(predicted: str, ground_truth: str) -> float:
    """Local fallback reward used while the remote LLM judge is unavailable.

    This is intentionally simple and deterministic: a small format bonus for
    emitting the observation marker plus token-overlap and length-ratio
    similarity between the simulated and ground-truth observations.  It keeps
    the GSPO pipeline runnable; replace it with the LLM judge for production.
    """
    predicted_clean = strip_observation_marker(predicted).strip()
    ground_truth_clean = strip_observation_marker(ground_truth).strip()
    if not predicted_clean or not ground_truth_clean:
        return 0.0

    format_bonus = 1.0 if OBS_MARKER in predicted else 0.0

    pred_tokens = set(re.findall(r"\w+", predicted_clean.lower()))
    truth_tokens = set(re.findall(r"\w+", ground_truth_clean.lower()))
    overlap = len(pred_tokens & truth_tokens) / max(1, len(truth_tokens))

    max_len = max(1, len(predicted_clean), len(ground_truth_clean))
    length_ratio = min(len(predicted_clean), len(ground_truth_clean)) / max_len

    similarity = 0.6 * overlap + 0.4 * length_ratio
    return max(0.0, min(5.0, format_bonus + 1.0 + 4.0 * similarity))


def normalize_judge_score(raw_score: float) -> float:
    """Normalize a 1..5 judge score into 0..1."""
    return max(0.0, min(1.0, (raw_score - 1.0) / 4.0))


def normalize_rule_based_reward(raw_score: float) -> float:
    """Normalize the 0..5 rule-based fallback reward into 0..1."""
    return max(0.0, min(1.0, raw_score / 5.0))


def weighted_score(scores: Dict[str, Any]) -> float:
    total = 0.0
    total_weight = 0.0
    for dimension, weight in _JUDGE_WEIGHTS.items():
        value = scores.get(dimension)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        total += value * weight
        total_weight += weight
    return total / total_weight if total_weight else 0.0


def _read_api_key(raw_value: str) -> str:
    """Return an API key from a literal value or a small env-style key file."""
    candidate = Path(raw_value).expanduser()
    if not candidate.is_file():
        return raw_value
    for line in candidate.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            _, value = line.split("=", 1)
        else:
            value = line
        value = value.strip().strip('"').strip("'")
        if value:
            return value
    return raw_value


def _parse_judge_base_urls(raw_value: Optional[str]) -> List[str]:
    if raw_value is None:
        raw_value = DEFAULT_BASE_URL
    urls = [item.strip() for item in raw_value.split(",")]
    urls = [item for item in urls if item]
    return urls or [DEFAULT_BASE_URL]


def _parse_similarity_urls(raw_value: Optional[str]) -> List[str]:
    if raw_value is None:
        raw_value = DEFAULT_SIMILARITY_URL
    urls = [item.strip() for item in raw_value.split(",")]
    urls = [item for item in urls if item]
    return urls or [DEFAULT_SIMILARITY_URL]


def _pick_judge_base_url(base_urls: List[str]) -> str:
    if len(base_urls) == 1:
        return base_urls[0]
    return random.choice(base_urls)


def _pick_similarity_url(base_urls: List[str]) -> str:
    if len(base_urls) == 1:
        return base_urls[0]
    return random.choice(base_urls)


def _pick_judge_base_url_sequence(base_urls: List[str], retries: int) -> List[str]:
    if not base_urls:
        return [DEFAULT_BASE_URL] * max(1, retries)
    if len(base_urls) == 1:
        return [base_urls[0]] * max(1, retries)

    sequence: List[str] = []
    while len(sequence) < max(1, retries):
        shuffled = random.sample(base_urls, k=len(base_urls))
        sequence.extend(shuffled)
    return sequence[: max(1, retries)]


def _pick_similarity_url_sequence(base_urls: List[str], retries: int) -> List[str]:
    if not base_urls:
        return [DEFAULT_SIMILARITY_URL] * max(1, retries)
    if len(base_urls) == 1:
        return [base_urls[0]] * max(1, retries)

    sequence: List[str] = []
    while len(sequence) < max(1, retries):
        shuffled = random.sample(base_urls, k=len(base_urls))
        sequence.extend(shuffled)
    return sequence[: max(1, retries)]


def _ensure_judge_client(base_url: str):
    global _JUDGE_RESOURCES, _JUDGE_WEIGHTS

    _JUDGE_WEIGHTS = _parse_weights(os.environ.get("WM_JUDGE_WEIGHTS"))
    loop = asyncio.get_running_loop()
    loop_resources = _JUDGE_RESOURCES.setdefault(loop, {})
    if base_url in loop_resources:
        return loop_resources[base_url]

    import httpx
    from openai import AsyncOpenAI

    model = os.environ.get("WM_JUDGE_MODEL", DEFAULT_MODEL)
    api_key = _read_api_key(os.environ.get("WM_JUDGE_API_KEY", DEFAULT_API_KEY))
    timeout = float(os.environ.get("WM_JUDGE_TIMEOUT", DEFAULT_TIMEOUT))
    concurrency = int(os.environ.get("WM_JUDGE_CONCURRENCY", DEFAULT_CONCURRENCY))

    http_client = httpx.AsyncClient(trust_env=False, timeout=timeout)
    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        max_retries=0,
        http_client=http_client,
    )
    semaphore = asyncio.Semaphore(concurrency)
    loop_resources[base_url] = (client, semaphore)
    logger.info("World-model LLM judge initialized: model=%s base_url=%s concurrency=%s", model, base_url, concurrency)
    return client, semaphore


def _clamp_unit(value: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, value))


def _get_fusion_weights() -> tuple[float, float]:
    judge_weight = _clamp_unit(os.environ.get("WM_JUDGE_WEIGHT", DEFAULT_JUDGE_WEIGHT))
    similarity_weight = _clamp_unit(os.environ.get("WM_SIMILARITY_WEIGHT", DEFAULT_SIMILARITY_WEIGHT))
    return judge_weight, similarity_weight


def _ensure_similarity_client():
    global _SIMILARITY_RESOURCES

    loop = asyncio.get_running_loop()
    if loop in _SIMILARITY_RESOURCES:
        return _SIMILARITY_RESOURCES[loop]

    import httpx

    timeout = float(os.environ.get("WM_SIMILARITY_TIMEOUT", DEFAULT_SIMILARITY_TIMEOUT))
    concurrency = int(os.environ.get("WM_SIMILARITY_CONCURRENCY", DEFAULT_CONCURRENCY))
    client = httpx.AsyncClient(trust_env=False, timeout=timeout)
    semaphore = asyncio.Semaphore(concurrency)
    _SIMILARITY_RESOURCES[loop] = (client, semaphore)
    logger.info("World-model similarity client initialized: concurrency=%s timeout=%s", concurrency, timeout)
    return client, semaphore


async def _semantic_similarity(predicted: str, ground_truth: str) -> tuple[Optional[float], Optional[str]]:
    """Query the local Qwen3 embedding similarity service.

    Returns ``(score, None)`` on success and ``(None, error)`` on failure.
    The service response already clamps cosine similarity to ``[0, 1]``, but
    this helper clamps again defensively.
    """
    text1 = strip_observation_marker(predicted).strip()
    text2 = strip_observation_marker(ground_truth).strip()
    if not text1 or not text2:
        return None, "empty similarity input"

    client, semaphore = _ensure_similarity_client()
    urls = _parse_similarity_urls(os.environ.get("WM_SIMILARITY_URL", DEFAULT_SIMILARITY_URL))
    retries = max(1, int(os.environ.get("WM_SIMILARITY_RETRIES", DEFAULT_RETRIES)))
    url_sequence = _pick_similarity_url_sequence(urls, retries)
    payload = {"text1": text1, "text2": text2}

    for attempt, url in enumerate(url_sequence, start=1):
        async with semaphore:
            try:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "Similarity request failed (attempt=%d/%d url=%s): %s",
                    attempt,
                    retries,
                    url,
                    error,
                )
                if attempt < retries:
                    await asyncio.sleep(1.0 * attempt)
                continue
        score = data.get("similarity") if isinstance(data, dict) else None
        if score is None:
            return None, "missing similarity field"
        return _clamp_unit(score), None

    return None, "similarity request failed after retries"


def _build_messages(
    domain: str,
    predicted: str,
    ground_truth: str,
) -> List[Dict[str, str]]:
    system_prompts = _load_judge_system_prompts()
    judge_system = system_prompts.get(domain, "")
    user_template = _load_judge_user_template()
    user_prompt = user_template.format(
        predicted_observation=(
            f"**World Model Output (Simulated):**\n```\n{strip_observation_marker(predicted)}\n```"
        ),
        ground_truth=f"**Ground Truth (Real Output):**\n```\n{strip_observation_marker(ground_truth)}\n```",
    ).strip()
    return [
        {"role": "system", "content": judge_system},
        {"role": "user", "content": user_prompt},
    ]


async def _custom_rm_single(args: Any, sample: Any, **kwargs: Any) -> float:
    """Score one rollout sample against its label using the remote LLM judge."""
    predicted = getattr(sample, "response", "") or ""
    ground_truth = getattr(sample, "label", "") or ""
    metadata = _ensure_sample_metadata(sample)
    think_mode = _think_mode()
    similarity_enabled = _similarity_enabled()
    # Judge and similarity both score the final observation only. When rollout
    # think mode is enabled, generated responses start with <think>...</think>;
    # always strip that prefix before either scoring path sees the text.
    predicted_for_scoring = strip_think_prefix(predicted)

    if not predicted or not ground_truth:
        _annotate_metadata(
            metadata,
            {
                "judge_enabled": _judge_enabled(),
                "judge_skipped": True,
                "judge_scores": {},
                "judge_reason": "missing predicted response or ground-truth label",
                "think_mode": think_mode,
                "similarity_enabled": similarity_enabled,
                "similarity_score": None,
                "similarity_error": None,
            },
        )
        return 0.0

    domain = str(metadata.get("domain", "") if isinstance(metadata, dict) else "")
    if not _judge_enabled():
        rule_reward = _rule_based_reward(predicted_for_scoring, ground_truth)
        reward = normalize_rule_based_reward(rule_reward)
        _annotate_metadata(
            metadata,
            {
                "judge_enabled": False,
                "judge_skipped": True,
                "judge_reward": reward,
                "judge_rule_based_reward": rule_reward,
                "judge_scores": {},
                "judge_reason": "LLM judge disabled; using local rule-based fallback",
                "think_mode": think_mode,
                "similarity_enabled": similarity_enabled,
                "similarity_score": None,
                "similarity_error": None,
            },
        )
        return _apply_think_length_penalty(reward, predicted, metadata)

    base_urls = _parse_judge_base_urls(os.environ.get("WM_JUDGE_BASE_URL"))
    model = os.environ.get("WM_JUDGE_MODEL", DEFAULT_MODEL)
    max_tokens = int(os.environ.get("WM_JUDGE_MAX_TOKENS", DEFAULT_MAX_TOKENS))
    retries = max(1, int(os.environ.get("WM_JUDGE_RETRIES", DEFAULT_RETRIES)))
    base_url_sequence = _pick_judge_base_url_sequence(base_urls, retries)
    messages = _build_messages(domain, predicted_for_scoring, ground_truth)

    raw_output = ""
    for attempt, base_url in enumerate(base_url_sequence, start=1):
        client, semaphore = _ensure_judge_client(base_url)
        async with semaphore:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    extra_body={"enable_thinking": False},
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Judge request failed (attempt=%d/%d base_url=%s): %s: %s",
                    attempt,
                    retries,
                    base_url,
                    type(exc).__name__,
                    exc,
                )
                if attempt < retries:
                    await asyncio.sleep(1.0 * attempt)
                continue
        raw_output = response.choices[0].message.content or ""
        break

    def _fallback_reward(reason: str) -> float:
        rule_reward = _rule_based_reward(predicted_for_scoring, ground_truth)
        reward = normalize_rule_based_reward(rule_reward)
        _annotate_metadata(
            metadata,
            {
                "judge_enabled": True,
                "judge_skipped": True,
                "judge_reward": reward,
                "judge_rule_based_reward": rule_reward,
                "judge_scores": {},
                "judge_reason": reason,
                "fused_reward": reward,
                "fused_reward_formula": "rule_based_fallback",
                "think_mode": think_mode,
                "similarity_enabled": similarity_enabled,
                "similarity_score": None,
                "similarity_error": None,
            },
        )
        return _apply_think_length_penalty(reward, predicted, metadata)

    judge_success = False
    judge_reward = None
    judge_error = None
    scores: Dict[str, Any] = {}
    weighted_score_raw = None

    if not raw_output:
        judge_error = "remote judge unavailable after retries"
        if not similarity_enabled:
            return _fallback_reward(judge_error)
    else:
        parsed = _parse_judge_output(raw_output)
        if not parsed.get("success", False) or not parsed.get("scores"):
            judge_error = parsed.get("parse_error", "judge parsing failed")
            logger.warning("Judge parsing failed, raw=%s", raw_output[:300])
            if not similarity_enabled:
                return _fallback_reward(judge_error)
        else:
            scores = parsed["scores"]
            weighted_score_raw = weighted_score(scores)
            judge_reward = normalize_judge_score(weighted_score_raw)
            judge_success = True

    judge_weight, similarity_weight = _get_fusion_weights()

    if not similarity_enabled:
        fused_reward = judge_reward
        similarity_score = None
        similarity_error = None
        fused_reward_formula = "judge_reward"
        if isinstance(metadata, dict):
            try:
                metadata["judge_raw_output"] = raw_output
                metadata["judge_enabled"] = True
                metadata["judge_skipped"] = False
                metadata["judge_success"] = judge_success
                metadata["judge_scores"] = dict(scores)
                metadata["judge_weighted_score"] = weighted_score_raw
                metadata["judge_reward"] = judge_reward
                metadata["judge_error"] = judge_error
                metadata["judge_weight"] = judge_weight
                metadata["similarity_weight"] = similarity_weight
                metadata["fused_reward"] = fused_reward
                metadata["fused_reward_formula"] = fused_reward_formula
                metadata["think_mode"] = think_mode
                metadata["similarity_enabled"] = similarity_enabled
                metadata["similarity_score"] = similarity_score
                metadata["similarity_error"] = similarity_error
            except Exception:  # noqa: BLE001
                pass
        return _apply_think_length_penalty(fused_reward, predicted, metadata)

    # With similarity enabled, apply configured weights only if both scorers
    # succeed. A single successful scorer supplies its unweighted score;
    # if both fail, use the local rule-based fallback.
    similarity_score, similarity_error = await _semantic_similarity(predicted_for_scoring, ground_truth)
    similarity_success = similarity_score is not None

    if judge_success and similarity_success:
        fused_reward = _clamp_unit(judge_weight * judge_reward + similarity_weight * similarity_score)
        fused_reward_formula = "judge_weight * judge_reward + similarity_weight * similarity_score"
    elif judge_success:
        fused_reward = judge_reward
        fused_reward_formula = "judge_only_fallback"
    elif similarity_success:
        fused_reward = _clamp_unit(similarity_score)
        fused_reward_formula = "similarity_only_fallback"
    else:
        fused_reward = _fallback_reward("both judge and similarity failed")
        _annotate_metadata(
            metadata,
            {
                "judge_error": judge_error,
                "similarity_error": similarity_error,
                "judge_success": False,
                "similarity_success": False,
            },
        )
        return fused_reward

    if isinstance(metadata, dict):
        try:
            metadata["judge_raw_output"] = raw_output
            metadata["judge_enabled"] = True
            metadata["judge_skipped"] = not judge_success
            metadata["judge_success"] = judge_success
            metadata["judge_scores"] = dict(scores)
            metadata["judge_weighted_score"] = weighted_score_raw
            metadata["judge_reward"] = judge_reward
            metadata["judge_error"] = judge_error
            metadata["judge_weight"] = judge_weight
            metadata["similarity_weight"] = similarity_weight
            metadata["fused_reward"] = fused_reward
            metadata["fused_reward_formula"] = fused_reward_formula
            metadata["think_mode"] = think_mode
            metadata["similarity_enabled"] = similarity_enabled
            metadata["similarity_score"] = similarity_score
            metadata["similarity_error"] = similarity_error
            metadata["similarity_success"] = similarity_success
        except Exception:  # noqa: BLE001
            pass

    return _apply_think_length_penalty(fused_reward, predicted, metadata)


async def custom_rm(args: Any, sample: Any, **kwargs: Any) -> Any:
    """miles custom reward entry point.

    ``sample`` may be either a single ``Sample`` or a list of ``Sample``
    objects. miles uses the batched form for group reward scoring and the
    single form for normal rollout scoring. The returned value is the
    normalized judge score or the configured judge/similarity fusion, with
    the thinking-length penalty applied. Similarity is disabled by default.
    Judge failures use similarity when enabled and successful, otherwise
    local rule-based scoring; failure does not imply a fixed 0.0 reward.
    """
    if isinstance(sample, (list, tuple)):
        return list(await asyncio.gather(*(_custom_rm_single(args, item, **kwargs) for item in sample)))
    return await _custom_rm_single(args, sample, **kwargs)
