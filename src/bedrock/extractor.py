from __future__ import annotations

import json
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import boto3
from botocore.exceptions import ClientError


_EXT_TO_BEDROCK_DOC_FORMAT = {
    ".pdf": "pdf",
    ".csv": "csv",
    ".doc": "doc",
    ".docx": "docx",
    ".xls": "xls",
    ".xlsx": "xlsx",
    ".html": "html",
    ".htm": "html",
    ".txt": "txt",
    ".md": "md",
}


@dataclass(frozen=True)
class BedrockExtractResult:
    raw_text: str
    parsed_json: Optional[dict[str, Any]]
    model_id: str
    usage: Optional[dict[str, Any]]
    stop_reason: Optional[str]


def _guess_document_format(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in _EXT_TO_BEDROCK_DOC_FORMAT:
        return _EXT_TO_BEDROCK_DOC_FORMAT[ext]
    raise ValueError(
        f"Unsupported file extension: {ext}. Supported: {sorted(_EXT_TO_BEDROCK_DOC_FORMAT)}"
    )


def _safe_json_loads(text: str) -> Optional[dict[str, Any]]:
    """Best-effort JSON parse.

    The model *should* return JSON-only, but in practice you might get:
    - leading/trailing whitespace
    - fenced blocks ```json ... ```
    - a short preamble (oops)
    This tries to recover without being too clever.
    """
    candidate = text.strip()

    # Strip common fenced block wrappers
    if candidate.startswith("```"):
        candidate = candidate.strip("`\n ")
        # if it started with ```json
        candidate = candidate.replace("json\n", "", 1).strip()

    # Try direct parse
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Try extracting first top-level JSON object in the string
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = candidate[start : end + 1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None

    return None


def extract_structured_event_history(
    *,
    file_path: str | Path,
    prompt: str,
    model_id: str,
    region: str,
    max_tokens: int = 2000,
    temperature: float = 0.0,
    top_p: float = 0.9,
) -> BedrockExtractResult:
    """Send a document + prompt to Bedrock Converse and return raw + parsed JSON."""
    p = Path(file_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)

    doc_format = _guess_document_format(p)
    doc_bytes = p.read_bytes()

    client = boto3.client("bedrock-runtime", region_name=region)

    messages = [
        {
            "role": "user",
            "content": [
                {"text": prompt},
                {
                    "document": {
                        "format": doc_format,
                        "name": p.name,
                        "source": {"bytes": doc_bytes},
                        # citations are optional; leaving disabled by default keeps output cleaner
                        # "citations": {"enabled": False},
                    }
                },
            ],
        }
    ]

    try:
        resp = client.converse(
            modelId=model_id,
            messages=messages,
            inferenceConfig={
                "maxTokens": max_tokens,
                "temperature": temperature,
                "topP": top_p,
            },
        )
    except ClientError as e:
        raise RuntimeError(f"Bedrock Converse call failed: {e}") from e

    # Bedrock returns content blocks for the assistant; stitch text blocks together
    out_text_parts: list[str] = []
    for block in resp.get("output", {}).get("message", {}).get("content", []) or []:
        if "text" in block and block["text"] is not None:
            out_text_parts.append(str(block["text"]))
    raw_text = "\n".join(out_text_parts).strip()

    parsed = _safe_json_loads(raw_text)

    usage = resp.get("usage")
    stop_reason = resp.get("stopReason")

    return BedrockExtractResult(
        raw_text=raw_text,
        parsed_json=parsed,
        model_id=model_id,
        usage=usage,
        stop_reason=stop_reason,
    )
