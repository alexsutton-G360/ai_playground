#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.core.config.logging import configure_logging, get_logger
from src.core.config.settings import load_env
from src.bedrock.extractor import extract_structured_event_history

log = get_logger(__name__)


def _read_text(path: str | Path) -> str:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    return p.read_text()


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Playground: send a document to Bedrock and extract event + hotel block data."
    )
    ap.add_argument("--file", required=True, help="Path to the input document (pdf/docx/xlsx/txt/etc).")

    ap.add_argument(
        "--prompt-file",
        default=str(Path("prompts") / "extract_event_history.txt"),
        help="Path to a prompt text file. Edit this to iterate quickly.",
    )

    ap.add_argument(
        "--model-id",
        default="anthropic.claude-3-5-sonnet-20240620-v1:0",
        help="Bedrock modelId (or inference profile ARN). Override if you want Haiku, etc.",
    )
    ap.add_argument(
        "--region",
        default="us-east-1",
        help="AWS region for Bedrock Runtime (must match where your model access is enabled). ",
    )

    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.9)

    ap.add_argument(
        "--out",
        default=None,
        help="Optional path to write parsed JSON (only written if JSON parsing succeeds). ",
    )
    ap.add_argument(
        "--raw-out",
        default=None,
        help="Optional path to write the raw model response text. Useful while tuning prompts.",
    )

    ap.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file (uses project-root discovery; override via ANALYTICS_AIRFLOW_ENV_FILE too). ",
    )
    ap.add_argument("--log-level", default=None, help="Override LOGGING_LEVEL.")

    return ap


def main() -> None:
    args = build_arg_parser().parse_args()

    load_env(args.env_file, override=False)
    configure_logging(level=args.log_level)

    prompt = _read_text(args.prompt_file)

    log.info("Calling Bedrock…")
    result = extract_structured_event_history(
        file_path=args.file,
        prompt=prompt,
        model_id=args.model_id,
        region=args.region,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    if args.raw_out:
        Path(args.raw_out).expanduser().write_text(result.raw_text)
        log.info("Wrote raw response: %s", args.raw_out)

    if result.parsed_json is None:
        log.warning("Could not parse JSON from model output. Printing raw output instead.\n")
        print(result.raw_text)
        return

    # Pretty print
    print(json.dumps(result.parsed_json, indent=2, sort_keys=False))

    if args.out:
        Path(args.out).expanduser().write_text(json.dumps(result.parsed_json, indent=2))
        log.info("Wrote parsed JSON: %s", args.out)

    if result.usage:
        log.info("Usage: %s", result.usage)
    if result.stop_reason:
        log.info("Stop reason: %s", result.stop_reason)


if __name__ == "__main__":
    main()
