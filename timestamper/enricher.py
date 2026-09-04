import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .utils import extract_timestamp_and_text, log_info, log_success, log_warn


THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


@dataclass
class Chapter:
    start: str
    title: str


@dataclass
class ChunkResult:
    summary: str
    chapters: list[Chapter]


def _strip_reasoning_and_fences(text: str) -> str:
    text = THINK_RE.sub("", text).strip()
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def _call_chat_completion(
    endpoint: str,
    model: str,
    messages: list[dict[str, str]],
    timeout: float,
    max_tokens: int,
    api_key_env: str | None = None,
    schema: dict | None = None,
) -> str:
    """Ollama/OpenAI互換エンドポイントへ Chat Completion リクエストを送信し、テキスト内容を返します。"""
    payload: dict = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens": max_tokens,
        "reasoning_effort": "none",
    }
    if schema is not None:
        payload["response_format"] = {"type": "json_schema", "schema": schema}

    headers = {"Content-Type": "application/json"}
    if api_key_env:
        api_key = os.environ.get(api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM server returned HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to connect to Ollama: {exc.reason}") from exc

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("LLM response did not include choices")

    message = choices[0].get("message") or {}
    content = message.get("content") or choices[0].get("text")
    if not content:
        raise RuntimeError("LLM response did not include text content")

    return str(content).strip()


def _post_chat_completion(
    endpoint: str,
    model: str,
    messages: list[dict[str, str]],
    schema: dict,
    timeout: float,
    max_tokens: int,
    api_key_env: str | None,
) -> dict:
    content = _call_chat_completion(
        endpoint=endpoint,
        model=model,
        messages=messages,
        timeout=timeout,
        max_tokens=max_tokens,
        api_key_env=api_key_env,
        schema=schema,
    )
    try:
        return json.loads(_strip_reasoning_and_fences(content))
    except Exception as exc:
        raise RuntimeError(f"LLM response was not valid JSON: {content[:300]!r}") from exc


def _split_transcript(text: str, max_chars: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in text.splitlines():
        line_len = len(line) + 1
        if current and current_len + line_len > max_chars:
            chunks.append("\n".join(current).strip())
            current = []
            current_len = 0
        current.append(line)
        current_len += line_len
    if current:
        chunks.append("\n".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def _split_chunk_for_retry(chunk: str) -> list[str]:
    lines = [line for line in chunk.splitlines() if line.strip()]
    if len(lines) < 2:
        return [chunk]
    midpoint = len(lines) // 2
    left = "\n".join(lines[:midpoint]).strip()
    right = "\n".join(lines[midpoint:]).strip()
    return [part for part in (left, right) if part]


def _target_paths(output_root: Path, transcript_path: Path) -> tuple[Path, Path, Path]:
    channel_dir = output_root / transcript_path.parent.name
    clean_path = channel_dir / "clean" / f"{transcript_path.stem}.txt"
    summary_path = channel_dir / "summary" / f"{transcript_path.stem}.md"
    chapters_path = channel_dir / "chapters" / f"{transcript_path.stem}.md"
    return clean_path, summary_path, chapters_path


def _normalize_clean_chunk(chunk: str) -> str:
    normalized_lines: list[str] = []
    for raw_line in chunk.splitlines():
        line = raw_line.rstrip()
        extracted = extract_timestamp_and_text(line)
        if not extracted:
            if line.strip():
                normalized_lines.append(line.strip())
            continue
        timestamp, text = extracted
        normalized_lines.append(f"[{timestamp}] {text}")
    return "\n".join(normalized_lines).strip()


def _validate_summary(summary_text: str) -> str:
    summary_text = _strip_reasoning_and_fences(summary_text)
    if not summary_text:
        raise RuntimeError("summary was empty")
    if "<think>" in summary_text.lower():
        raise RuntimeError("summary leaked think tags")
    return summary_text


def _parse_chapters(raw_chapters: object) -> list[Chapter]:
    if raw_chapters in (None, ""):
        return []
    if not isinstance(raw_chapters, list):
        raise RuntimeError("chapters must be a list")

    chapters: list[Chapter] = []
    for item in raw_chapters:
        if not isinstance(item, dict):
            raise RuntimeError("chapter items must be objects")
        start = str(item.get("start", "")).strip().strip("[]")
        title = str(item.get("title", "")).strip()
        if not re.fullmatch(r"\d{2}:\d{2}:\d{2}", start):
            raise RuntimeError(f"invalid chapter timestamp: {start!r}")
        if not title:
            raise RuntimeError("chapter title was empty")
        chapters.append(Chapter(start=start, title=title))
    return chapters


def _dedupe_chapters(chapters: list[Chapter]) -> list[Chapter]:
    deduped: list[Chapter] = []
    seen: set[tuple[str, str]] = set()
    for chapter in chapters:
        key = (chapter.start, chapter.title)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(chapter)
    return deduped


def _chunk_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "chapters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "string"},
                        "title": {"type": "string"},
                    },
                    "required": ["start", "title"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["summary", "chapters"],
        "additionalProperties": False,
    }


def _build_chunk_messages(chunk: str, index: int, total: int) -> list[dict[str, str]]:
    system_prompt = (
        "あなたは日本語配信の文字起こし編集者です。"
        "内容を捏造せず、与えられたタイムスタンプ行をそのまま保った整形結果を返してください。"
    )
    user_prompt = f"""
/no_think

次の文字起こしチャンク {index}/{total} を JSON で処理してください。

制約:
- summary はこのチャンクの要点だけを日本語で簡潔に書く
- chapters は重要な話題転換だけを配列で返す。start は HH:MM:SS 形式、title は短く簡潔にする

補足:
- clean はこちらで元のタイムスタンプ付き行をそのまま保持するので、あなたは summary と chapters のみを返す

文字起こし:
{chunk}
""".strip()
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _build_summary_messages(chunk_summaries: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "あなたは日本語配信アーカイブの要約編集者です。考察や思考過程を出さず、要約だけを返します。",
        },
        {
            "role": "user",
            "content": (
                "/no_think\n\n"
                "以下はチャンクごとの要約です。重複を取り除き、事実だけを残した全体要約を日本語で簡潔に返してください。\n\n"
                f"{chunk_summaries}"
            ),
        },
    ]


def _request_chunk_result(chunk: str, args: argparse.Namespace, index: int, total: int) -> ChunkResult:
    return _request_chunk_result_inner(chunk, args, index, total, depth=0)


def _merge_chunk_results(results: list[ChunkResult]) -> ChunkResult:
    merged_summaries: list[str] = []
    merged_chapters: list[Chapter] = []
    for result in results:
        if result.summary:
            merged_summaries.append(result.summary)
        merged_chapters.extend(result.chapters)
    return ChunkResult(
        summary="\n\n".join(merged_summaries).strip(),
        chapters=merged_chapters,
    )


def _request_chunk_result_inner(
    chunk: str,
    args: argparse.Namespace,
    index: int,
    total: int,
    depth: int,
) -> ChunkResult:
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            data = _post_chat_completion(
                endpoint=args.llm_endpoint,
                model=args.llm_model,
                messages=_build_chunk_messages(chunk, index, total),
                schema=_chunk_schema(),
                timeout=args.llm_timeout,
                max_tokens=args.llm_max_tokens,
                api_key_env=getattr(args, "llm_api_key_env", None),
            )
            summary = _validate_summary(str(data.get("summary", "")).strip())
            chapters = _parse_chapters(data.get("chapters", []))
            return ChunkResult(summary=summary, chapters=chapters)
        except Exception as exc:
            last_error = exc
            if attempt == 0:
                log_warn(f"LLM応答の解析に失敗しました。再試行します: chunk {index}/{total} ({exc})")

    retry_parts = _split_chunk_for_retry(chunk)
    if len(retry_parts) >= 2 and depth < 3:
        log_warn(f"LLM応答が不安定なため、chunk {index}/{total} をさらに分割して再試行します。")
        subresults = [
            _request_chunk_result_inner(part, args, index, total, depth=depth + 1)
            for part in retry_parts
        ]
        return _merge_chunk_results(subresults)

    raise RuntimeError(f"LLM応答の解析に失敗しました: chunk {index}/{total} ({last_error})")


def _request_final_summary(summary_parts: list[str], args: argparse.Namespace) -> str:
    if len(summary_parts) == 1:
        return _validate_summary(summary_parts[0])
    content = _call_chat_completion(
        endpoint=args.llm_endpoint,
        model=args.llm_model,
        messages=_build_summary_messages("\n\n".join(summary_parts)),
        timeout=args.llm_timeout,
        max_tokens=args.llm_max_tokens,
        api_key_env=getattr(args, "llm_api_key_env", None),
    )
    return _validate_summary(content)


def enrich_transcript_file(transcript_path: Path, output_root: Path, args: argparse.Namespace) -> bool:
    clean_path, summary_path, chapters_path = _target_paths(output_root, transcript_path)
    if not getattr(args, "force", False) and clean_path.exists() and summary_path.exists() and chapters_path.exists():
        log_info(f"enrich済みのためスキップします: {transcript_path.name}")
        return False

    text = transcript_path.read_text(encoding="utf-8")
    chunks = _split_transcript(text, getattr(args, "max_chars", 1500))
    if not chunks:
        raise RuntimeError(f"Transcript was empty: {transcript_path}")

    clean_parts: list[str] = []
    summary_parts: list[str] = []
    chapters: list[Chapter] = []
    started_at = time.time()

    for index, chunk in enumerate(chunks, start=1):
        log_info(f"LLM enrich中: {transcript_path.name} ({index}/{len(chunks)})")
        clean_parts.append(_normalize_clean_chunk(chunk))
        result = _request_chunk_result(chunk, args, index, len(chunks))
        summary_parts.append(result.summary)
        chapters.extend(result.chapters)

    final_summary = _request_final_summary(summary_parts, args)
    final_chapters = _dedupe_chapters(chapters)

    for path in (clean_path, summary_path, chapters_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    clean_path.write_text("\n".join(part.strip() for part in clean_parts if part.strip()) + "\n", encoding="utf-8")
    summary_path.write_text(final_summary.strip() + "\n", encoding="utf-8")
    chapters_path.write_text(
        "\n".join(f"[{chapter.start}] {chapter.title}" for chapter in final_chapters).strip() + "\n",
        encoding="utf-8",
    )

    log_success(f"enrich完了: {transcript_path.name} ({time.time() - started_at:.2f}秒)")
    return True