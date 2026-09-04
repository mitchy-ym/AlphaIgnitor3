import argparse
import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import setup_parsers
from timestamper import utils
from timestamper.cli import DEFAULT_OLLAMA_ENDPOINT, DEFAULT_OLLAMA_MODEL
from timestamper.enricher import (
    Chapter,
    ChunkResult,
    _dedupe_chapters,
    _normalize_clean_chunk,
    _parse_chapters,
    _split_transcript,
    _strip_reasoning_and_fences,
    _target_paths,
)
from timestamper.merge import (
    clean_content,
    get_year,
    get_year_month,
    merge_directory,
    resolve_channel_name,
)
from timestamper.transcriber import AudioChunk, ChunkError, build_decode_options, build_output_paths
from timestamper.utils import (
    extract_timestamp_and_text,
    format_seconds,
    log_error,
    log_info,
    log_progress,
    log_success,
    log_warn,
    sanitize_channel_title,
    strip_timestamp,
)


def test_utils():
    print("Testing utils...")
    assert format_seconds(3665.0) == "01:01:05"
    assert sanitize_channel_title("Test@Channel 123", "@TestFallback") == "TestChannel 123"
    assert sanitize_channel_title("???", "@Fallback") == "Fallback"

    # Timestamp helpers
    res = extract_timestamp_and_text("[01:23:45] Hello world")
    assert res == ("01:23:45", "Hello world"), f"Unexpected: {res}"
    assert strip_timestamp("[01:23:45] Hello world") == "Hello world"
    assert strip_timestamp("[1:23:45]   Hello world") == "Hello world"
    assert strip_timestamp("No timestamp here") == "No timestamp here"

    # Logger calls (1-arg and 3-args)
    log_info("Test log_info")
    log_warn("Test log_warn")
    log_error("Test log_error single arg")
    log_error(1, 5, "Test log_error with counters")
    log_success("Test log_success single arg")
    log_success(1, 5, "Test log_success with counters")
    log_progress("Test log_progress single arg")
    log_progress(1, 5, "Test log_progress with counters")
    print("utils tests passed!")


def test_cli_parsing():
    print("Testing CLI parsing...")
    parser = setup_parsers()

    # pipeline default
    args = parser.parse_args(["pipeline", "@test"])
    assert args.enrich is True
    assert args.merge is True
    assert args.merge_mode == "yearly"
    assert args.llm_endpoint == DEFAULT_OLLAMA_ENDPOINT
    assert args.llm_model == DEFAULT_OLLAMA_MODEL

    # pipeline with --no-enrich and --no-merge
    args = parser.parse_args(["pipeline", "@test", "--no-enrich", "--no-merge"])
    assert args.enrich is False
    assert args.merge is False

    # pipeline with aliases
    args = parser.parse_args([
        "pipeline", "@test",
        "--enrich-endpoint", "http://localhost:11434/v1",
        "--enrich-model", "test-model",
        "--enrich-timeout", "120",
    ])
    assert args.llm_endpoint == "http://localhost:11434/v1"
    assert args.llm_model == "test-model"
    assert args.llm_timeout == 120.0

    # merge default
    args = parser.parse_args(["merge"])
    assert args.mode == "yearly"
    print("CLI parsing tests passed!")


def test_merge_module():
    print("Testing merge module...")
    p = Path("20240501_some_video_123.txt")
    assert get_year(p) == "2024"
    assert get_year_month(p) == "202405"

    text = "[00:01:02] First line\n[00:02:03] Second line"
    assert clean_content(text) == "First line\nSecond line"

    # Test merge_directory in a temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        channel_dir = root / "MyChannel"
        channel_dir.mkdir()
        (channel_dir / "20240101_vid1_abc.txt").write_text("[00:00:10] Happy New Year", encoding="utf-8")
        (channel_dir / "20240501_vid2_def.txt").write_text("[00:00:20] Spring stream", encoding="utf-8")
        (channel_dir / "20250101_vid3_ghi.txt").write_text("[00:00:30] Next Year", encoding="utf-8")

        ret = merge_directory(channel_dir, mode="yearly", strip_timestamps=True)
        assert ret == 0

        summary_dir = root / "MyChannel_summary"
        assert summary_dir.exists()
        assert (summary_dir / "2024.txt").exists()
        assert (summary_dir / "2025.txt").exists()

        content_2024 = (summary_dir / "2024.txt").read_text(encoding="utf-8")
        assert "Happy New Year" in content_2024
        assert "Spring stream" in content_2024
        assert "[00:00:10]" not in content_2024  # timestamps stripped
    print("Merge module tests passed!")


def test_enricher_module():
    print("Testing enricher module...")
    raw = "<think>some thought</think>```json\n{\"summary\": \"ok\"}\n```"
    assert _strip_reasoning_and_fences(raw) == '{"summary": "ok"}'

    chunk = "[00:01:00] line 1\n[00:02:00] line 2"
    assert _normalize_clean_chunk(chunk) == "[00:01:00] line 1\n[00:02:00] line 2"

    raw_chapters = [{"start": "[00:01:00]", "title": "Intro"}, {"start": "00:05:00", "title": "Main"}]
    chapters = _parse_chapters(raw_chapters)
    assert len(chapters) == 2
    assert chapters[0].start == "00:01:00"
    assert chapters[1].start == "00:05:00"

    # dedupe
    duped = chapters + [Chapter("00:01:00", "Intro")]
    assert len(_dedupe_chapters(duped)) == 2
    print("Enricher module tests passed!")


def test_transcriber_module():
    print("Testing transcriber module...")
    args = argparse.Namespace(
        task="transcribe",
        beam_size=5,
        language="ja",
        initial_prompt="hello",
        vad_threshold=0.5,
    )
    opts = build_decode_options(args)
    assert opts["task"] == "transcribe"
    assert opts["language"] == "ja"
    assert opts["vad_parameters"]["threshold"] == 0.5

    chunk = AudioChunk(index=0, start_time=0.0, duration=10.0, audio_data=None)
    assert chunk.index == 0
    err = ChunkError(ValueError("test error"))
    assert isinstance(err.exception, ValueError)
    print("Transcriber module tests passed!")


if __name__ == "__main__":
    test_utils()
    test_cli_parsing()
    test_merge_module()
    test_enricher_module()
    test_transcriber_module()
    print("\n=========================================")
    print("ALL REFACTORING UNIT TESTS PASSED SUCCESSFULLY!")
    print("=========================================")
