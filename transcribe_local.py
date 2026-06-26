#!/usr/bin/env python
import argparse
import sys
import os
from pathlib import Path

# ローカルモジュールへのインポートを可能にする
sys.path.append(str(Path(__file__).parent))

# バックワードコンパティビティ（後方互換性）のためにモジュールレベルで関数を公開
from timestamper.transcriber import (
    load_transcribe_cache as load_cache,
    save_transcribe_cache as save_cache,
    resolve_input_media,
    select_device,
    load_whisper_model,
    build_decode_options,
    detect_silent_chunks,
    build_output_paths,
    write_outputs,
    transcribe_file,
    run_transcribe_worker
)
from timestamper.utils import (
    MEDIA_EXTENSIONS,
    format_seconds,
    get_media_duration
)

def main() -> int:
    from timestamper.cli import add_transcribe_arguments
    parser = argparse.ArgumentParser(
        description="Local media transcription script optimized for AMD ROCm GPU using faster-whisper."
    )
    add_transcribe_arguments(parser)
    args = parser.parse_args()
    return run_transcribe_worker(args)

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)