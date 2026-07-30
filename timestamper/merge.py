import argparse
import re
from pathlib import Path


def get_year_month(file_path: Path) -> str | None:
    """ファイル名の先頭の日付（YYYYMMDD_）から YYYYMM を抽出します。"""
    name = file_path.name
    match = re.match(r"^(\d{4})(\d{2})\d{2}_", name)
    if match:
        return match.group(1) + match.group(2)
    return None


def get_year(file_path: Path) -> str | None:
    """ファイル名の先頭の日付（YYYYMMDD_）から YYYY を抽出します。"""
    name = file_path.name
    match = re.match(r"^(\d{4})\d{4}_", name)
    if match:
        return match.group(1)
    return None


def clean_content(content: str) -> str:
    """各行の先頭にあるタイムスタンプ [HH:MM:SS] または [H:MM:SS] を削除します。"""
    pattern = re.compile(r"^\[\d{1,2}:\d{2}:\d{2}\]\s*")
    cleaned_lines = []
    for line in content.splitlines():
        cleaned_line = pattern.sub("", line)
        cleaned_lines.append(cleaned_line)
    return "\n".join(cleaned_lines)


def resolve_channel_name(file_path: Path, input_dir: Path) -> str:
    """入力パスの構造に応じてチャンネル名を解決します。"""
    # enriched/<channel>/clean/*.txt を直接入力にした場合
    if input_dir.name == "clean" and file_path.parent == input_dir:
        return input_dir.parent.name

    try:
        rel = file_path.relative_to(input_dir)
    except ValueError:
        return file_path.parent.name

    # input_dir 直下に txt がある場合
    if len(rel.parts) == 1:
        return "general"

    # input_dir/<channel>/*.txt 形式を想定
    return rel.parts[0]


def run_merge(args: argparse.Namespace) -> int:
    mode = args.mode

    input_dir = Path(args.input_dir)
    if args.output_dir:
        output_root = Path(args.output_dir)
    elif input_dir.name == "transcripts":
        output_root = Path("transcripts")
    elif input_dir.name == "clean" and input_dir.parent.parent != input_dir.parent:
        output_root = input_dir.parent.parent
    else:
        output_root = input_dir.parent

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        return 1

    # 集約データを格納する辞書
    # monthly: (channel_name, YYYYMM) -> list of (filename, cleaned_content)
    # yearly:  (channel_name, YYYY)   -> list of (filename, cleaned_content)
    # all:     channel_name           -> list of (filename, cleaned_content)
    aggregated_data: dict = {}

    txt_files = []
    for p in input_dir.rglob("*.txt"):
        if any("_summary" in part or part == "summary" for part in p.parts):
            continue
        txt_files.append(p)

    txt_files.sort(key=lambda x: x.name)

    print(f"Found {len(txt_files)} text files to process.")

    for file_path in txt_files:
        channel_name = resolve_channel_name(file_path, input_dir)

        if mode == "monthly":
            group_key = get_year_month(file_path)
            if not group_key:
                print(f"Skipping file (no date prefix): {file_path.name}")
                continue
            key = (channel_name, group_key)
        elif mode == "yearly":
            group_key = get_year(file_path)
            if not group_key:
                print(f"Skipping file (no date prefix): {file_path.name}")
                continue
            key = (channel_name, group_key)
        else:  # all
            key = channel_name

        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Failed to read {file_path.name}: {e}")
            continue

        merged_text = clean_content(content) if args.strip_timestamps else content

        if key not in aggregated_data:
            aggregated_data[key] = []

        aggregated_data[key].append((file_path.name, merged_text))

    for key, data_list in sorted(aggregated_data.items()):
        if mode == "all":
            channel_name = key
            output_filename = "all.txt"
        else:
            channel_name, group_key = key
            output_filename = f"{group_key}.txt"

        channel_summary_dir = output_root / f"{channel_name}_summary"
        channel_summary_dir.mkdir(parents=True, exist_ok=True)

        output_file = channel_summary_dir / output_filename

        merged_blocks = []
        for filename, content in data_list:
            block = f"[{filename}]\n{content}"
            merged_blocks.append(block)

        merged_content = "\n\n\n".join(merged_blocks) + "\n"

        try:
            output_file.write_text(merged_content, encoding="utf-8")
            print(f"Created summary file: {channel_summary_dir.name}/{output_file.name} (Contains {len(data_list)} transcripts)")
        except Exception as e:
            print(f"Failed to write {output_file.name}: {e}")

    return 0