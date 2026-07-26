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


def main():
    parser = argparse.ArgumentParser(
        description="文字起こしファイルを指定した粒度で結合します。",
    )
    parser.add_argument(
        "--mode",
        choices=["monthly", "yearly", "all"],
        default="monthly",
        help=(
            "集約モードを選択します。"
            " monthly: 月ごとに結合（デフォルト）。"
            " yearly: 年ごとに結合。"
            " all: チャンネルごとに1ファイルに結合。"
        ),
    )
    args = parser.parse_args()
    mode = args.mode

    transcripts_dir = Path("transcripts")

    if not transcripts_dir.exists():
        print("transcripts directory not found.")
        return

    # 集約データを格納する辞書
    # monthly: (channel_name, YYYYMM) -> list of (filename, cleaned_content)
    # yearly:  (channel_name, YYYY)   -> list of (filename, cleaned_content)
    # all:     channel_name           -> list of (filename, cleaned_content)
    aggregated_data: dict = {}

    # transcripts 配下の txt ファイルを再帰的に検索
    txt_files = []
    for p in transcripts_dir.rglob("*.txt"):
        # _summary を含むフォルダ（マージ先フォルダ）配下にあるファイルは除外
        # また、過去の summary フォルダ配下も除外
        if any("_summary" in part or part == "summary" for part in p.parts):
            continue
        txt_files.append(p)

    # 時系列（ファイル名順）に処理するためソート
    txt_files.sort(key=lambda x: x.name)

    print(f"Found {len(txt_files)} text files to process.")

    for file_path in txt_files:
        # 配信者名（親フォルダ名）の解決
        # transcripts 直下の場合は "general" とし、サブフォルダがある場合はそのフォルダ名を使う
        if file_path.parent == transcripts_dir:
            channel_name = "general"
        else:
            channel_name = file_path.parent.name

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

        cleaned = clean_content(content)

        if key not in aggregated_data:
            aggregated_data[key] = []

        aggregated_data[key].append((file_path.name, cleaned))

    # 集約単位ごとに結合して書き出し
    for key, data_list in sorted(aggregated_data.items()):
        if mode == "all":
            channel_name = key
            output_filename = "all.txt"
        else:
            channel_name, group_key = key
            output_filename = f"{group_key}.txt"

        # 保存先フォルダ: transcripts/[配信者名]_summary/
        channel_summary_dir = transcripts_dir / f"{channel_name}_summary"
        channel_summary_dir.mkdir(parents=True, exist_ok=True)

        output_file = channel_summary_dir / output_filename

        merged_blocks = []
        for filename, content in data_list:
            # ファイル名を [] で囲って先頭に配置し、本文を続ける
            block = f"[{filename}]\n{content}"
            merged_blocks.append(block)

        # 各ファイルのブロック間を改行で区切る
        merged_content = "\n\n\n".join(merged_blocks) + "\n"

        try:
            output_file.write_text(merged_content, encoding="utf-8")
            print(f"Created summary file: {channel_summary_dir.name}/{output_file.name} (Contains {len(data_list)} transcripts)")
        except Exception as e:
            print(f"Failed to write {output_file.name}: {e}")


if __name__ == "__main__":
    main()
