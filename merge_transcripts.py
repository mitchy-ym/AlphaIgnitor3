import re
from pathlib import Path

def get_year_month(file_path: Path) -> str | None:
    """ファイル名の先頭の日付（YYYYMMDD_）から YYYYMM を抽出します。"""
    name = file_path.name
    match = re.match(r"^(\d{4})(\d{2})\d{2}_", name)
    if match:
        return match.group(1) + match.group(2)
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
    transcripts_dir = Path("transcripts")
    
    if not transcripts_dir.exists():
        print("transcripts directory not found.")
        return

    # 月ごとのデータを格納する辞書
    # (channel_name, YYYYMM) -> list of (filename, cleaned_content)
    monthly_data = {}

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
        ym = get_year_month(file_path)
        if not ym:
            print(f"Skipping file (no date prefix): {file_path.name}")
            continue

        # 配信者名（親フォルダ名）の解決
        # transcripts 直下の場合は "general" とし、サブフォルダがある場合はそのフォルダ名を使う
        if file_path.parent == transcripts_dir:
            channel_name = "general"
        else:
            channel_name = file_path.parent.name

        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Failed to read {file_path.name}: {e}")
            continue

        cleaned = clean_content(content)
        
        key = (channel_name, ym)
        if key not in monthly_data:
            monthly_data[key] = []
            
        monthly_data[key].append((file_path.name, cleaned))

    # 配信者名・月ごとに結合して書き出し
    for (channel_name, ym), data_list in sorted(monthly_data.items()):
        # 保存先フォルダ: transcripts/[配信者名]_summary/
        channel_summary_dir = transcripts_dir / f"{channel_name}_summary"
        channel_summary_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = channel_summary_dir / f"{ym}.txt"
        
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
