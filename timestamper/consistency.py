import argparse
from pathlib import Path
from .utils import MEDIA_EXTENSIONS

def check_consistency_report(
    args: argparse.Namespace,
    videos: list[dict],
    channel_title: str,
    downloaded_ids: set[str],
    output_dir: Path,
    t_out_dir: Path
) -> tuple[int, list[str]]:
    """YouTubeの動画リストとローカルのダウンロードキャッシュ、音声ファイル、文字起こしファイルの整合性を照合します。"""
    print(f"\n========================================================")
    print(f" 整合性確認レポート: {channel_title} ({args.channel_handle})")
    print(f"========================================================\n")

    youtube_video_ids = {v["id"] for v in videos}
    video_map = {v["id"]: v for v in videos}

    # 1. transcriptsフォルダ内のファイルを解析します
    existing_files = []
    if t_out_dir.exists():
        existing_files = [p for p in t_out_dir.iterdir() if p.is_file()]

    # 動画IDごとの文字起こしファイルの存在状況
    # v_id -> {"txt": path/None, "json": path/None}
    transcript_status = {v_id: {"txt": None, "json": None} for v_id in youtube_video_ids}

    for path in existing_files:
        stem = path.stem
        for v_id in youtube_video_ids:
            if stem.endswith(f"_{v_id}"):
                if path.suffix == ".txt":
                    transcript_status[v_id]["txt"] = path
                elif path.suffix == ".json":
                    transcript_status[v_id]["json"] = path
                break

    # 2. downloadsフォルダ内の音声ファイルを解析します
    # v_id -> [audio_paths]
    audio_status = {v_id: [] for v_id in youtube_video_ids}
    if output_dir.exists():
        for path in output_dir.iterdir():
            if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS:
                for v_id in youtube_video_ids:
                    if f"_{v_id}" in path.name:
                        audio_status[v_id].append(path)
                        break

    # 3. 各動画の状態を分類します
    complete = []           # .txt と .json の両方が存在
    partial = []            # .txt または .json の片方のみ存在
    only_audio = []         # 音声ファイルはあるが、文字起こしが全くない
    cache_but_no_trans = [] # キャッシュにあるが、文字起こしファイルがない
    trans_but_no_cache = [] # 文字起こしはあるが、キャッシュにない
    unprocessed = []        # キャッシュなし・音声なし・文字起こしなし

    for v_id in youtube_video_ids:
        v = video_map[v_id]
        status = transcript_status[v_id]
        has_txt = status["txt"] is not None
        has_json = status["json"] is not None
        has_audio = len(audio_status[v_id]) > 0
        in_cache = v_id in downloaded_ids

        if has_txt and has_json:
            complete.append(v_id)
            if not in_cache:
                trans_but_no_cache.append(v_id)
        elif has_txt or has_json:
            partial.append((v_id, has_txt, has_json))
        elif has_audio:
            only_audio.append(v_id)
        elif in_cache:
            cache_but_no_trans.append(v_id)
        else:
            unprocessed.append(v_id)

    # 4. キャッシュにあるがYouTube上にない動画の検出（非公開・削除など）
    orphaned_cache_ids = downloaded_ids - youtube_video_ids

    # 5. レポートの出力
    print(f"【概要統計】")
    print(f"  - YouTube上の公開アーカイブ総数 : {len(videos)} 本")
    print(f"  - 文字起こし完了 (.txt & .json) : {len(complete)} 本")
    print(f"  - 未処理（新規処理対象）        : {len(unprocessed)} 本")
    print(f"  - 音声あり・文字起こし未実行    : {len(only_audio)} 本")
    print(f"  - 文字起こしの一部欠損          : {len(partial)} 本")
    print(f"  - キャッシュ済・文字起こし無    : {len(cache_but_no_trans)} 本 (※エラーでスキップされた可能性あり)")
    print(f"  - 文字起こし有・キャッシュ未登録 : {len(trans_but_no_cache)} 本")
    print(f"  - キャッシュ内のみ存在（非公開等）: {len(orphaned_cache_ids)} 本")
    print(f"--------------------------------------------------------")

    # 詳細な不整合などのアラート
    has_issues = False

    if partial:
        has_issues = True
        print(f"\n[⚠️ 警告] 文字起こしファイルの一部欠損 ({len(partial)} 件):")
        for v_id, has_txt, has_json in partial:
            v = video_map[v_id]
            missing = "JSON" if has_txt else "TXT"
            print(f"  - ID: {v_id} | {missing}ファイルが不足しています | URL: {v['url']}")
            print(f"    タイトル: {v['title']}")

    if only_audio:
        has_issues = True
        print(f"\n[ℹ️ 情報] 音声ファイルは存在するが文字起こしがありません ({len(only_audio)} 件):")
        for v_id in only_audio:
            v = video_map[v_id]
            audios = ", ".join(p.name for p in audio_status[v_id])
            print(f"  - ID: {v_id} | 音声ファイル: {audios} | URL: {v['url']}")
            print(f"    タイトル: {v['title']}")

    if trans_but_no_cache:
        has_issues = True
        print(f"\n[⚠️ 警告] 文字起こしは完了していますが、キャッシュファイルに登録されていません ({len(trans_but_no_cache)} 件):")
        print(f"  ※ 次回の実行時に再ダウンロードされる可能性があります。")
        for v_id in trans_but_no_cache:
            v = video_map[v_id]
            print(f"  - ID: {v_id} | URL: {v['url']}")
            print(f"    タイトル: {v['title']}")

    if cache_but_no_trans:
        print(f"\n[ℹ️ 参考] キャッシュに登録されていますが、文字起こしがありません ({len(cache_but_no_trans)} 件):")
        print(f"  ※ 年齢制限、非公開、メンバー限定動画など、過去の処理時にスキップされた可能性があります。")
        for v_id in cache_but_no_trans:
            v = video_map[v_id]
            print(f"  - ID: {v_id} | URL: {v['url']}")
            print(f"    タイトル: {v['title']}")

    if orphaned_cache_ids:
        print(f"\n[ℹ️ 参考] キャッシュ内にのみ存在する動画ID ({len(orphaned_cache_ids)} 件):")
        print(f"  ※ YouTube上で非公開・削除されたか、Cookieなしでアクセスできないメンバー限定動画の可能性があります。")
        for v_id in sorted(orphaned_cache_ids):
            print(f"  - ID: {v_id}")

    if not has_issues:
        print(f"\n[✅ 正常] 重大な不整合（一部欠損やキャッシュ未登録など）は検出されませんでした。")

    print(f"========================================================\n")

    retry_video_ids = []
    if cache_but_no_trans:
        try:
            print(f"\n[❓ 提案] キャッシュ済・文字起こし無の動画が {len(cache_but_no_trans)} 件あります。")
            print("これらは以前の実行で年齢制限やメンバー限定、非公開などの原因でエラー終了した可能性があります。")
            response = input("これらの動画に対して再処理（再ダウンロード・文字起こし）を試行しますか？ (y/N): ").strip().lower()
            if response in ("y", "yes"):
                retry_video_ids = cache_but_no_trans
        except KeyboardInterrupt:
            print("\n処理がキャンセルされました。")
            return 0, []
        except Exception:
            pass

    return 0, retry_video_ids


def sync_cache_report_and_fix(
    args: argparse.Namespace,
    videos: list[dict],
    channel_title: str,
    downloaded_ids: set[str],
    t_out_dir: Path,
    cache_file: Path,
    cache_data: dict
) -> int:
    """実ファイル（文字起こしテキスト）の状態に基づき、ダウンロードキャッシュを自動的に同期・修復します。"""
    from .utils import log_info, log_success, log_warn
    import time
    from .downloader import save_download_cache

    log_info("実ファイル（transcripts）とキャッシュの同期を開始します...")
    
    youtube_video_ids = {v["id"] for v in videos}
    
    # transcripts フォルダ内のファイルを解析して、実際に文字起こしテキスト（.txt）と JSON の存在状況を検出
    # v_id -> {"txt": path/None, "json": path/None}
    file_status = {v_id: {"txt": None, "json": None} for v_id in youtube_video_ids}
    if t_out_dir.exists():
        for p in t_out_dir.iterdir():
            if p.is_file() and p.suffix in (".txt", ".json"):
                stem = p.stem
                for v_id in youtube_video_ids:
                    if stem.endswith(f"_{v_id}"):
                        file_status[v_id][p.suffix[1:]] = p
                        break
                        
    # 完全な文字起こし（.txt と .json の両方が存在）がある動画 ID
    existing_txt_ids = set()
    
    # 欠損ファイル（片方しかない）のクリーンアップと削除対象の検出
    partial_removed_ids = set()
    cleaned_count = 0
    modified = False

    for v_id, status in file_status.items():
        has_txt = status["txt"] is not None
        has_json = status["json"] is not None
        
        if has_txt and has_json:
            existing_txt_ids.add(v_id)
        elif has_txt or has_json:
            # 片方しか存在しない場合、残っているファイルを削除
            for file_path in (status["txt"], status["json"]):
                if file_path is not None and file_path.exists():
                    try:
                        file_path.unlink()
                        cleaned_count += 1
                    except Exception as e:
                        log_warn(f"欠損ファイルの削除に失敗しました ({file_path.name}): {e}")
            partial_removed_ids.add(v_id)

    if cleaned_count > 0:
        log_info(f"【同期】一部欠損（TXTまたはJSONのみ存在）していた {cleaned_count} 件のトランスクリプトファイルを削除しました。")
                        
    # 1. キャッシュへの自動追加（両方のファイルがあるがキャッシュにない場合）
    added_ids = existing_txt_ids - downloaded_ids
    
    # 2. キャッシュからの自動削除（キャッシュにあるが、実ファイルが欠落または一部欠損している場合）
    missing_txt_ids = youtube_video_ids - existing_txt_ids
    removed_ids = (downloaded_ids & missing_txt_ids) | partial_removed_ids
    
    # 3. YouTube上にない動画IDのキャッシュ内クリーンアップ
    orphaned_ids = downloaded_ids - youtube_video_ids
    
    if added_ids:
        for v_id in added_ids:
            if v_id not in cache_data["downloaded_ids"]:
                cache_data["downloaded_ids"].append(v_id)
        log_info(f"【同期】文字起こしファイルが存在する {len(added_ids)} 件の動画IDをキャッシュに追加しました。")
        modified = True
        
    if removed_ids:
        new_downloaded = [v_id for v_id in cache_data["downloaded_ids"] if v_id not in removed_ids]
        cache_data["downloaded_ids"] = new_downloaded
        log_info(f"【同期】文字起こしファイルが存在しない、または不完全な {len(removed_ids)} 件の動画IDをキャッシュから削除しました。")
        modified = True
        
    if orphaned_ids:
        new_downloaded = [v_id for v_id in cache_data["downloaded_ids"] if v_id not in orphaned_ids]
        cache_data["downloaded_ids"] = new_downloaded
        log_info(f"【同期】YouTube上に存在しない {len(orphaned_ids)} 件の動画IDをキャッシュから削除（クリーンアップ）しました。")
        modified = True
        
    if modified:
        cache_data["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        save_download_cache(cache_file, cache_data)
        log_info("キャッシュファイルの同期・更新が完了しました。")
    else:
        log_info("キャッシュと実ファイルは完全に一致しています。更新は不要です。")
        
    return 0

