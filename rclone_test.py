import argparse
import subprocess
import tempfile
from pathlib import Path

def run_rclone(args_list: list[str]) -> subprocess.CompletedProcess:
    cmd = ["rclone"] + args_list
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result

def main():
    parser = argparse.ArgumentParser(description="Test rclone read/write with Google Drive.")
    parser.add_argument(
        "--remote",
        default="gdrive",
        help="Name of the rclone remote configured for Google Drive (default: gdrive)"
    )
    parser.add_argument(
        "--folder",
        default="TimeStamper",
        help="Target folder in Google Drive (default: TimeStamper)"
    )
    args = parser.parse_args()

    remote_path = f"{args.remote}:{args.folder}"
    print(f"Target remote path: {remote_path}")

    # 1. ローカルの一時テストファイルを作成
    test_content = "Hello, this is a test file for rclone integration in TimeStamper!"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as temp_file:
        temp_file.write(test_content)
        temp_file_path = Path(temp_file.name)
    
    print(f"Created temporary local file: {temp_file_path}")

    try:
        # 2. Google Drive へ書き出し (Upload)
        dest_filename = "rclone_test_file.txt"
        upload_path = f"{remote_path}/{dest_filename}"
        print(f"\n--- Step 1: Uploading file to Google Drive: {upload_path} ---")
        
        res = run_rclone(["copyto", str(temp_file_path), upload_path])
        if res.returncode != 0:
            print(f"Upload failed (exit code {res.returncode}):\n{res.stderr}")
            print("\n[HINT] Google Driveのリモート設定が済んでいないか、リモート名が違います。")
            print("rclone config を実行して、Google Driveのリモートを作成してください。")
            return
        print("Upload successful!")

        # 3. Google Drive のディレクトリ一覧を取得 (Read directory)
        print(f"\n--- Step 2: Listing files in {remote_path} ---")
        res = run_rclone(["lsf", remote_path])
        if res.returncode != 0:
            print(f"List failed (exit code {res.returncode}):\n{res.stderr}")
            return
        print(f"Files found in remote folder:\n{res.stdout.strip()}")

        # 4. Google Drive から読み込み (Download)
        download_local_path = Path("./rclone_test_downloaded.txt")
        print(f"\n--- Step 3: Downloading file from Google Drive to: {download_local_path} ---")
        res = run_rclone(["copyto", upload_path, str(download_local_path)])
        if res.returncode != 0:
            print(f"Download failed (exit code {res.returncode}):\n{res.stderr}")
            return
        
        # ダウンロードしたファイルの中身を確認
        if download_local_path.exists():
            downloaded_content = download_local_path.read_text()
            print(f"Downloaded file content: '{downloaded_content}'")
            if downloaded_content == test_content:
                print("Success: Downloaded content matches original content!")
            else:
                print("Warning: Content mismatch!")
            
            # ローカルのダウンロードファイルを削除
            download_local_path.unlink()
        else:
            print("Error: Downloaded file not found on local disk.")

        # 5. リモートのテストファイルを削除 (Cleanup)
        print(f"\n--- Step 4: Cleaning up test file on Google Drive ---")
        res = run_rclone(["deletefile", upload_path])
        if res.returncode != 0:
            print(f"Cleanup failed (exit code {res.returncode}):\n{res.stderr}")
        else:
            print("Remote test file deleted successfully.")

    finally:
        # ローカルの一時ファイルを削除
        if temp_file_path.exists():
            temp_file_path.unlink()

if __name__ == "__main__":
    main()
