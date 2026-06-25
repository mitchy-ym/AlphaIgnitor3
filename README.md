# TimeStamper テストダウンローダー

## 1. 依存関係のインストールと環境構築 (AMD ROCm + Windows 対応)

Ryzen / Radeon GPU を搭載した AMD 環境 (ROCm) で `faster-whisper` による高速な文字起こしを行うためのセットアップ手順です。

### 1-1. 必須要件
- OS: Windows 10/11
- AMD GPU (RDNA 3/3.5 アーキテクチャの iGPU/dGPU, 例: Radeon 780M / 890M / RX 7000以上)
- 仮想環境の構築 (推奨)
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```

### 1-2. パッケージのインストール

1. **基本的な依存関係のインストール**
   ```powershell
   pip install -r requirements.txt
   ```

2. **PyTorch ROCm版のインストール**
   AMD GPU の HIP SDK を利用可能なバージョンの PyTorch をインストールします（通常、PyTorch が正常に ROCm GPU を検知するために必要です）。
   ```powershell
   pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/rocm7.2
   ```

3. **CTranslate2 (faster-whisper バックエンド) Windows ROCm 版のインストール**
   `faster-whisper` が内部で使用する `ctranslate2` は、標準の PyPI 版が CUDA (NVIDIA) のみの対応であるため、公式リリースの Windows ROCm ビルドを手動で導入します。
   
   - [CTranslate2 Releases](https://github.com/OpenNMT/CTranslate2/releases) にアクセスし、**`rocm-python-wheels-Windows.zip`** をダウンロードします（例: v4.8.0）。
   - ダウンロードした zip ファイルを展開します。
   - 展開された wheel の中から、使用している Python のバージョンに対応するファイルをインストールします（Python 3.12 を使用している場合は `cp312` を含む wheel）。
     ```powershell
     pip install path/to/ctranslate2-4.8.0-cp312-cp312-win_amd64.whl --force-reinstall
     ```

### 1-3. 環境変数と互換性の自動適用
文字起こしスクリプト `transcribe_local.py` の実行時、以下の設定が自動的に適用されるため、RDNA 3/3.5 の統合 GPU も自動的に動作します。
- `HSA_OVERRIDE_GFX_VERSION=11.0.0` (RDNA 3/3.5 iGPU/APU 互換)
- `HF_HUB_DISABLE_SYMLINKS_WARNING=1` (Windows上のHFシンボリックリンク警告抑止)
- `KMP_DUPLICATE_LIB_OK=TRUE` (OpenMP ランタイム競合回避)

## 2. 配信アーカイブ（音声）を一括ダウンロードする

指定したYouTubeチャンネルのライブ配信アーカイブから、音声を一括でダウンロード・抽出します。
ダウンロード済みの配信や、非公開・年齢制限等でダウンロードできなかった配信はキャッシュに記録され、次回以降スキップされます。

**特徴:**
* **最大 5並行ダウンロード** に対応しており、ダウンロード時間を劇的に短縮します（規制回避のための staggering スリープ機能付き）。
* デフォルトでは **再エンコードなし（そのままの形式で抽出）** で保存するため、ダウンロード完了後の FFmpeg 変換オーバーヘッドがほぼゼロです。

```powershell
python youtube_live_audio_downloader.py "@ChannelHandle"
```

よく使うオプション:

```powershell
python youtube_live_audio_downloader.py "@ChannelHandle" -f best
python youtube_live_audio_downloader.py "@ChannelHandle" -f mp3 -b 192k
python youtube_live_audio_downloader.py "@ChannelHandle" --cookies-from-browser chrome
```

- `channel_handle` (第1引数): `@`から始まるチャンネル名（例: `@Google`）
- `-f, --format`: 出力音声形式。`best` (再エンコードなしでコピー・最速), `mp3`, `m4a`, `wav`, `opus`, `flac` (既定: `best`)
- `-b, --bitrate`: 音声のビットレート (既定: `128k`)
- `-o, --output`: 保存先フォルダパス (既定: `./downloads/[チャンネル名]/`)
- `--cookies-from-browser`: メンバーシップ限定・年齢制限・非公開動画の回避用にブラウザからCookieを読み込む (例: `chrome`, `edge`, `firefox`)
- `--ffmpeg-location`: ffmpegバイナリのパス (システム環境変数に通っていない場合)
- `--debug`: 詳細なログを出力する



## 3. ローカルで文字起こしする

文字起こしスクリプトは、メディアファイルのパスまたはディレクトリを受け取ります。ディレクトリを指定した場合は、その中で更新日時が最新の対応ファイルを使います。

既定では ROCm (CUDA) が利用可能な場合は自動的に GPU を使用します。

**特徴:**
* **モデルは起動時に1回だけメモリにロード**され、複数ファイル間で使い回されるためバッチ処理が非常に高速です。
* **BatchedInferencePipeline による並列推論**が組み込まれており、複数の音声チャンクを同時にバッチ処理（既定: 16チャンク並列）することで GPU 使用率を高め、スループットを劇的に向上させています。

```powershell
python transcribe_local.py downloads
python transcribe_local.py downloads --language ja
# AMD GPU (ROCm) を明示的に指定して高速に文字起こしをする例
python transcribe_local.py downloads --device cuda --compute-type float16
```

出力ファイルは既定で `transcripts/` に保存されます。
- `.txt`: 文字起こし本文
- `.json`: 文字起こし本文とタイムスタンプ付きセグメント

主なオプション:
- `--device`: `auto`, `cuda` (ROCm), `cpu` (既定: `auto`)
- `--model`: `tiny`, `base`, `small`, `medium`, `large`, `turbo`
- `--language`: `ja` などの言語ヒント、または `auto`
- `--compute-type`: `auto`, `int8`, `float16`, `int8_float16`, `float32` (GPU で実行する場合は `float16` または `int8_float16` が推奨されます。既定: `float16`)
- `--task`: `transcribe` または `translate`
- `--batch-size`: 並列処理するチャンク数。数値を大きくすると GPU 使用率と処理速度が上がりますが、メモリ消費量が増加します（既定: `16`）

音声ファイルを直接指定する例:

```powershell
python transcribe_local.py "downloads\\some_file.mp3" --language ja
python transcribe_local.py "downloads\\20260614_【＃作業配信】色々作業＆雑談　休憩時筋トレ💪 _92 【セルフ受肉VTuber⧸静寧】_w0-3N5kjmT0.mp3"
```

既定の動作:
- `model=turbo`
- `task=transcribe`
- `device=auto` (利用可能であれば `cuda`、不可であれば `cpu`)
- `compute_type=float16`

