# TimeStamper テストダウンローダー

## 1. 依存関係のインストール

```powershell
pip install -r requirements.txt
```

このプロジェクトは fast-whisper 専用です。Ryzen / Radeon GPU を搭載した AMD 環境 (ROCm) を利用できるよう構築されています。

AMD GPU を利用する場合、バックエンドである `ctranslate2` の ROCm ビルドを導入しており、自動的に環境変数 `HSA_OVERRIDE_GFX_VERSION=11.0.0` が適用されて RDNA 3/3.5 の統合 GPU (Radeon 780M / 890M 等) も動作するよう最適化されています。

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
* バッチサイズとして `8`（`batch_size=8`）が指定されており、GPUの処理効率が最大化されています。

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

