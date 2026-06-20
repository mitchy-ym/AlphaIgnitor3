# TimeStamper テストダウンローダー

## 1. 依存関係のインストール

```powershell
pip install -r requirements.txt
```

このプロジェクトは fast-whisper 専用です。Ryzen / Radeon GPU を搭載した AMD 環境 (ROCm) を利用できるよう構築されています。

AMD GPU を利用する場合、バックエンドである `ctranslate2` の ROCm ビルドを導入しており、自動的に環境変数 `HSA_OVERRIDE_GFX_VERSION=11.0.0` が適用されて RDNA 3/3.5 の統合 GPU (Radeon 780M / 890M 等) も動作するよう最適化されています。

## 2. テスト動画をダウンロードする

```powershell
python download_test.py
```

既定の URL:
- https://www.youtube.com/watch?v=w0-3N5kjmT0

よく使うオプション:

```powershell
python download_test.py "https://www.youtube.com/watch?v=w0-3N5kjmT0" -o downloads
python download_test.py "https://www.youtube.com/watch?v=w0-3N5kjmT0" --cookies-from-browser edge
python download_test.py "https://www.youtube.com/watch?v=w0-3N5kjmT0" --max-height 360
python download_test.py "https://www.youtube.com/watch?v=w0-3N5kjmT0" --audio-only
```

- `-o, --output-dir`: 出力先ディレクトリ
- `--cookies-from-browser`: 会員限定・年齢制限・非公開動画の取得にブラウザ Cookie を使う
- `--max-height`: 画質の上限を指定する。例: `360`, `480`, `720`
- `--audio-only`: 音声のみをダウンロードして抽出する
- `--audio-format`: `--audio-only` 使用時の音声形式。`mp3`, `m4a`, `wav`, `opus`, `flac`

軽量にダウンロードする例:

```powershell
python download_test.py --max-height 360
```

音声のみをダウンロードする例:

```powershell
python download_test.py --audio-only
python download_test.py --audio-only --audio-format m4a
```

## 3. ローカルで文字起こしする

文字起こしスクリプトは、メディアファイルのパスまたはディレクトリを受け取ります。ディレクトリを指定した場合は、その中で更新日時が最新の対応ファイルを使います。

既定では ROCm (CUDA) が利用可能な場合は自動的に GPU を使用します。

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
- `--compute-type`: `auto`, `int8`, `float16`, `int8_float16`, `float32` (GPU で実行する場合は `float16` または `int8_float16` が推奨されます。既定: `auto`)
- `--task`: `transcribe` または `translate`

音声ファイルを直接指定する例:

```powershell
python transcribe_local.py "downloads\\some_file.mp3" --language ja
python transcribe_local.py "downloads\\20260614_【＃作業配信】色々作業＆雑談　休憩時筋トレ💪 #92 【セルフ受肉VTuber⧸静寧】_w0-3N5kjmT0.mp3"
```

既定の動作:
- `model=turbo`
- `task=transcribe`
- `device=auto` (利用可能であれば `cuda`、不可であれば `cpu`)
- `compute_type=auto` (GPU時は `float16`、CPU時は `float32`)
