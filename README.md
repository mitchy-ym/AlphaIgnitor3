# TimeStamper テストダウンローダー

## 1. 依存関係のインストールと環境構築 (AMD ROCm + Windows 対応)

Ryzen / Radeon GPU を搭載した AMD 環境 (ROCm) で `faster-whisper` による高速な文字起こしを行うためのセットアップ手順です。

### 1-1. 必須要件
- OS: Windows 10/11 または Linux
- AMD GPU (RDNA 3/3.5 アーキテクチャの iGPU/dGPU, 例: Radeon 780M / 890M / RX 7000以上)
- **JavaScript 実行環境**: YouTubeのダウンロード処理（n-parameter難読化チャレンジの解除）を行うために、システムに **[Deno](https://deno.com/)**（推奨）または Node.js がインストールされ、PATH に追加されている必要があります。
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
文字起こし処理（`python main.py pipeline ...`）の実行時、以下の設定が自動的に適用されるため、RDNA 3/3.5 の統合 GPU も自動的に動作します。
- `HSA_OVERRIDE_GFX_VERSION=11.5.0` (RDNA 3.5 iGPU/APU 互換)
- `HF_HUB_DISABLE_SYMLINKS_WARNING=1` (Windows上のHFシンボリックリンク警告抑止)
- `KMP_DUPLICATE_LIB_OK=TRUE` (OpenMP ランタイム競合回避)

## 2. 統一エントリーポイント `main.py` の使用方法

本プロジェクトの各機能は、統合 CLI エントリーポイントである `main.py` から実行することを推奨します。

```powershell
# ヘルプを表示する
python main.py --help

# ① 音声のダウンロードのみ実行
python main.py download "@ChannelHandle"

# ② ダウンロード + 文字起こしのパイプラインを実行 (推奨)
python main.py pipeline "@ChannelHandle" --transcribe-device cuda --transcribe-model turbo

# ③ 文字起こし後に Windows 側 Ollama で整形・要約・章立ても生成
python main.py pipeline "@ChannelHandle" --enrich --llm-endpoint http://172.19.208.1:11434/v1 --llm-model qwen3.6:latest

# ④ 文字起こしテキストを月次/年次/全件で結合
python main.py merge --mode monthly
python main.py merge --mode yearly
python main.py merge --mode all
```

各サブコマンドのオプション詳細を確認するには、`python main.py [サブコマンド] --help` を実行してください。


## 3. 配信アーカイブ（音声）を一括ダウンロードする

指定したYouTubeチャンネルのライブ配信アーカイブから、音声を一括でダウンロード・抽出します。
ダウンロード済みの配信や、非公開・年齢制限等でダウンロードできなかった配信はキャッシュに記録され、次回以降スキップされます。

**特徴:**
* **同時 2並行ダウンロード（デフォルト）** に対応しており、ダウンロード時間を短縮します（規制回避のための staggering スリープ機能付き）。
* デフォルトでは **再エンコードなし（そのままの形式で抽出）** で保存するため、ダウンロード完了後の FFmpeg 変換オーバーヘッドがほぼゼロです。

```powershell
python main.py download "@ChannelHandle"
```

よく使うオプション:

```powershell
python main.py download "@ChannelHandle" -f best
python main.py download "@ChannelHandle" -f mp3 -b 192k
python main.py download "@ChannelHandle" --cookies-from-browser chrome
```

- `channel_handle` (第1引数): `@`から始まるチャンネル名（例: `@Google`）
- `-f, --format`: 出力音声形式。`best` (再エンコードなしでコピー・最速), `mp3`, `m4a`, `wav`, `opus`, `flac` (既定: `best`)
- `-b, --bitrate`: 音声のビットレート (既定: `128k`)
- `-o, --output`: 保存先フォルダパス (既定: `./downloads/[チャンネル名]/`)
- `--cookies-from-browser`: メンバーシップ限定・年齢制限・非公開動画の回避用にブラウザからCookieを読み込む (例: `chrome`, `edge`, `firefox`)
- `--cookies`: Netscape形式のCookieファイルパス (例: `cookies/cookies.txt`)。指定がない場合、`cookies/cookies.txt` が存在すれば自動的に読み込みます。
- `--ffmpeg-location`: ffmpegバイナリのパス (システム環境変数に通っていない場合)
- `-w, --max-workers`: 最大同時ダウンロード数 (既定: `2`)
- `--no-verbose-progress`, `--quiet`, `-q`: 進捗ログ（[INFO]や[PROGRESS]等）の標準出力を無効にし、プログレスバー表示に切り替えます（既定では標準出力テキスト表示がONになっています）
- `--debug`: 詳細なログを出力する

## 3-2. ダウンロードと文字起こし・後段処理・テキスト結合の統合実行

音声のダウンロード、Whisper 文字起こし、Ollama による後段処理（clean / summary / chapters）、および年別テキスト結合を、ひとつのコマンドで実行します。
ダウンロードは並列実行され、ダウンロード完了した音声から順次、単一の GPU 文字起こしワーカーで処理されます。
既定で `--enrich`（Ollama 後段処理）および年別テキスト結合（`--merge`）が有効になっているため、実行するだけで文字起こしから整形・要約・章立て・年別まとめファイルまで一括生成されます。

```powershell
# ダウンロード -> 文字起こし -> Ollama enrich -> 年別テキスト結合を一括実行（デフォルト）
python main.py pipeline "@ChannelHandle"

# 文字起こしオプションを指定する例 (GPUの使用、モデル指定、文字起こし後も音声を保持)
python main.py pipeline "@ChannelHandle" --transcribe-device cuda --transcribe-model turbo --transcribe-keep-audio

# Ollama enrich を無効化して文字起こし＋テキスト結合のみ実行する例
python main.py pipeline "@ChannelHandle" --no-enrich

# テキスト結合処理のみスキップする例
python main.py pipeline "@ChannelHandle" --no-merge
```

主な統合オプション:
- `--transcribe-model`: 使用するWhisperモデル（既定: `turbo`）
- `--transcribe-language`: 対象言語（既定: `ja`, 自動検出時は `auto`）
- `--transcribe-device`: 推論デバイス（既定: `auto`）
- `--transcribe-compute-type`: 計算精度（既定: `float16`）
- `--transcribe-output-dir`: 文字起こしファイルの出力先（既定: `transcripts`）
- `--transcribe-keep-audio`: 文字起こし完了後も元の音声ファイルを保持します（既定では自動削除）。
- `--enrich`: 文字起こし完了後、Ollamaで整形・要約・章立てを生成します（既定: 有効）。
- `--no-enrich`: Ollamaによる後段処理を無効化します。
- `--merge`: パイプライン完了後、テキストを自動結合します（既定: 有効）。
- `--no-merge`: テキスト結合処理を無効化します。
- `--merge-mode`: 結合モード（既定: `yearly`, 選択肢: `yearly`, `monthly`, `all`）。
- `--merge-strip-timestamps`: 結合時にタイムスタンプを除去します（既定: 保持）。
- `--cookies`: Netscape形式のCookieファイルパス。既定では `cookies/cookies.txt` が自動的に読み込まれます。
## 4. Windows 側 Ollama で後段処理する

パイプライン実行時、`--enrich` は既定で有効になっています（無効化する場合は `--no-enrich`）。Whisper の文字起こし `.txt` を Windows 11 側で動作している Ollama に渡し、以下の3成果物を常にまとめて生成します。クラウド API には送信しません。

- `clean`: タイムスタンプ付きの整形済み文字起こし
- `summary`: 要約
- `chapters`: タイムスタンプ付き章立て

Ollama は Windows 側で起動しておきます。WSL2 からは `localhost` ではなく、Windows ホストのゲートウェイ IP を使うのが前提です。

既定値:
- エンドポイント: `http://172.19.208.1:11434/v1`
- モデル名: `qwen3.6:latest`

```powershell
python main.py pipeline "@静寧Shizune" --llm-endpoint http://172.19.208.1:11434/v1 --llm-model qwen3.6:latest
```

既定の出力先:

```text
enriched/
   ChannelName/
      clean/
         *.txt
      summary/
         *.md
      chapters/
         *.md
   ChannelName_summary/
      2023.txt
      2024.txt
      ...
```

主なオプション:
- `--llm-endpoint`: Ollama の OpenAI 互換 API エンドポイント（既定: `http://172.19.208.1:11434/v1`）
- `--llm-model`: Ollama 側のモデル名（既定: `qwen3.6:latest`）
- `--enrich-output-dir`: `pipeline` 実行時の enrich 出力先（既定: `enriched`）
- `--llm-max-tokens`: LLM APIレスポンスの最大トークン数（既定: `3072`）
- `--enrich-max-chars`: LLMに渡す1チャンクの最大文字数（既定: `1500`）
- `--enrich-force`: 既存の enrich 成果物がある場合も再生成します

備考:
- Windows 側の Ollama では `Expose Ollama to the network` を有効にしておくと接続しやすくなります。
- WSL2 から接続確認するには `curl http://172.19.208.1:11434/api/tags` を使います。


## 5. 文字起こし結果をまとめる `main.py merge`

指定ディレクトリ配下の文字起こしファイル（`*.txt`）を、チャンネルごとに集約して1つのファイルに結合します。  
`_summary` フォルダ配下のファイルは自動的に除外されます。  
※ `pipeline` 実行時は末尾で自動実行されますが、本コマンドで個別・再集約することも可能です。

`main.py` 経由（推奨）:

```powershell
# 年ごとに結合（デフォルト）
python main.py merge
python main.py merge --mode yearly

# 月ごとに結合
python main.py merge --mode monthly

# チャンネルごとに1ファイルにまとめる
python main.py merge --mode all

# enriched の clean ディレクトリを結合対象にする
python main.py merge --input-dir "enriched/静寧  Shizune - Live/clean"

# タイムスタンプを除去して結合する（デフォルトは保持）
python main.py merge --strip-timestamps
```

主なオプション:
- `--mode`: 集約モード（既定: `yearly`, 選択肢: `yearly`, `monthly`, `all`）
- `--input-dir`: 結合対象ディレクトリ（既定: `transcripts`）
- `--output-dir`: 結合結果の出力先ルート（未指定時: `input-dir` が `transcripts` なら `transcripts`、それ以外は `input-dir` の親）
- `--strip-timestamps`: 行頭の `[HH:MM:SS]` を除去して結合（既定は除去しない）

出力先: `{output_dir}/{channel_name}_summary/`

| モード | 出力ファイル名 |
|---|---|
| `yearly` | `{YYYY}.txt`（既定） |
| `monthly` | `{YYYYMM}.txt` |
| `all` | `all.txt` |

- `monthly` / `yearly` モードでは、ファイル名先頭に `YYYYMMDD_` の日付プレフィックスがないファイルはスキップされます。
- `all` モードでは、日付プレフィックスの有無にかかわらずすべてのファイルが対象になります。


