# AlphaIgnitor3

AlphaIgnitor3 は、米国株の日足 `close` 系列を使って Chronos-2 / TimesFM 2.5 / TiREX の zero-shot 予測を行い、Optuna で銘柄別の最適アンサンブル重みを探索する日次予測パイプラインです。

## 概要

`main.py run-daily` は次の順に実行します。

| ステージ | 内容 | 主な出力 |
|---|---|---|
| download | Massive REST から日足 OHLCV を取得 | `aggs/us_stock_day/YYYY-MM-DD.parquet` |
| forecast | 3モデルの zero-shot 予測、Optuna重み最適化、アンサンブル | `predict/<asof>_us_stock_ensemble_forecast.parquet` |
| report | アンサンブルHTMLレポート生成 | `report/<asof>/report.html` |

legacy の特徴量パネル生成は通常不要です。必要な場合のみ `--build-panel` を指定します。

## インストール

```powershell
python -m pip install -r requirements.txt
```

`requirements.txt` は `submodules/chronos-forecasting`、`submodules/timesfm`、`submodules/tirex` を editable install します。

## 認証情報

`secrets/credentials.env` に API キーを設定します。

```dotenv
MASSIVE_API_KEY=YOUR_REST_API_KEY
# Optional: API_KEY=YOUR_REST_API_KEY
HF_TOKEN=YOUR_HUGGINGFACE_TOKEN
```

`MASSIVE_API_KEY` は日足データ取得に使います。`HF_TOKEN` は Hugging Face モデル取得で必要になる場合があります。

## 実行

```powershell
python main.py run-daily --config config/prod.yaml
```

ローカルの日足データを使って予測とレポートだけ作る場合:

```powershell
python main.py run-daily --config config/prod.yaml --skip-download
```

軽量確認:

```powershell
python main.py run-daily --config config/test.yaml --skip-download
```

Optuna の探索量を一時的に上書きする場合:

```powershell
python main.py run-daily --config config/prod.yaml --optuna-trials 20 --optuna-timeout-minutes 2
```

## アンサンブル仕様

- モデル: `chronos2`, `timesfm`, `tirex`
- 入力: `close` の単系列
- 予測: `t+1` から `t+5` の close 価格
- アンサンブル: 価格予測値の重み付き平均
- 重み制約: 非負・合計1
- 重み単位: 銘柄ごと、3ホライズン共通
- 目的関数: 直近 `optuna_window_days` 営業日の walk-forward 平均方向精度
- 方向判定: 予測 close が as-of close より高ければ `UP`、低ければ `DOWN`
- モデル欠損時: 利用可能モデルが `min_available_models` 以上なら重みを再正規化して続行

Optuna study は `optuna_storage_path`、zero-shot モデル予測キャッシュと最新重みは `prediction_cache_path` に保存します。

## 設定

主な設定は `config/prod.yaml` と `config/test.yaml` にあります。

| 項目 | 説明 |
|---|---|
| `prediction_days` | 予測ホライズン数。通常は5 |
| `context_days` | 各zero-shotモデルへ渡す過去営業日数。通常は60 |
| `optuna_window_days` | 重み最適化に使う直近営業日数。通常は直近1年相当の252 |
| `optuna_n_trials` | 1銘柄あたりのOptuna trial数 |
| `optuna_timeout_minutes` | 1銘柄あたりのOptuna時間上限 |
| `optuna_storage_path` | Optuna study SQLite |
| `prediction_cache_path` | モデル予測キャッシュと最新重みSQLite |
| `ensemble_models` | 使用するzero-shotモデル |
| `min_available_models` | 続行に必要な成功モデル数 |
| `optimizer_workers` | Optuna trial 並列数。nullならCPUコア数 |
| `max_tickers` | 小規模確認用の銘柄数制限 |
| `report_outdir` | HTMLレポート出力先 |

環境変数 `TICKERS` または `TICKERS_CSV` で対象銘柄を上書きできます。

## レポート

HTMLレポートは全銘柄を対象に、各予測営業日を列として表示します。

主な列:

- Chronos-2 / TimesFM / TiREX 個別予測
- アンサンブル予測値
- 予測リターンと方向判定
- 実績 close / 実績方向（利用可能な場合）
- 最新のアンサンブル重み
- 方向精度、RMSE、MAE

ランキングは全予測営業日のアンサンブル予測リターン平均順です。

## ログと進捗

標準出力と `log/YYYY-MM-DD.log` に進捗を出します。長時間処理中も、モデルロード、価格パネル読込、銘柄ごとの予測キャッシュ生成、Optuna trial 実行状況が分かるようになっています。
