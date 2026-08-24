# Embedding Cache Design

## Goal

`trade_days=504` 以上を維持したまま学習時間を大きく削るため、TiREX 埋め込み計算を再利用する。
現行コードでは LightGBM ヘッドは十分に速く、支配的なボトルネックは `train_and_save()` 内の埋め込み計算である。

対象コード:

- `alphaignitor/pipeline/daily_tirex_train_and_forecast.py`
- 特に `_build_train_tensors()`
- 特に `train_and_save()`
- 特に `_compute_embeddings_batched()`

## Current Flow

現行の horizon ごとの学習フローは以下。

1. `_build_train_tensors()` が全銘柄・全ウィンドウの `X`, `y` を CPU Tensor で構築する
2. `emb_holder._create_train_val_datasets()` が train/val を分割する
3. `_compute_embeddings_batched()` が train と val の全ウィンドウを毎回 GPU で埋め込む
4. その埋め込みを使って quantile ごとに LightGBM を学習する

このため、毎日フル再学習すると「前日とほぼ同じ窓」に対しても TiREX 埋め込みを毎回再計算している。

## Best ROI Strategy

最も効くのは「window ごとの埋め込みキャッシュ」ではなく「ticker-horizon ごとの埋め込みキャッシュ」である。

理由:

- 日次運用では、前日までの履歴窓の大半が翌日も再利用可能
- quantile ごとの再利用は既に実装済みで、次に大きい重複は日次 run 間の再計算
- window 単位ファイルはファイル数が爆発し、管理コストが高い
- ticker-horizon 単位なら append と tail 切り出しで管理しやすい

## Cache Unit

キャッシュ単位は `run universe x horizon x ticker` とする。

1ファイルは「1銘柄・1horizon の全学習窓に対応する埋め込み行列」とする。

保存対象:

- `embeddings`: shape `(n_windows, embedding_dim)` の `float32`
- `targets`: shape `(n_windows,)` の `float32`
- `trade_dates`: 各 window の target 日付配列
- `window_end_dates`: 各 window のコンテキスト終端日付配列
- `meta`: キャッシュ整合性判定に必要な設定値

推奨フォーマット:

- `parquet` ではなく `npz` か `pt`
- 第一候補は `np.savez_compressed()`

理由:

- 数値行列中心で実装が単純
- 読み込みが速い
- pandas 依存を増やさずに済む

## Directory Layout

```
models/{run_date}_us_stock/
  h1_q0.1.joblib
  h1_q0.5.joblib
  h1_q0.9.joblib
  meta.json

cache/embeddings/us_stock/
  context252_trade756_aug0_chunk4096_batch512/
    cache_meta.json
    h1/
      AAPL.npz
      MSFT.npz
      ...
    h2/
      ...
```

`context_days`, `trade_days`, `data_augmentation`, `embedding_chunk_size`, `tirex_batch_size` をディレクトリ名に含める。
モデル互換性に関わる値を cache namespace に入れることで、雑な取り違えを防ぐ。

## Cache Key

キャッシュキーは以下の組で定義する。

- market
- ticker
- horizon
- context_days
- trade_days
- data_augmentation
- covariate schema hash
- panel schema version
- TiREX embedding signature

`TiREX embedding signature` は少なくとも以下を含める。

- TiREX package version
- data augmentation on/off
- embedding output dim
- device type は含めない

device をキーに含めない理由:

- 埋め込み値は推論結果であり、CPU/GPU に依らず再利用したい
- 実運用ではキャッシュ生成時の device と学習時の device が違っても問題ない

## Data Model Per File

各 `ticker.npz` の中身:

- `embeddings`: `float32[n_windows, emb_dim]`
- `targets`: `float32[n_windows]`
- `target_trade_dates`: `U10[n_windows]`
- `window_end_trade_dates`: `U10[n_windows]`
- `window_hashes`: `U32[n_windows]`  — 各 window 入力テンソル `(n_channels, context_days)` の内容ハッシュ
- `meta_json`: JSON 文字列 1 個

`target_trade_dates` と `window_end_trade_dates` は再利用対象行の対応付けに使う。
`window_hashes` は「日付は同じだが中身が変わった window」を stale と判定するために使う。

推奨 `meta_json` の必須項目:

- `cache_format_version`
- `ticker`
- `horizon`
- `context_days`
- `data_augmentation`
- `cov_cols`
- `tirex_version`
- `panel_parquet_name`
- `panel_build_signature`
- `window_hash_algo`

## Build Rule

初回作成時:

1. 既存の `_create_windows()` と同じロジックで単一銘柄 `X`, `y` を作る
2. その銘柄だけを train/val にまだ分けず、全 windows を日付順のまま埋め込む
3. `ticker.npz` に保存する

更新時:

1. 現在の panel から同じ条件で最新 windows を再計算する
2. 各 window の入力テンソルから `window_hashes` を再計算する
3. `(target_trade_date, window_end_trade_date)` と `window_hash` が両方一致した行だけ再利用する
4. 不一致 window だけ埋め込みを再計算する
5. 最新の `window_hashes` と meta で cache file 全体を書き直す

## Cache Safety

旧形式 cache は `window_hashes` と新 meta を持たないため、自動的に非互換として扱う。
これは「安全性向上を入れたら初回だけ cache を作り直す」前提の挙動であり、
以後の日次運用では大半の window が再利用される。

## Why Split After Caching

train/val 分割後の埋め込みをキャッシュしない。

理由:

- split ルール変更に弱い
- 現状の `val_split_ratio=0.1` 固定でも、将来に時系列 split へ移行したい
- 生の埋め込み行列を持っていれば、split は何度でもやり直せる

したがってキャッシュ対象は `full windows in chronological order` にする。

## Integration Plan

### 1. New Helpers

`alphaignitor/pipeline/daily_tirex_train_and_forecast.py` に以下を追加する。

- `_cache_namespace(...) -> str`
- `_cache_file(cache_root: Path, namespace: str, horizon: int, ticker: str) -> Path`
- `_build_train_windows_for_ticker(...) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]`
- `_load_embedding_cache(path: Path) -> dict | None`
- `_save_embedding_cache(path: Path, payload: dict) -> None`
- `_refresh_embedding_cache_for_ticker(...) -> dict`
- `_load_or_compute_horizon_dataset(...) -> tuple[np.ndarray, np.ndarray, list[str]]`

### 2. Replace Horizon-Wide Tensor Build

現行の `train_and_save()` は horizon ごとに一度 `X, y` を全件メモリ構築している。
これを以下に置き換える。

1. ticker ごとに cache を読む
2. 不足分だけ埋め込みを計算する
3. `embeddings` と `targets` を ticker ごとに結合する
4. 結合後に train/val split する
5. LightGBM を学習する

これにより、巨大な `X` の 3D Tensor を horizon 単位で毎回フル生成しなくてもよくなる。

### 3. Keep Existing Public CLI

CLI は増やしすぎない。

追加候補は以下の3つだけに留める。

- `--embedding-cache-root` default `cache/embeddings`
- `--disable-embedding-cache` default false
- `--refresh-embedding-cache` default false

`refresh` は整合性不安時の全再構築用。

## Recommended Split Fix At Same Time

キャッシュ導入時に、split は `shuffle` ではなく時系列末尾 10% を val にする。

理由:

- 日次予測でランダム split は評価が甘くなりやすい
- キャッシュは時系列順の全埋め込みを持つので、末尾 split が自然

実装は簡単で、結合した `embeddings` と `targets` を末尾比率で切るだけでよい。

## Invalidation Rules

以下のいずれかが変わったら cache miss 扱いにする。

- `context_days`
- `trade_days`
- `data_augmentation`
- covariate columns の並び
- TiREX version
- embedding output dimension

以下は cache 再利用可。

- quantile の変更
- LightGBM ハイパーパラメータ変更
- top_n
- report 周り

## Storage Estimate

`trade_days=756`, `context_days=252`, all tickers では約 179,776 windows。

`data_augmentation=false`, `embedding_dim=13,312`, `float32` の場合:

$$
179,776 \times 13,312 \times 4 \approx 9.57 \text{ GB}
$$

target や日付配列を含めても 10GB 強で収まる見込み。

これは 1 回の再学習時間短縮効果に対して十分に安い。

## Expected Effect

日次運用では毎日ほぼ 1 営業日ぶんしか新規 window が増えないため、理想的には再埋め込み量はフル再計算の数 % まで落ちる。

期待効果:

- 初回フルビルド: 今まで通り重い
- 2日目以降: TiREX 埋め込み時間を大幅短縮
- LightGBM 再学習だけ残るため、全体時間は埋め込み支配から head 支配に近づく

## Implementation Order

1. cache の read/write helper 追加
2. 1銘柄・1horizon の full-window embedding 保存
3. horizon ごとの結合ローダ追加
4. train/val split を chronological split に変更
5. `--disable-embedding-cache` を追加
6. 初回 run と 2 回目 run の時間比較テストを追加

## Minimal Safe Version

最初の実装では差分 append まで入れなくてよい。

まずは以下で十分に効果が出る。

1. cache file があれば丸ごと読む
2. なければ銘柄単位で full rebuild する
3. 翌日 run では「最終 trade_date が一致しない ticker だけ再計算」する

これでも大半の ticker で再利用できる。

## Recommended Next Code Change

最初の実装対象は `prediction_days=1` に限定する。

理由:

- 現在の主運用は 1 日先予測
- 複数 horizon を同時に扱うと cache file 数と整合性チェックが増える
- まず 1 日先で ROI を確認してから horizon 一般化すべき

その後、`horizon` を namespace かサブディレクトリに含めて拡張する。