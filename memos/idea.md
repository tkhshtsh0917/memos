# CachedMultipleNegativeRankingLossでの訓練精度をより高めるアイデア

## 1. 反復的なハードネガティブ・マイニング

### 1-1. 概要

`MultipleNegativeRankingLoss`あるいは`CachedMultipleNegativeRankingLoss`で使用するデータセットとしては、以下の3つのフォーマットが許容されている。

```plaintext
1. (anchor, positive)
2. (anchor, positive, negative)
3. (anchor, positive, negative_1, ..., negative_n)
```

1つ以上の負例を指定した場合は、それらはin-batch負例とともに学習に使用される。

ここでは単にベースモデルやBM25ベースでマイニングするのではなく、訓練済モデルを使用してエンコードしたベクトルを使用してクエリごとにTop10などを検索し、それらの中からハードネガティブをマイニングしてデータセットに反映し訓練に使用することで更なる精度向上が望める。

つまり、改めてコーパス全体から「今のモデルが最も正解と勘違いしやすい上位負例」をマイニングするというアイデア。

### 1-2. 出典

http://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss

## 2. `MatryoshkaLoss`の使用

### 2-1. 概要

`MultipleNegativeRankingLoss`あるいは`CachedMultipleNegativeRankingLoss`が持つ「異方性が生じる恐れがある」という弱みを解消しつつ、より小さい次元の部分ベクトルでも動作するように訓練することで強力な正則化が働き、最大次元においても通常のMNRLによる訓練を行った場合よりも汎化性能が向上する可能性があるらしい。

### 2-2. 実装イメージ

```python
from sentence_transformers import losses

base_loss = losses.CachedMultipleNegativesRankingLoss(model=model, mini_batch_size=32)
# 768次元のモデルを、256, 128, 64次元でも機能するように同時に訓練
train_loss = losses.MatryoshkaLoss(model=model, loss=base_loss, matryoshka_dims=[768, 256, 128, 64])
```

### 2-3. 出典

[原著論文](https://arxiv.org/abs/2205.13147)に以下の記載あり。

"*MRL learns coarse-to-fine representations that are at least as accurate and rich as independently trained low-dimensional representations.*"

### 2-3. 注意事項

`Matryoshka2dLoss`というバリエーションもあるが、こちらは`CachedMultipleNegativeRankingLoss`と組み合わせることはできない模様。

## 3. `scale`パラメータの調整

### 3-1. 概要

`MultipleNegativeRankingLoss`あるいは`CachedMultipleNegativeRankingLoss`で指定可能な`scale`パラメータ（デフォルト: `20.0`）をチューニングすることで精度向上に繋がる可能性がある。

この`scale`パラメータは損失計算時の類似度に掛け合わされるため、値を大きくすることで正例と負例の差をよりハッキリさせることができる可能性があるため。

まずは `30.0`-`40.0` くらいを試してみるといいかも。

### 3-2. 出典

https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss

## 4. `directions`, `partition_mode`パラメータの調整

### 4-1. 概要

`(anchor, positive)`あるいは`(anchor, positive, negative)`形式でデータセットを作成する場合、以下に示す設定値を使用するのが推奨されている。

### 4-2. 実装イメージ

```python
loss = MultipleNegativesRankingLoss(
    model,
    directions=("query_to_doc", "query_to_query", "doc_to_query", "doc_to_doc"),
    partition_mode="joint",  # single softmax over all selected interaction terms
)
```

### 4-3. 出典

https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss

## 5. 推論時に埋め込み計算の後処理として白色化処理を追加

### 5-1. 概要

モデルの訓練自体は`MultipleNegativeRankingLoss`あるいは`CachedMultipleNegativeRankingLoss`でそのまま行い、推論時に出力された埋め込みベクトルに対して、後処理として数学的な補正（主成分分析と標準化）をかける古典的かつ強力な手法。

この後処理を行うことで、偏っている（可能性がある）ベクトル空間を「平均0、共分散行列が単位行列」へと変形させることができる。

訓練データセットをもとに白色化行列を事前計算し環境にデプロイする必要があるが、推論時は線形変換するだけなのでレイテンシーは低い想定。

### 5-2. 実装イメージ

#### 5-2-1. 事前準備

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. モデルの読込み
model = SentenceTransformer("your-finetuned-model")

# 訓練データなど、ドメインのテキストを1,000〜2,000件ほど用意しておく
texts = ["テキスト1", "テキスト2", ...]
embeddings = model.encode(texts, convert_to_numpy=True)

# 平均を計算する
mu = np.mean(embeddings, axis=0)

# 共分散行列の計算と特異値分解（SVD）により、白色化行列Wを算出する
cov = np.cov(embeddings - mu, rowvar=False)
U, S, V = np.linalg.svd(cov)
W = np.dot(U, np.diag(1.0 / np.sqrt(S)))

# 算出した`mu`と`W`は`pickle`などで保存しておく
```

#### 5-2-2. 推論時

```python
def whiten_vectors(raw, mu, W):
    """ベクトルを理想的な空間（平均0、共分散行列=単位行列）に変換する"""
    return np.dot(raw - mu, W)

# 検索クエリとデータベースのドキュメントをベクトル化
query_emb = model.encode(["クエリ"], convert_to_numpy=True)
doc_embs = model.encode(["ドキュメント1", "ドキュメント2", ...], convert_to_numpy=True)

# 後処理として白色化を適用する
# `mu`, `W`は事前に`pickle`などで保存しておいたものを読み込んで使用する
whitened_query = whiten_vectors(query_emb, mu, W)
whitened_docs = whiten_vectors(doc_embs, mu, W)

# `whitened_query`と`whitened_docs`を使ってANNを行う。
```

### 5-3. 出典

* https://arxiv.org/abs/2103.15316
* https://aclanthology.org/2023.findings-acl.778/
* https://aclanthology.org/2020.emnlp-main.733/
* https://aclanthology.org/2023.acl-long.882/
* https://medium.com/codex/when-bad-negatives-bend-space-anisotropy-in-contrastive-learning-eae0fd97ca46
