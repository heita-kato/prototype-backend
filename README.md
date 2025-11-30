# Landscape Video Sound API - Backend

風景動画から音を生成するWebアプリのバックエンドAPI。画像のセマンティックセグメンテーションと深度推定を提供します。

## 機能

- **セマンティックセグメンテーション**: SegFormer (ADE20K)を使用した領域分類
- **深度推定**: MiDaSを使用した深度マップ生成
- **REST API**: FastAPIによる高速なAPI提供

## ディレクトリ構成

```
prototype-backend/
│
├── app/
│   ├── main.py                 # FastAPI メインアプリケーション
│   ├── semantic_segformer.py   # セマンティックセグメンテーション処理
│   └── depth_estimater.py      # 深度推定処理
│
├── requirements.txt            # Python依存パッケージ
└── README.md                   # このファイル
```

## セットアップ

### ローカル環境

1. **リポジトリのクローン**
   ```bash
   git clone <your-repository-url>
   cd prototype-backend
   ```

2. **仮想環境の作成と有効化**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **依存パッケージのインストール**
   ```bash
   pip install -r requirements.txt
   ```

4. **サーバーの起動**
   ```bash
   cd app
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

5. **APIにアクセス**
   - ブラウザで `http://localhost:8000` を開く
   - API ドキュメント: `http://localhost:8000/docs`

### Renderへのデプロイ

1. **Renderアカウントの作成**
   - [Render](https://render.com/)でアカウントを作成

2. **新しいWeb Serviceを作成**
   - "New +" → "Web Service" を選択
   - GitHubリポジトリを接続

3. **デプロイ設定**
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `cd app && uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: Free (または有料プラン)

4. **環境変数（オプション）**
   必要に応じて環境変数を設定:
   - `PYTHON_VERSION`: `3.10.0` (推奨)

5. **デプロイ**
   - "Create Web Service" をクリック
   - 自動的にビルドとデプロイが開始されます

## API エンドポイント

### POST /process

画像をアップロードして、セグメンテーションと深度推定を実行します。

**リクエスト:**
```bash
curl -X POST "https://your-api.onrender.com/process" \
  -F "image=@path/to/image.jpg"
```

**レスポンス:**
```json
{
  "segmentation": [[0, 1, 2, ...], [3, 4, 5, ...]],
  "depth": [[100.5, 102.3, ...], [99.8, 101.2, ...]],
  "width": 1080,
  "height": 1920
}
```

### GET /

ヘルスチェック用エンドポイント

**レスポンス:**
```json
{
  "status": "ok",
  "message": "Landscape Video Sound API is running"
}
```

### GET /health

詳細なヘルスチェック

**レスポンス:**
```json
{
  "status": "healthy",
  "service": "Landscape Video Sound API",
  "endpoints": [
    "/process - POST: 画像処理エンドポイント",
    "/health - GET: ヘルスチェック"
  ]
}
```

## 使用モデル

- **セマンティックセグメンテーション**: [nvidia/segformer-b2-finetuned-ade-512-512](https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512)
  - ADE20Kデータセットで学習済み
  - 150クラスの物体認識が可能

- **深度推定**: [MiDaS Small](https://github.com/isl-org/MiDaS)
  - 軽量で高速な深度推定モデル

## トラブルシューティング

### モデルのダウンロードに失敗する

初回起動時、HuggingFaceからモデルをダウンロードするため時間がかかります。Renderの無料プランでは制限があるため、タイムアウトする場合があります。

**対策:**
- 有料プランを使用する
- モデルを事前にダウンロードしてリポジトリに含める（ただしサイズが大きい）

### メモリ不足エラー

深度推定やセグメンテーションは大量のメモリを使用します。

**対策:**
- Renderのインスタンスタイプをアップグレード
- 画像サイズを制限する
- より軽量なモデルを使用する

### CORS エラー

フロントエンドからAPIにアクセスできない場合、CORS設定を確認してください。

**対策:**
- `main.py`の`allow_origins`を適切に設定
- 本番環境では具体的なドメインを指定（例: `["https://your-github-pages.github.io"]`）

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 貢献

プルリクエストを歓迎します！大きな変更の場合は、まずissueを開いて変更内容を議論してください。
