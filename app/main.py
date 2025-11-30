from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import logging

from semantic_segformer import segment_image
from depth_estimater import estimate_depth

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Landscape Video Sound API")

# CORS設定（GitHub Pagesからのアクセスを許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では具体的なドメインを指定
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """ヘルスチェック用エンドポイント"""
    return {"status": "ok", "message": "Landscape Video Sound API is running"}

@app.post("/process")
async def process_image(image: UploadFile = File(...)):
    """
    画像をアップロードして、セマンティックセグメンテーションと深度推定を実行
    
    Parameters:
    - image: アップロードされた画像ファイル
    
    Returns:
    - segmentation: セグメンテーションマップ (2D配列)
    - depth: 深度マップ (2D配列)
    - width: 画像の幅
    - height: 画像の高さ
    """
    try:
        # 画像の読み込み
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        logger.info(f"Processing image: {image.filename}, size: {pil_image.size}")
        
        # セマンティックセグメンテーション
        logger.info("Starting semantic segmentation...")
        segmentation_result = segment_image(pil_image)
        
        # 深度推定
        logger.info("Starting depth estimation...")
        depth_result = estimate_depth(pil_image)
        
        # レスポンスデータの構築
        response_data = {
            "segmentation": segmentation_result["map"],
            "depth": depth_result["map"],
            "width": segmentation_result["width"],
            "height": segmentation_result["height"]
        }
        
        logger.info("Processing completed successfully")
        
        return JSONResponse(content=response_data)
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/health")
async def health_check():
    """詳細なヘルスチェック"""
    return {
        "status": "healthy",
        "service": "Landscape Video Sound API",
        "endpoints": [
            "/process - POST: 画像処理エンドポイント",
            "/health - GET: ヘルスチェック"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
