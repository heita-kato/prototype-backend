import torch
import numpy as np
import cv2
import torchvision.transforms as transforms
from PIL import Image
import logging

logger = logging.getLogger(__name__)

# グローバル変数（モデルのキャッシュ用）
_model = None

def load_model():
    """
    MiDaSモデルをロードする（初回のみ）
    """
    global _model
    
    if _model is None:
        logger.info("Loading MiDaS model...")
        _model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        _model.eval()
        logger.info("MiDaS model loaded successfully")
    
    return _model

def estimate_depth(image: Image.Image) -> dict:
    """
    画像から深度推定を実行
    
    Parameters:
    - image: PIL Image オブジェクト (RGB)
    
    Returns:
    - dict: {
        "width": 画像の幅,
        "height": 画像の高さ,
        "map": 深度マップ (2D配列)
      }
    """
    try:
        # モデルのロード
        model = load_model()
        
        # 元の画像サイズを保存
        original_size = image.size  # (width, height)
        
        # 変換処理の定義
        transform = transforms.Compose([
            transforms.Resize((384, 384)),  # 入力画像を384x384にリサイズ
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # 画像の前処理
        input_tensor = transform(image).unsqueeze(0)
        
        # 深度推定
        with torch.no_grad():
            depth_map = model(input_tensor)
        
        # NumPy配列に変換
        depth_map = depth_map.squeeze().cpu().numpy()
        
        # 深度マップのリサイズ（元の画像サイズに戻す）
        depth_map_resized = cv2.resize(
            depth_map,
            (original_size[0], original_size[1]),
            interpolation=cv2.INTER_CUBIC
        )
        
        # 画像サイズを取得
        height, width = depth_map_resized.shape
        
        # 結果を辞書形式で返す
        result = {
            "width": width,
            "height": height,
            "map": depth_map_resized.tolist()
        }
        
        logger.info(f"Depth estimation completed: {width}x{height}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error in estimate_depth: {str(e)}")
        raise
