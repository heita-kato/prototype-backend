import torch
import numpy as np
from PIL import Image
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import torchvision.transforms.functional as F
import logging

logger = logging.getLogger(__name__)

# グローバル変数（モデルのキャッシュ用）
_model = None
_feature_extractor = None

def load_model():
    """
    Segformerモデルをロードする（初回のみ）
    """
    global _model, _feature_extractor
    
    if _model is None or _feature_extractor is None:
        logger.info("Loading Segformer model...")
        model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
        _feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
        _model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        _model.eval()
        logger.info("Segformer model loaded successfully")
    
    return _model, _feature_extractor

def segment_image(image: Image.Image) -> dict:
    """
    画像からセマンティックセグメンテーションを実行
    
    Parameters:
    - image: PIL Image オブジェクト (RGB)
    
    Returns:
    - dict: {
        "width": 画像の幅,
        "height": 画像の高さ,
        "map": セグメンテーションマップ (2D配列)
      }
    """
    try:
        # モデルのロード
        model, feature_extractor = load_model()
        
        # 元の画像サイズを保存
        original_size = image.size  # (width, height)
        
        # 前処理
        inputs = feature_extractor(images=image, return_tensors="pt")
        
        # 推論
        with torch.no_grad():
            outputs = model(**inputs)
        
        # セグメンテーション結果
        logits = outputs.logits
        segmentation_map = torch.argmax(logits.squeeze(), dim=0).cpu().numpy()
        
        # 元の画像サイズにリサイズ
        segmentation_map_resized = F.resize(
            Image.fromarray(segmentation_map.astype(np.uint8)),
            size=(original_size[1], original_size[0]),
            interpolation=Image.NEAREST
        )
        
        # NumPy配列に変換
        segmentation_map_resized_array = np.array(segmentation_map_resized)
        
        # 結果を辞書形式で返す
        result = {
            "width": original_size[0],
            "height": original_size[1],
            "map": segmentation_map_resized_array.tolist()
        }
        
        logger.info(f"Segmentation completed: {original_size[0]}x{original_size[1]}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error in segment_image: {str(e)}")
        raise
