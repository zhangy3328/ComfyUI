import onnxruntime as ort
from insightface.app import FaceAnalysis
import numpy as np
import cv2

print("Available providers:", ort.get_available_providers())

# 初始化 FaceAnalysis，匹配 face_detector.py 的配置
app = FaceAnalysis(
    allowed_modules=["detection", "landmark_2d_106"],
    root="F:/zy/ComfyUI_windows/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper/checkpoints/auxiliary",
    providers=["CUDAExecutionProvider"],
)
app.prepare(ctx_id=0, det_size=(512, 512))
print("Model loaded successfully")

# 测试一张图片
test_image = np.zeros((512, 512, 3), dtype=np.uint8)  # 空白图像，仅用于测试加载
faces = app.get(test_image)
print("Faces detected:", len(faces))