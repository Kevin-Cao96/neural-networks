from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from torch import nn
from torchvision.transforms import transforms
from torchvision import models
from PIL import Image
import io
import traceback

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# 加载模型
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load("animal_faces_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
class_names = ["cat", "dog", "wild"]

app = FastAPI(title="动物人脸分类高级API", version="2.0")
# 允许前端网页跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"msg": "高级动物分类接口文档：/docs"}

# 单图预测，返回置信度
@app.post("/predict", response_class=JSONResponse)
async def predict(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(("jpg", "jpeg", "png")):
            raise HTTPException(status_code=400, detail="仅支持jpg/png图片")
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(img_tensor)
            prob = torch.softmax(logits, dim=1)[0]
        pred_idx = torch.argmax(prob).item()
        return {
            "filename": file.filename,
            "predict_class": class_names[pred_idx],
            "confidence": round(prob[pred_idx].item() * 100, 2),
            "all_prob": {cls: round(prob[i].item()*100,2) for i,cls in enumerate(class_names)}
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="图片解析失败，上传清晰动物人脸")

# 批量上传多张图片接口
@app.post("/batch_predict")
async def batch_predict(files: list[UploadFile] = File(...)):
    res_list = []
    for file in files:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tensor_img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor_img)
            prob = torch.softmax(out, dim=1)[0]
        idx = torch.argmax(prob).item()
        res_list.append({
            "file": file.filename,
            "label": class_names[idx],
            "confidence": round(prob[idx].item()*100,2)
        })
    return {"batch_result": res_list}
