from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from torch import nn
from torchvision.transforms import transforms
from PIL import Image
import io
import traceback

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pooling = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(32768, 128)
        self.output = nn.Linear(128, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pooling(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.pooling(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.pooling(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.output(x)
        return x

# 加载模型
model = CNN().to(device)
model.load_state_dict(torch.load("animal_faces_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float)
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
