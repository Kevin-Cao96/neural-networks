import torch
from torch import nn
from torch.optim import Adam
from torchvision import transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import random
from PIL import Image
from pathlib import Path
import numpy as np
import pandas as pd
import os
import yaml

script_dir = os.path.dirname(os.path.abspath(__file__))

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ======== 从 config.yaml 读取配置 ========
config_path = os.path.join(script_dir, "..", "config.yaml")
with open(config_path) as f:
    cfg = yaml.safe_load(f)

set_seed(cfg.get("seed", 42))

dev = cfg.get("device", "auto")
if dev == "auto":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
else:
    device = torch.device(dev)
print(f"Device: {device}  |  Config: {config_path}")

# ============================================================
# 数据加载
# ============================================================
image_path = []
labels = []
data_root = Path(script_dir) / cfg["data"]["root"]
root = data_root
for i in root.rglob("*.jpg"):
    image_path.append(str(i))
    labels.append(i.parent.name)

data_df = pd.DataFrame(zip(image_path, labels), columns = ["image_path", "labels"])

train_ratio = cfg["data"]["train_ratio"]
val_ratio = cfg["data"]["val_ratio"]
train = data_df.sample(frac=train_ratio)
left = data_df.drop(train.index)
val  = left.sample(frac=val_ratio / (1 - train_ratio))
test = left.drop(val.index)

label_encoder = LabelEncoder()
label_encoder.fit(data_df['labels'])
print(f"类别: {label_encoder.classes_}")

hy_size = cfg["models"]["hybrid"]["input_size"]
hybrid_transform = transforms.Compose([
    transforms.Resize((hy_size, hy_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================================
# Dataset
# ============================================================
class ImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.label = torch.tensor(label_encoder.transform(dataframe['labels']), dtype=torch.long).to(device)

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        label = self.label[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image).to(device)
        return image, label

hy_train  = ImageDataset(train, hybrid_transform)
hy_val    = ImageDataset(val,   hybrid_transform)
hy_test   = ImageDataset(test,  hybrid_transform)

# ============================================================
# 模型 2：ResNet50 骨干 + 三层 CNN 
# ============================================================
class ExtraCNNHead(nn.Module):
    def __init__(self, in_c=2048, n_cls=3):
        super().__init__()
        self.cnn_stack = nn.Sequential(
            nn.Conv2d(in_c, 1024, 3, padding=1), nn.BatchNorm2d(1024), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(1024, 512,  3, padding=1), nn.BatchNorm2d(512),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(512,  256,  3, padding=1), nn.BatchNorm2d(256),  nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.cls  = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, n_cls)
        )

    def forward(self, x):
        x = self.cnn_stack(x)
        x = self.pool(x)
        return self.cls(x)

class HybridNet(nn.Module):
    def __init__(self, n_cls=3):
        super().__init__()
        bk_name = cfg["models"]["hybrid"]["backbone"]
        bk_fn = getattr(models, bk_name)
        backbone = nn.Sequential(*list(bk_fn(weights="DEFAULT").children())[:-2])
        self.backbone = backbone
        self.head = ExtraCNNHead(2048, n_cls)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)

# ============================================================
# 训练函数（两个模型共用）
# ============================================================
def train_model(model, name, train_loader, val_loader, test_loader, epochs, lr=1e-4):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # 只训练 requires_grad=True 的参数
    trainable = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(trainable, lr=lr)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "test_acc": []}

    for epoch in range(epochs):
        # --- 训练 ---
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = total_loss / len(train_loader)
        train_acc  = correct / total

        # --- 验证 ---
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs).squeeze(1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc  = val_correct / val_total

        # --- 测试 ---
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs).squeeze(1)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        test_acc = test_correct / test_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["test_acc"].append(test_acc)

        print(f"[{name}] Epoch {epoch:2d} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | "
              f"train_acc: {train_acc:.4f} | val_acc: {val_acc:.4f} | test_acc: {test_acc:.4f}")

    return history

# ============================================================
# 训练 2：Hybrid（ResNet50 骨干 + CNN 头）
# ============================================================
print()
print("=" * 60)
print("训练 [Hybrid] — ResNet50骨干 + 3层CNN头（骨干冻结）")
print("=" * 60)
hy_bs = cfg["models"]["hybrid"]["batch_size"]
hy_loader  = DataLoader(hy_train, batch_size=hy_bs, shuffle=True)
hy_v_loader = DataLoader(hy_val,  batch_size=hy_bs, shuffle=False)
hy_t_loader = DataLoader(hy_test, batch_size=hy_bs, shuffle=False)

hy_model = HybridNet(n_cls=len(data_df['labels'].unique()))
# 冻结骨干参数
for p in hy_model.backbone.parameters():
    p.requires_grad = False
print(f"  骨干参数: {sum(p.numel() for p in hy_model.backbone.parameters()):,} (冻结)")
print(f"  头部参数: {sum(p.numel() for p in hy_model.head.parameters()):,} (可训练)")

hy_hist = train_model(hy_model, "Hybrid",
                      hy_loader, hy_v_loader, hy_t_loader,
                      epochs=cfg["models"]["hybrid"]["epochs"], lr=float(cfg["models"]["hybrid"]["lr"]))
torch.save(hy_model.state_dict(), cfg["output"]["hybrid_model"])

# ============================================================
# 对比
# ============================================================
print()
print("=" * 60)
print("           Hybrid 训练结果")
print("=" * 60)
print(f"{'Metric':<22} {'Hybrid':>12}")
print("-" * 36)
best_acc = max(hy_hist['test_acc'])
final_acc = hy_hist['test_acc'][-1]
print(f"{'Best Test Acc':<22} {best_acc:>11.2%}")
print(f"{'Final Test Acc':<22} {final_acc:>11.2%}")
print(f"{'Epochs':<22} {cfg['models']['hybrid']['epochs']:>12}")
print(f"{'Input Size':<22} {cfg['models']['hybrid']['input_size']:>12}")
print(f"{'Params (trainable)':<22} {sum(p.numel() for p in hy_model.parameters() if p.requires_grad):>12,}")

# 画对比图
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(hy_hist["train_acc"],  label="Hybrid train", ls="--")
plt.plot(hy_hist["val_acc"],    label="Hybrid val")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(hy_hist["train_loss"],  label="Hybrid train", ls="--")
plt.plot(hy_hist["val_loss"],    label="Hybrid val")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Loss")

plt.tight_layout()
plt.savefig(cfg["output"]["comparison_chart"], dpi=150)
print(f"\n对比图已保存 → {cfg['output']['comparison_chart']}")
