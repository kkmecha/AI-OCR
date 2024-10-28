import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from main import HandwritingDataset  # 例：データセットクラスを定義したファイル名を指定

data_dir = '/home/kenshin/Desktop/Application/data/train_grayscales'
json_file = '/home/kenshin/Desktop/Application/data/train_labels.json'

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

dataset = HandwritingDataset(data_dir, json_file, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

print("[TEST] Starting batch loading...")

for idx, (images, labels) in enumerate(dataloader):
    print(f"[TEST BATCH] Batch {idx + 1} - Images shape: {images.shape}, Labels: {labels}")
    if idx >= 5:  # 最初の5バッチだけ確認
        break
