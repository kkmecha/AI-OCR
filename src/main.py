import os
import json
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

def check_gpu():
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
    else:
        device = 'cpu'
        print("GPU is not available. Using CPU.")
    return device

# データの前処理と変換
def inkml_to_image(inkml_file, output_dir, scale_factor=1.0, line_width=2):
    # InkMLファイルの解析
    tree = ET.parse(inkml_file)
    root = tree.getroot()
    traces = root.findall(".//{http://www.w3.org/2003/InkML}trace")
    all_points = []

    # InkMLファイルから座標を取得
    for trace in traces:
        points = trace.text.strip().split(',')
        trace_points = []
        for point in points:
            coords = point.split()
            if len(coords) >= 2:
                try:
                    x, y = map(float, coords[:2])  # X, Y座標を取得
                    trace_points.append((x, y))
                except ValueError:
                    print(f"Invalid point format: {point}")
        all_points.append(trace_points)

    if not all_points:
        return None

    # すべてのポイントからX, Yの範囲を計算
    all_xs = [x for trace in all_points for (x, y) in trace]
    all_ys = [y for trace in all_points for (x, y) in trace]

    min_x, max_x = min(all_xs), max(all_xs)
    min_y, max_y = min(all_ys), max(all_ys)

    # スケーリングに基づいた画像サイズを計算
    width = int((max_x - min_x) * scale_factor) + 10  # +10でマージンを追加
    height = int((max_y - min_y) * scale_factor) + 10

    # 出力ファイルのパスを先に定義
    output_file = os.path.join(output_dir, os.path.basename(inkml_file).replace('.inkml', '.png'))

    # CPUで描画する場合
    img = Image.new('L', (width, height), 255)  # 'L'はグレースケール、白背景
    draw = ImageDraw.Draw(img)
    for trace in all_points:
        scaled_trace = [(int((x - min_x) * scale_factor), int((y - min_y) * scale_factor)) for (x, y) in trace]
        draw.line(scaled_trace, fill=0, width=line_width)  # 線として描画
    img.save(output_file)

    return output_file

def extract_label_from_inkml(inkml_file):
    tree = ET.parse(inkml_file)
    root = tree.getroot()
    label_elem = root.find(".//{http://www.w3.org/2003/InkML}annotation[@type='label']")
    if label_elem is not None:
        return label_elem.text
    return 'unknown'

def process_inkml_file(inkml_file, output_dir):
    try:
        image_file = inkml_to_image(inkml_file, output_dir)
        if image_file:
            label = extract_label_from_inkml(inkml_file)
            return {"filename": os.path.basename(image_file), "label": label}
    except Exception as e:
        print(f"Error processing {inkml_file}: {e}")
    return None

def process_inkml_files(input_dir, output_dir, json_file):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    processed_files = []

    inkml_files = [os.path.join(root, file) for root, _, files in os.walk(input_dir) for file in files if file.endswith('.inkml')]

    # 全体の進捗を表示するためのtqdmバー
    total_files = len(inkml_files)
    with tqdm(total=total_files, desc="Overall Progress") as overall_progress:
        for inkml_file in inkml_files:
            result = process_inkml_file(inkml_file, output_dir)
            if result:
                processed_files.append(result)
            overall_progress.update(1)  # ファイル処理が終わったら進捗を更新

    with open(json_file, 'w') as f:
        json.dump(processed_files, f, indent=4)

#  CNNモデルの定義
class CNNModel(nn.Module):
    def __init__(self, num_classes, device):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 32 * 32, 256)  # 256x256の場合のサイズを更新
        self.fc2 = nn.Linear(256, num_classes)  # クラス数に応じて調整
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        x = torch.relu(self.conv1(x))
        x = nn.MaxPool2d(2)(x)
        x = torch.relu(self.conv2(x))
        x = nn.MaxPool2d(2)(x)
        x = torch.relu(self.conv3(x))
        x = nn.MaxPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# データセットクラス
class HandwritingDataset(Dataset):
    def __init__(self, data_dir, json_file, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        print("[Info] Loading labels from JSON file...")
        with open(json_file, 'r') as f:
            self.labels = json.load(f)
        print(f"[Info] Successfully loaded {len(self.labels)} labels.")

        # ラベルを数値にエンコード
        self.label_encoder = LabelEncoder()
        self.all_labels = [item['label'] for item in self.labels]
        self.label_encoder.fit(self.all_labels)

        # 画像とラベルの辞書を作成
        self.labels_dict = {
            os.path.join(data_dir, item['filename']): self.label_encoder.transform([item['label']])[0]
            for item in self.labels
        }
        self.image_paths = list(self.labels_dict.keys())
        
        print(f"[Info] Number of unique labels: {len(set(self.all_labels))}")
        print(f"[Info] Sample labels: {self.all_labels[:5]}")
        print(f"[Info] Number of images found: {len(self.image_paths)}")
        print(f"[Info] Sample image paths: {self.image_paths[:5]}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        print(f"[Info] Accessing image: {img_path}")

        try:
            with Image.open(img_path) as image:
                image = image.convert('L')  # グレースケールに変換
                label = self.labels_dict[img_path]  # ラベルを取得

                if self.transform:
                    image = self.transform(image)  # 前処理があれば適用
        except Exception as e:
            print(f"[Error] Failed to open image {img_path}: {e}")
            raise  # エラーを再発生させて呼び出し元に伝える

        print(f"[Info] Loaded image of size: {image.size}, with label: {label}")
        return image, label


# 訓練と評価の関数
def train_and_evaluate(data_dir, json_file, batch_size=32, epochs=10, learning_rate=0.001):
    device = check_gpu()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = HandwritingDataset(data_dir, json_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    num_classes = len(dataset.label_encoder.classes_)
    print(f"[Training] Number of classes: {num_classes}")
    model = CNNModel(num_classes, device).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # バッチごとの損失と精度を表示
            if (batch_idx + 1) % 10 == 0:
                batch_loss = loss.item()
                batch_accuracy = 100 * (predicted == labels).sum().item() / labels.size(0)
                print(f"Batch {batch_idx + 1}/{len(dataloader)} - Loss: {batch_loss:.4f}, Accuracy: {batch_accuracy:.2f}%")

        epoch_loss = running_loss / len(dataloader.dataset)
        accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

    print("Training complete.")

def load_dataset(data_dir, json_file):
    # JSONファイルからラベルを読み込み
    try:
        with open(json_file, 'r') as f:
            labels = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    print(f"[Info] Loaded {len(labels)} labels from JSON.")

    # 画像パスとラベルのリストを作成
    for item in labels:
        img_path = os.path.join(data_dir, item['filename'])
        print(f"[Info] Checking image: {img_path}")

        # ラベルを表示
        label = item['label']
        print(f"[Info] Corresponding label: {label}")

        # 画像を開いて確認
        if os.path.exists(img_path):
            try:
                with Image.open(img_path) as img:
                    print(f"[Info] Successfully opened image: {img_path}, Size: {img.size}")
            except Exception as e:
                print(f"[Error] Could not open image {img_path}: {e}")
        else:
            print(f"[Warning] Image does not exist: {img_path}")

if __name__ == "__main__":
    # input_dirs = ['/home/kenshin/Desktop/Application/data/train', '/home/kenshin/Desktop/Application/data/valid', '/home/kenshin/Desktop/Application/data/test']
    # output_dirs = ['/home/kenshin/Desktop/Application/data/train_grayscales', '/home/kenshin/Desktop/Application/data/valid_grayscales', '/home/kenshin/Desktop/Application/data/test_grayscales']
    # json_files = ['/home/kenshin/Desktop/Application/data/train_labels.json', '/home/kenshin/Desktop/Application/data/valid_labels.json', '/home/kenshin/Desktop/Application/data/test_labels.json']

    # for input_dir, output_dir, json_file in zip(input_dirs, output_dirs, json_files):
    #     process_inkml_files(input_dir, output_dir, json_file)
    
    # process_image_for_model('/home/kenshin/Desktop/Application/data/test_grayscales/0a0b310001bedb73.png')

    # CNNモデルの訓練
    # data_dirs = ['/home/kenshin/Desktop/Application/data/train_grayscales_re_png', '/home/kenshin/Desktop/Application/data/valid_grayscales_re_png']
    # json_files = ['/home/kenshin/Desktop/Application/data/train_labels.json', '/home/kenshin/Desktop/Application/data/valid_labels.json']

    # for data_dir, json_file in zip(data_dirs, json_files):
    #     train_and_evaluate(data_dir, json_file)





    data_dir = '/home/kenshin/Desktop/Application/data/train_grayscales_re_png'
    json_file = '/home/kenshin/Desktop/Application/data/train_labels.json'

    dataset = HandwritingDataset(data_dir, json_file)

