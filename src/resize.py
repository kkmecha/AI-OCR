import os
from PIL import Image, ImageOps
from tqdm import tqdm

def resize_and_pad_image(img, target_size=(128, 128)):
    # アスペクト比を維持してリサイズ
    img.thumbnail(target_size, Image.LANCZOS)
    
    # 背景を白で塗りつぶし、余白を追加して目的のサイズにする
    background = Image.new("L", target_size, color=255)  # グレースケールの白背景
    img_w, img_h = img.size
    bg_w, bg_h = target_size
    offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
    background.paste(img, offset)
    
    return background

def resize_images_in_directories(input_dirs, target_size=(128, 128)):
    for input_dir in input_dirs:
        base_dir, folder_name = os.path.split(input_dir)
        output_dir = os.path.join(base_dir, f"{folder_name}_re_png")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"Processing directory: {input_dir}")
        for filename in tqdm(os.listdir(input_dir), desc=f"Resizing images in {folder_name}"):
            if filename.endswith(".png"):
                img_path = os.path.join(input_dir, filename)
                img = Image.open(img_path)

                # 画像のリサイズとパディング
                img_resized = resize_and_pad_image(img, target_size)

                # 出力先パスを設定し、画像を保存
                output_path = os.path.join(output_dir, filename)
                img_resized.save(output_path)
        print(f"Finished processing {folder_name} directory.")
# 使用例
input_dirs = [
    '/home/kenshin/Desktop/Application/data/train_grayscales',
    '/home/kenshin/Desktop/Application/data/valid_grayscales',
    '/home/kenshin/Desktop/Application/data/test_grayscales'
]
resize_images_in_directories(input_dirs, target_size=(256, 256))  # 任意のサイズを指定
