from PIL import Image
import os

def measure_image_sizes(image_dir):
    # ディレクトリが存在するか確認
    if not os.path.isdir(image_dir):
        print(f"Error: '{image_dir}' is not a valid directory.")
        return

    # ディレクトリ内のすべてのファイルをループ
    for filename in os.listdir(image_dir):
        img_path = os.path.join(image_dir, filename)

        # ファイルが画像かどうかを確認
        if os.path.isfile(img_path) and (filename.endswith('.png') or filename.endswith('.jpeg')):
            try:
                with Image.open(img_path) as image:  # withステートメントで自動解放
                    width, height = image.size
                    print(f"{filename}: Width = {width}, Height = {height}")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

# 使用例
image_directory = '/home/kenshin/Desktop/Application/data/train_resize'  # 画像が保存されているディレクトリのパス
measure_image_sizes(image_directory)
