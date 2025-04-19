import os
import json
import matplotlib.pyplot as plt

# Define the dataset path
dataset_path = "/Users/chuhanku/Documents/GitHub/Image-Recommendation-for-Wikipedia-Articles/WikiFeaturedArticlesDataset/data"  # Update this with your actual dataset path

'''
# Collect the number of images per Wikipedia article
article_image_counts = []

for article in os.listdir(dataset_path):
    article_path = os.path.join(dataset_path, article)
    img_folder_path = os.path.join(article_path, "img")
    meta_json_path = os.path.join(img_folder_path, "meta.json")

    num_images = 0  # Default to 0

    if os.path.isdir(img_folder_path) and os.path.exists(meta_json_path):
        with open(meta_json_path, "r", encoding="utf-8") as f:
            try:
                meta_data = json.load(f)

                # If JSON is stored as a string, decode it again
                if isinstance(meta_data, str):
                    meta_data = json.loads(meta_data)

                # Ensure "img_meta" exists and is a list
                if "img_meta" in meta_data and isinstance(meta_data["img_meta"], list):
                    num_images = len(meta_data["img_meta"])  # Count filenames

            except (json.JSONDecodeError, TypeError):
                pass  # Ignore invalid JSON files

    article_image_counts.append(num_images)  # Add to list

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(article_image_counts, bins=50, edgecolor="black", alpha=0.7)
plt.xlabel("Number of Images per Article")
plt.ylabel("Number of Articles")
plt.title("Distribution of Images per Wikipedia Article")
plt.yscale("log")  # Log scale to better visualize skewed distributions
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show the plot
plt.show()

'''
'''
def remove_original_suffix(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".ORIGINAL"):
                old_path = os.path.join(dirpath, filename)
                new_filename = filename[:-9]  # 去掉最后的 ".ORIGINAL"（长度9）
                new_path = os.path.join(dirpath, new_filename)
                os.rename(old_path, new_path)
                print(f"✅ Renamed: {filename} -> {new_filename}")

# 替换成你的项目根路径
remove_original_suffix(dataset_path)

import os

import os

def clean_extra_jpg_suffix(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            lower_name = filename.lower()
            if lower_name.count(".jpg") >= 2:
                # 保留第一个 .jpg 的大小写形式
                split_point = lower_name.find(".jpg") + 4
                cleaned_filename = filename[:split_point]

                old_path = os.path.join(dirpath, filename)
                new_path = os.path.join(dirpath, cleaned_filename)

                if old_path != new_path:
                    try:
                        os.rename(old_path, new_path)
                        print(f"✅ Renamed: {filename} → {cleaned_filename}")
                    except Exception as e:
                        print(f"❌ Failed to rename {filename}: {e}")

clean_extra_jpg_suffix(dataset_path)


def clean_jpg_original_filename(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            lower_name = filename.lower()

            # 必须同时满足：包含多个 .jpg，并以 .original 结尾
            if lower_name.endswith(".jpg.original"):
                # 保留第一个 .jpg（包含大小写）
                first_jpg_index = lower_name.find(".jpg") + 4
                cleaned_filename = filename[:first_jpg_index]

                old_path = os.path.join(dirpath, filename)
                new_path = os.path.join(dirpath, cleaned_filename)

                if old_path != new_path:
                    try:
                        os.rename(old_path, new_path)
                        print(f"✅ Renamed: {filename} → {cleaned_filename}")
                    except Exception as e:
                        print(f"❌ Failed to rename {filename}: {e}")


clean_jpg_original_filename(dataset_path)


# Search for missing data
def find_empty_img_folders(root_dir, image_exts={'.jpg', '.jpeg', '.png', '.webp'}):
    empty_img_folders = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if os.path.basename(dirpath) == 'img':
            has_image = any(
                os.path.splitext(file)[1].lower() in image_exts
                for file in filenames
            )
            if not has_image:
                empty_img_folders.append(dirpath)

    print(f"\n📂 Found {len(empty_img_folders)} empty 'img' folders:\n")
    for folder in empty_img_folders:
        print(f"❌ {folder}")
'''
# 替换成你的根目录路径（比如 "data"）
#find_empty_img_folders(dataset_path)


'''
# 检查数据问题
import os
import json
from PIL import Image, UnidentifiedImageError

ROOT_DIR = dataset_path
MIN_WIDTH = 50 #设定阈值
MIN_HEIGHT = 50

bad_images = []
small_images = []
missing_texts = []
broken_json = []
weird_aspect_ratio = []

def is_text_valid(text):
    return text and isinstance(text, str) and len(text.strip()) > 20

for item in os.listdir(ROOT_DIR):
    item_path = os.path.join(ROOT_DIR, item)
    if not os.path.isdir(item_path):
        continue

    # 检查图片
    img_folder = os.path.join(item_path, "img")
    if os.path.exists(img_folder):
        for fname in os.listdir(img_folder):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                fpath = os.path.join(img_folder, fname)
                try:
                    with Image.open(fpath) as img:
                        width, height = img.size
                        ratio = max(width / height, height / width)

                        if width < MIN_WIDTH or height < MIN_HEIGHT:
                            if ratio > 3:
                                weird_aspect_ratio.append(fpath)
                            else:
                                small_images.append(fpath)
                        elif ratio > 3:
                            weird_aspect_ratio.append(fpath)
                except UnidentifiedImageError:
                    bad_images.append(fpath)
    # 检查文本
    text_json_path = os.path.join(item_path, "text.json")
    if os.path.exists(text_json_path):
        try:
            with open(text_json_path, "r", encoding="utf-8") as f:
                raw = json.load(f)

                # 修复：如果是字符串，说明是字符串化的 JSON
                if isinstance(raw, str):
                    data = json.loads(raw)
                else:
                    data = raw

                if not is_text_valid(data.get("text", "")):
                    missing_texts.append(text_json_path)
        except Exception as e:
            broken_json.append((text_json_path, str(e)))

# 输出统计
print("Bad images:", len(bad_images))
print("Small images:", len(small_images))
print("Weird aspect ratio images:", len(weird_aspect_ratio))
print("Missing texts:", len(missing_texts))
print("Broken JSON files:", len(broken_json))

# 输出具体路径（可选）
if bad_images:
    print("\nBad image files:")
    for path in bad_images:
        print(path)

if small_images:
    print("\nSmall image files:")
    for path in small_images:
        print(path)

if missing_texts:
    print("\nMissing or invalid text.json files:")
    for path in missing_texts:
        print(path)

if broken_json:
    print("\nBroken JSON files:")
    for path, err in broken_json:
        print(f"{path} -- {err}")
'''

'''
ROOT_DIR = "data"  # 替换为你的数据目录
no_image_folders = []

for folder in os.listdir(ROOT_DIR):
    folder_path = os.path.join(ROOT_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    img_folder = os.path.join(folder_path, "img")

    if not os.path.exists(img_folder):
        no_image_folders.append(folder)
        continue

    # 检查是否有除了 json 以外的文件
    has_non_json = any(
        not fname.lower().endswith(".json")
        for fname in os.listdir(img_folder)
    )

    if not has_non_json:
        no_image_folders.append(folder)

# 输出结果
print("Total folders with NO actual image files:", len(no_image_folders))
print("Folders without image files:")
for name in no_image_folders:
    print("-", name)
'''








