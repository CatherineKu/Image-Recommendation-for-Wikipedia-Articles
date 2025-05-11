import os
import json
import torch
import ssl
#ssl._create_default_https_context = ssl._create_unverified_context
import clip
from PIL import Image
from tqdm import tqdm

'''
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

ROOT_DIR = '/Users/chuhanku/Documents/GitHub/Image-Recommendation-for-Wikipedia-Articles/data'
MAX_FOLDER = 10
count = 0

for folder in tqdm(os.listdir(ROOT_DIR)):

    if count >= MAX_FOLDER:
        break

    folder_path = os.path.join(ROOT_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    img_dir = os.path.join(folder_path, "img")
    meta_path = os.path.join(folder_path, "meta.json")
    if not os.path.exists(img_dir):
        continue

    # 加载 meta
    meta_map = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_list = json.load(f)
                for entry in meta_list:
                    fname = entry.get("filename")
                    if fname:
                        meta_map[fname] = entry
        except:
            pass

    image_output = {}

    for fname in os.listdir(img_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue

        fpath = os.path.join(img_dir, fname)
        try:
            image = Image.open(fpath).convert("RGB")
            image_input = preprocess(image).unsqueeze(0).to(device)

            with torch.no_grad():
                feat = model.encode_image(image_input)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                feat_np = feat.cpu().numpy()[0]

            image_output[fname] = {
                "features": feat_np.tolist(),
                "title": meta_map.get(fname, {}).get("title", ""),
                "description": meta_map.get(fname, {}).get("description", ""),
                "url": meta_map.get(fname, {}).get("url", "")
            }

        except Exception as e:
            print(f"⚠️ Error processing {fpath}: {e}")

    # 保存为 image.json
    out_path = os.path.join(folder_path, "image.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(image_output, f, indent=2, ensure_ascii=False)

print("Done: image vectors + metadata saved.")

'''

'''
def combine_image_features(raw_data_dir, output_mapping_file):
    """
    Traverse each passage folder in raw_data_dir.
    For each folder, read the "image.json" file which stores precomputed CLIP features.
    Then, create a mapping from the passage folder name to its features and save
    the mapping as a JSON file.

    Args:
        raw_data_dir (str): The root directory containing passage folders.
        output_mapping_file (str): The path where the combined mapping file will be saved.
    """
    mapping = {}

    # Iterate over each folder in the raw_data_dir.
    for passage in os.listdir(raw_data_dir):
        passage_dir = os.path.join(raw_data_dir, passage)
        if not os.path.isdir(passage_dir):
            continue  # Skip if not a folder

        # Look for the image features file in the passage folder.
        image_feature_file = os.path.join(passage_dir, "image.json")
        if not os.path.exists(image_feature_file):
            print(f"No image.json file found in {passage_dir}, skipping.")
            continue

        try:
            with open(image_feature_file, "r", encoding="utf-8") as f:
                # Assume the file content is already a JSON object (list/dict of features)
                features = json.load(f)
        except Exception as e:
            print(f"Error reading {image_feature_file}: {e}")
            continue

        # Add the features to the mapping with the passage folder name as key.
        mapping[passage] = features

    # Save the consolidated mapping to the output file.
    with open(output_mapping_file, "w", encoding="utf-8") as out_f:
        json.dump(mapping, out_f, indent=2)

    print(f"Combined image features mapping saved to {output_mapping_file}")


if __name__ == "__main__":
    raw_data_dir = "/Users/chuhanku/Documents/GitHub/Image-Recommendation-for-Wikipedia-Articles/data"  # Root directory containing your passage folders.
    output_mapping_file = os.path.join("/Users/chuhanku/Documents/GitHub/Image-Recommendation-for-Wikipedia-Articles/data_reformat", "mapped_image_features.json")
    combine_image_features(raw_data_dir, output_mapping_file)
'''
import json

def count_images(mapping_path):
    # Load the consolidated mapping
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)

    total_passages = len(mapping)
    total_images = 0

    #print("Passage Name\t# Images")
    #print("-" * 30)
    for passage, images in list(mapping.items()):
        # images is expected to be a dict mapping image filenames -> feature objects
        num_imgs = len(images)
        total_images += num_imgs
        #print(f"{passage}\t{num_imgs}")

    print(("-" * 30))
    print(f"Total passages: {total_passages}")
    print(f"Total images:   {total_images}")

if __name__ == "__main__":
    mapping_file = "data_reformat/mapped_image_features.json"  # or your actual path
    count_images(mapping_file)

