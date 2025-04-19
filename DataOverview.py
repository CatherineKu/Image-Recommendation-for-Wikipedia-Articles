import os
import json
import random

# Define the dataset path
dataset_path = "/Users/chuhanku/Documents/GitHub/Image-Recommendation-for-Wikipedia-Articles/data_3"  # Update this with your actual path

# Store articles based on image count
no_image_articles = []
article_image_counts = {}

for article in os.listdir(dataset_path):
    article_path = os.path.join(dataset_path, article)
    img_folder_path = os.path.join(article_path, "img")
    meta_json_path = os.path.join(img_folder_path, "meta.json")

    num_images = 0  # Default to no images

    if os.path.isdir(img_folder_path):
        # If meta.json exists, count the number of images from "img_meta"
        if os.path.exists(meta_json_path):
            with open(meta_json_path, "r", encoding="utf-8") as f:
                try:
                    meta_data = json.load(f)  # Load the JSON

                    # If JSON is stored as a string, decode it again
                    if isinstance(meta_data, str):
                        meta_data = json.loads(meta_data)

                    # Ensure "img_meta" exists and is a list
                    if "img_meta" in meta_data and isinstance(meta_data["img_meta"], list):
                        num_images = len(meta_data["img_meta"])  # Count filenames

                except (json.JSONDecodeError, TypeError):
                    pass  # Ignore invalid JSON files

    # Store articles with no images
    if num_images == 0:
        no_image_articles.append(article)
    else:
        article_image_counts[article] = num_images

# Debugging prints
print(f"Total articles found: {len(os.listdir(dataset_path))}")
print(f"Total articles with no images: {len(no_image_articles)}")

# Select up to 15 random articles with no images
random_no_image_articles = random.sample(no_image_articles, min(15, len(no_image_articles)))

# Get top 15 articles with the most images
top_15_articles = sorted(article_image_counts.items(), key=lambda x: x[1], reverse=True)[:15]

# Print results
print("\n15 Random Wikipedia Articles with No Images:")
for article in random_no_image_articles:
    print(article)

print("\nTop 15 Wikipedia Articles with the Most Images:")
for article, count in top_15_articles:
    print(f"{article}: {count} images")
