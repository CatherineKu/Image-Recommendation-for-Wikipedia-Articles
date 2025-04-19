import os
import json

'''
def build_train_file(data_dir, output_file):
    """
    Traverse the data_dir and for each subdirectory, process the file named "text.json".
    Extract the 'title' and 'text' fields from the JSON, concatenate them, and write the result
    into the output file (one passage per line).
    """
    with open(output_file, 'w', encoding='utf-8') as fout:
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.lower() != "text.json":
                    continue
                json_path = os.path.join(root, file)
                try:
                    with open(json_path, 'r', encoding='utf-8') as fin:
                        data = json.load(fin)
                except Exception as e:
                    print(f"Failed to load file {json_path}: {e}")
                    continue

                # If the loaded data is not a dict, try to convert it if it's a string.
                if not isinstance(data, dict):
                    if isinstance(data, str):
                        try:
                            data = json.loads(data)
                        except Exception as e:
                            print(f"File {json_path} does not contain a valid JSON object after re-parsing. Skipping.")
                            continue
                    else:
                        print(f"File {json_path} did not load as a dictionary. Skipping.")
                        continue

                title = data.get('title', '').strip()
                text = data.get('text', '').strip()

                if title and text:
                    passage = f"{title} {text}"
                else:
                    passage = title or text

                if passage:
                    fout.write(passage + "\n")
                else:
                    print(f"File {json_path} does not contain valid title or text. Skipping.")
    print(f"Training file created: {output_file}")

if __name__ == '__main__':
    data_directory = "data"          # The root folder containing the subdirectories for each passage.
    output_train_file = "data/train_bert_lm.txt"
    build_train_file(data_directory, output_train_file)
'''

import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split


def load_passage_data(raw_data_dir):
    """
    For each subfolder in raw_data_dir (each representing one passage),
    read the text.json file and extract the title and text.
    Returns a DataFrame with columns: 'passage_id', 'folder_name', 'title', 'text', 'combined_text'
    """
    passages = []
    # Each folder in raw_data_dir is a passage folder.
    for folder in os.listdir(raw_data_dir):
        folder_path = os.path.join(raw_data_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        json_path = os.path.join(folder_path, "text.json")
        if not os.path.exists(json_path):
            print(f"Folder {folder_path} does not have text.json, skipping.")
            continue
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # If the loaded data is a string (e.g., quoted JSON), re-parse it:
            if isinstance(data, str):
                data = json.loads(data)
        except Exception as e:
            print(f"Failed to load or parse {json_path}: {e}")
            continue
        # Extract title and text; strip whitespace.
        title = data.get("title", "").strip()
        text = data.get("text", "").strip()
        # Create a combined text string (modify as you wish; here we simply join them).
        combined_text = f"{title} {text}".strip()
        if not combined_text:
            print(f"{json_path} does not contain valid title or text, skipping.")
            continue

        passages.append({
            "passage_id": folder,  # use folder name as an ID; adjust if you have a different unique identifier
            "folder_name": folder,
            "title": title,
            "text": text,
            "combined_text": combined_text
        })

    df = pd.DataFrame(passages)
    return df


def split_and_save_data(raw_data_dir, output_data_dir, test_size=0.2, random_state=42):
    """
    Splits the raw data into train and test sets and saves them in the format expected by the util file.

    The resulting structure will be:
      output_data_dir/
         train/
           full/
             train_data.feather
         test/
           original/
             test.tsv
             test_caption_list.csv
    """
    df = load_passage_data(raw_data_dir)
    print(f"Total passages loaded: {len(df)}")
    if df.empty:
        print("No valid passages found. Exiting.")
        return

    # Perform a train-test split
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    print(f"Train passages: {len(train_df)}, Test passages: {len(test_df)}")

    # Create directories if they do not exist
    train_full_dir = os.path.join(output_data_dir, "train", "full")
    test_original_dir = os.path.join(output_data_dir, "test", "original")
    os.makedirs(train_full_dir, exist_ok=True)
    os.makedirs(test_original_dir, exist_ok=True)

    # Save train DataFrame to a Feather file.
    train_file = os.path.join(train_full_dir, "train_data.feather")
    train_df.to_feather(train_file)
    print(f"Training data saved to {train_file}")

    # For test data, we need two files: test.tsv and test_caption_list.csv.
    # Here, we assume test.tsv will contain at least an 'id' and possibly the folder name or title,
    # while test_caption_list.csv will contain the 'combined_text' (i.e., the caption).
    test_tsv = os.path.join(test_original_dir, "test.tsv")
    test_caption_csv = os.path.join(test_original_dir, "test_caption_list.csv")

    # Prepare test.tsv – for this example, we'll include an ID and the title.
    test_tsv_df = test_df[["passage_id", "title"]].copy()
    test_tsv_df.rename(columns={"passage_id": "id", "title": "name"}, inplace=True)
    test_tsv_df.to_csv(test_tsv, sep="\t", index=False)
    print(f"Test TSV data saved to {test_tsv}")

    # Prepare test_caption_list.csv – here we'll use the combined text.
    test_caption_df = test_df[["combined_text"]].copy()
    test_caption_df.to_csv(test_caption_csv, index=False)
    print(f"Test Caption data saved to {test_caption_csv}")


if __name__ == "__main__":
    # Assuming your raw dataset is in "data" where each subfolder is a passage.
    raw_data_dir = "data"  # Modify this if your raw data is in a different folder.
    # The output directory should match what the util file expects.
    output_data_dir = "data_reformat"  # This creates data/train/full and data/test/original in the "data" folder.
    split_and_save_data(raw_data_dir, output_data_dir)
