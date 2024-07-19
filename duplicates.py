import os
import re
from collections import defaultdict
from PIL import Image
import imagehash  # type: ignore
import concurrent.futures


total_supported_threads = 16


def compute_hash(filename):
    try:
        image = Image.open(filename)
        image_hash = imagehash.whash(image)
        image.close()
        return image_hash
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None


def find_duplicates(directory):
    # Regex to extract tube and length numbers
    pattern = re.compile(r"T(\d{3})_L(\d{3})")

    # Dictionary to hold files by tube number and length number
    files_dict = defaultdict(lambda: defaultdict(list))

    # Preparing list of filenames and paths
    files = []
    for filename in os.listdir(directory):
        match = pattern.search(filename)
        if match:
            file_path = os.path.join(directory, filename)
            files.append((match.groups(), file_path))

    # ThreadPoolExecutor to compute hashes in parallel
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=total_supported_threads
    ) as executor:
        future_to_file = {
            executor.submit(compute_hash, file_path): (tube_len, file_path)
            for tube_len, file_path in files
        }
        for future in concurrent.futures.as_completed(future_to_file):
            tube_len, file_path = future_to_file[future]
            hash_value = future.result()
            if hash_value is not None:
                files_dict[int(tube_len[0])][int(tube_len[1])].append(
                    (file_path, hash_value)
                )

    # Finding duplicates and grouping them
    duplicates = []
    processed = set()
    for tube_number, lengths in files_dict.items():
        for length_number, files in lengths.items():

            group = set()
            for file, hash1 in files:
                if file in processed:
                    continue  # Skip if this file has already been processed as a duplicate
                processed.add(file)
                for i in range(length_number - 1, length_number + 2):
                    if i in lengths and i != length_number:
                        for file2, hash2 in lengths[i]:
                            if hash1 - hash2 <= 0.01 and file2 not in processed:
                                group.add(file2)
                                processed.add(file2)
                if group:
                    group.add(file)
                    duplicates.append(group)
                    group = set()
    return duplicates


# directory = (
#     r"F:\Summer Research at IGB\2023 EF 1st Imaging 1-10\2023 EF 1st Imaging 1-10"
# )
# duplicates = find_duplicates(directory)


# # Print or process duplicates
# for group in duplicates:
#     print("Duplicate group:")
#     for file in group:
#         print(file)


# import os
# import re
# from collections import defaultdict
# from concurrent.futures import ThreadPoolExecutor
# from PIL import Image
# import torch
# import torchvision.transforms as transforms
# from torchvision import models
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# import concurrent.futures

# total_supported_threads = 16


# # Function to extract image embeddings using a pre-trained CNN
# def extract_embedding(model, filename, transform):
#     try:
#         image = Image.open(filename).convert("RGB")
#         image = transform(image).unsqueeze(0)  # Add batch dimension
#         with torch.no_grad():
#             embedding = model(image).squeeze(0).flatten().numpy()
#         return embedding
#     except Exception as e:
#         print(f"Error processing {filename}: {e}")
#         return None


# def find_duplicates(directory):
#     # Regex to extract tube and length numbers
#     pattern = re.compile(r"T(\d{3})_L(\d{3})")

#     # Dictionary to hold files by tube number and length number
#     files_dict = defaultdict(lambda: defaultdict(list))

#     # Preparing list of filenames and paths
#     files = []
#     for filename in os.listdir(directory):
#         match = pattern.search(filename)
#         if match:
#             file_path = os.path.join(directory, filename)
#             files.append((match.groups(), file_path))

#     # Pre-trained model and transformation
#     model = models.resnet18(weights="ResNet18_Weights.DEFAULT")
#     model = torch.nn.Sequential(
#         *(list(model.children())[:-1])
#     )  # Remove the classification layer
#     model.eval()
#     transform = transforms.Compose(
#         [
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ]
#     )

#     # ThreadPoolExecutor to compute embeddings in parallel
#     with ThreadPoolExecutor(max_workers=total_supported_threads) as executor:
#         future_to_file = {
#             executor.submit(extract_embedding, model, file_path, transform): (
#                 tube_len,
#                 file_path,
#             )
#             for tube_len, file_path in files
#         }
#         for future in concurrent.futures.as_completed(future_to_file):
#             tube_len, file_path = future_to_file[future]
#             embedding = future.result()
#             if embedding is not None:
#                 files_dict[int(tube_len[0])][int(tube_len[1])].append(
#                     (file_path, embedding)
#                 )

#     # Finding duplicates and grouping them
#     duplicates = []
#     processed = set()
#     for tube_number, lengths in files_dict.items():
#         for length_number, files in lengths.items():
#             for file, emb1 in files:
#                 if file in processed:
#                     continue  # Skip if this file has already been processed as a duplicate
#                 processed.add(file)
#                 group = set()
#                 for i in range(length_number - 5, length_number + 6):
#                     if i in lengths and i != length_number:
#                         for file2, emb2 in lengths[i]:
#                             if (
#                                 cosine_similarity([emb1], [emb2])[0][0] > 0.95
#                                 and file2 not in processed
#                             ):
#                                 group.add(file2)
#                                 processed.add(file2)
#                 if group:
#                     group.add(file)
#                     duplicates.append(group)
#     return duplicates


# # directory = r".\2021 Final Root Images\2021 Final Root Images\2021 Energy Farm Sorghum Root Imaging\Data"
# directory = (
#     r"F:\Summer Research at IGB\2023 EF 1st Imaging 1-10\2023 EF 1st Imaging 1-10"
# )
# duplicates = find_duplicates(directory)

# from tqdm import tqdm

# # Print or process duplicates
# for group in tqdm(duplicates, desc="Processing duplicates"):
#     print("Duplicate group:")
#     for file in group:
#         print(file)
