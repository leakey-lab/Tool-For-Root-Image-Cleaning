import os
import re
from collections import defaultdict
from PIL import Image
import imagehash  # type: ignore
import concurrent.futures


total_supported_threads = 16


def compute_hash(filename):
    try:
        image = Image.open(filename).convert("L")  # Grey Scale converison
        image_hash = imagehash.whash(image)
        image.close()
        return image_hash
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None


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

#     # ThreadPoolExecutor to compute hashes in parallel
#     with concurrent.futures.ThreadPoolExecutor(
#         max_workers=total_supported_threads
#     ) as executor:
#         future_to_file = {
#             executor.submit(compute_hash, file_path): (tube_len, file_path)
#             for tube_len, file_path in files
#         }
#         for future in concurrent.futures.as_completed(future_to_file):
#             tube_len, file_path = future_to_file[future]
#             hash_value = future.result()
#             if hash_value is not None:
#                 files_dict[int(tube_len[0])][int(tube_len[1])].append(
#                     (file_path, hash_value)
#                 )

#     # Finding duplicates and grouping them
#     duplicates = []
#     processed = set()
#     for tube_number, lengths in files_dict.items():
#         for length_number, files in lengths.items():

#             group = set()
#             for file, hash1 in files:
#                 if file in processed:
#                     continue  # Skip if this file has already been processed as a duplicate
#                 processed.add(file)
#                 for i in range(length_number - 1, length_number + 2):
#                     if i in lengths and i != length_number:
#                         for file2, hash2 in lengths[i]:
#                             if hash1 - hash2 <= 0.01 and file2 not in processed:
#                                 group.add(file2)
#                                 processed.add(file2)
#                 if group:
#                     group.add(file)
#                     duplicates.append(group)
#                     group = set()
#     return duplicates


def find_duplicates(directory, total_supported_threads=4):
    # Regex to extract tube and length numbers
    pattern = re.compile(r"T(\d{3})_L(\d{3})")

    # Dictionary to hold files by tube number and length number
    files_dict = defaultdict(lambda: defaultdict(lambda: None))

    # Preparing list of filenames and paths
    files = []
    for filename in os.listdir(directory):
        match = pattern.search(filename)
        if match:
            file_path = os.path.join(directory, filename)
            tube_num, length_num = match.groups()
            files.append(((int(tube_num), int(length_num)), file_path))

    # Determine the min and max length numbers for each tube
    tube_lengths = defaultdict(list)
    for (tube_num, length_num), file_path in files:
        tube_lengths[tube_num].append(length_num)

    for tube_num, lengths in tube_lengths.items():
        min_length = min(lengths)
        max_length = max(lengths)
        for length_num in range(min_length, max_length + 1):
            files_dict[tube_num][length_num] = None

    # Fill the dictionary with actual files
    for (tube_num, length_num), file_path in files:
        files_dict[tube_num][length_num] = file_path

    # Compute hashes in parallel
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=total_supported_threads
    ) as executor:
        future_to_file = {
            executor.submit(compute_hash, file_path): (tube_num, length_num)
            for tube_num, lengths in files_dict.items()
            for length_num, file_path in lengths.items()
            if file_path is not None
        }
        for future in concurrent.futures.as_completed(future_to_file):
            tube_num, length_num = future_to_file[future]
            hash_value = future.result()
            if hash_value is not None:
                files_dict[tube_num][length_num] = (
                    files_dict[tube_num][length_num],
                    hash_value,
                )

    # Finding duplicates and grouping them
    duplicates = []
    processed = set()
    for tube_number, lengths in files_dict.items():
        sorted_lengths = sorted(lengths.keys())
        min_length, max_length = sorted_lengths[0], sorted_lengths[-1]

        for length_number in sorted_lengths:
            current_file_data = lengths[length_number]
            if current_file_data is None or length_number in processed:
                continue

            file, hash1 = current_file_data
            if file in processed:
                continue  # Skip if this file has already been processed as a duplicate
            processed.add(file)

            # Check within the range of -2 to +3 indices
            valid_comparison = True
            for offset in range(-2, 3):
                neighbor_length = length_number + offset
                if not (
                    min_length <= neighbor_length <= max_length
                    and lengths[neighbor_length] is not None
                ):
                    valid_comparison = False
                    break

            if valid_comparison:
                group = []
                for offset in range(-2, 3):
                    neighbor_length = length_number + offset
                    neighbor_file_data = lengths[neighbor_length]
                    if (
                        neighbor_file_data is not None
                        and neighbor_length != length_number
                    ):
                        neighbor_file, hash2 = neighbor_file_data
                        if hash1 - hash2 <= 1 and neighbor_file not in processed:
                            group.append(neighbor_file)
                            processed.add(neighbor_file)
                if group:
                    group.append(file)
                    duplicates.append(group)
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
