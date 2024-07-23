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
        dirHash = imagehash.dhash(image)
        perHash = imagehash.phash(image)
        image.close()
        return dirHash, perHash
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None


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

            file, (dhash1, phash1) = current_file_data
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
                        neighbor_file, (dhash2, phash2) = neighbor_file_data
                        if (
                            dhash1 - dhash2 <= 5 and phash1 - phash2 <= 5
                        ) and neighbor_file not in processed:
                            group.append(neighbor_file)
                            processed.add(neighbor_file)
                if group:
                    group.append(file)
                    duplicates.append(group)
    return duplicates
