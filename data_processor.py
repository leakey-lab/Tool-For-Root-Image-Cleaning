import os
import re
from collections import defaultdict


def process_images(directory):
    tube_counts = defaultdict(int)
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            match = re.search(r"T(\d{3})_L(\d{3})", filename)
            if match:
                tube_number = match.group(1)
                tube_counts[tube_number] += 1
    return tube_counts


# TODO: Empty Images, i.e Images with no roots or anyother artifact.

# To do this we are suing otsu thresholding.
