import tkinter as tk
from tkinter import filedialog
import os
import json

CONFIG_FILE = "directory_config.json"


def load_last_directory():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
            return config.get("last_directory", "")
    return ""


def save_last_directory(directory):
    config = {"last_directory": directory}
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f)


def select_folder():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes("-topmost", True)  # Bring the dialog above all windows

    initial_dir = load_last_directory()
    folder_path = filedialog.askdirectory(
        initialdir=initial_dir
    )  # Open the directory selection dialog

    if folder_path:  # Only save if a folder was actually selected
        save_last_directory(folder_path)

    root.destroy()  # Close the root window
    return folder_path
