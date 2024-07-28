import tkinter as tk
from tkinter import filedialog
import os


def select_folder():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes("-topmost", True)  # Bring the dialog above all windows
    folder_path = filedialog.askdirectory()  # Open the directory selection dialog
    root.destroy()  # Close the root window
    return os.path.abspath(folder_path)
