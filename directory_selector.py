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


# from PyQt5.QtWidgets import QFileDialog, QApplication, QWidget
# from PyQt5.QtCore import Qt


# def select_folder():
#     """Opens a file dialog to select a folder and returns the selected path.

#     Returns:
#         str: The path to the selected folder, or an empty string if the user cancels.
#     """

#     app = QApplication([])
#     # Create a top-level window with Qt::Popup flag
#     top_window = QWidget()
#     top_window.setWindowFlags(Qt.Popup | Qt.WindowStaysOnTopHint)  # Combine flags
#     top_window.show()

#     folder_path = QFileDialog.getExistingDirectory(top_window, "Select Folder")
#     top_window.close()
#     app.exit()
#     return folder_path
