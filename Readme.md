# Tool for Root Image Cleaning

## Description

This tool is designed for cleaning and analyzing root images. It provides a dashboard interface for various image processing tasks, including blur detection, duplicate identification, and missing tube analysis.
# Tool for Root Image Cleaning

## Description

This tool is designed for cleaning and analyzing root images. It provides a dashboard interface for various image processing tasks, including blur detection, duplicate identification, and missing tube analysis.

## Installation

### Prerequisites

1. **CUDA 12.1**
   - Required for Torch installation.
   - Download and install from the [NVIDIA CUDA Toolkit website](https://developer.nvidia.com/cuda-12-1-0-download-archive).
   - Note: If you have a higher CUDA version, you don't need to downgrade.

2. **Python 3.10+**
   - Python 3.10.10 is recommended for optimal compatibility.
   - Download from the [official Python website](https://www.python.org/downloads/release/python-31010/).
   - If you are on linux please install python with tkinter support using ```sudo apt-get install python3.10-tk ```


### Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/leakey-lab/Tool-For-Root-Image-Cleaning.git
   cd Tool-For-Root-Image-Cleaning
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment**
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Linux/macOS:
     ```bash
     source venv/bin/activate
     ```

4. **Install Required Packages**
3. **Activate the Virtual Environment**
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Linux/macOS:
     ```bash
     source venv/bin/activate
     ```

4. **Install Required Packages**
   ```bash
   pip install -r requirements.txt
   ```

5. **Install PyTorch**
   
   PyTorch installation varies depending on your operating system and CUDA version. To get the correct installation command:

   - Visit the [PyTorch Get Started page](https://pytorch.org/get-started/locally/)
   - Select your preferences (PyTorch version, operating system, package manager, CUDA version)
   - Use the generated command to install PyTorch

   -For Windows Use This:
5. **Install PyTorch**
   
   PyTorch installation varies depending on your operating system and CUDA version. To get the correct installation command:

   - Visit the [PyTorch Get Started page](https://pytorch.org/get-started/locally/)
   - Select your preferences (PyTorch version, operating system, package manager, CUDA version)
   - Use the generated command to install PyTorch

   -For Windows Use This:

   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

   -For Linux Use This (This command can change, so please check the above link.):
   ```bash
   pip3 install torch torchvision torchaudio
   ```
   But make sure to use the command provided by the PyTorch website for your specific setup.

## Usage

1. **Run the Application**

   Activate your Virtual environemnt first.
   Activate your Virtual environemnt first.

   - Windows:
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Linux/macOS:
   - Linux/macOS:
     ```bash
     source venv/bin/activate
     ```

   -Run the App:
   
   ```bash
   python app.py
   ```

2. **Access the Dashboard**
   - Open the URL displayed in the terminal in your web browser.
   - This will take you to the Root Image Analysis Dashboard.

## Features

- Image Quantity Graph generation
- Missing tube analysis
- Blur detection and analysis
- Duplicate image identification

## Note

This application is not compatible with Apple Silicon devices due to CUDA requirements.
   -Run the App:
   
   ```bash
   python app.py
   ```

2. **Access the Dashboard**
   - Open the URL displayed in the terminal in your web browser.
   - This will take you to the Root Image Analysis Dashboard.

## Features

- Image Quantity Graph generation
- Missing tube analysis
- Blur detection and analysis
- Duplicate image identification

## Note

This application is not compatible with Apple Silicon devices due to CUDA requirements.
