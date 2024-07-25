
## Installation

To set up the project, follow these steps:

### Prerequisites

1. **CUDA 12.1 Installation**

   Install CUDA 12.1. It is essential for Torch installation. If you already have a higher version of CUDA installed, there's no need to worry; the lower version installations won't affect them. You can download and install CUDA 12.1 from the [NVIDIA CUDA Toolkit website](https://developer.nvidia.com/cuda-12-1-0-download-archive).


2. **Python Installation**

   Ensure you have Python 3.10 or above installed on your system. Python 3.10.10 is recommended for compatibility. You can download it from the [official Python website](https://www.python.org/downloads/release/python-31010/).



### Installing Project Dependencies

1. **Clone the Repository**

   Clone the project repository:

   ```bash
   git clone https://github.com/leakey-lab/Tool-For-Root-Image-Cleaning.git
   cd Tool-For-Root-Image-Cleaning
   ```

2. **Create a Virtual Environment**

   Create a virtual environment named `venv` to manage project dependencies:

   ```bash
   python -m venv venv
   ```


3. **Install Required Packages**

   Install the necessary dependencies listed in the `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

4. **Install PyTorch**

   After CUDA installation, set up PyTorch for the project using the following command:

   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

### Running Instructions

1. **Activate the Virtual Environment**

   Activate the virtual environment:

   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

   - On macOS and Linux:
     ```bash
     source venv/bin/activate
     ```

2. **Run the App Server**

    ```python .\app.py```

3. **Opening the App**

    Click on the URL displayed on the terminal, It will take you to the Dashboard. 

