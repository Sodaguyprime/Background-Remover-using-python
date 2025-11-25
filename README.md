
# 🖼️ AI Background Remover

A powerful, easy-to-use Python script to remove image backgrounds automatically. Built on top of the robust [InSPyReNet](https://github.com/plemeri/transparent-background) library, this tool supports high-quality processing, batch operations, and GPU acceleration.

## ✨ Features

  * **High Precision:** Uses advanced AI (InSPyReNet) to handle complex details like hair and transparent objects.
  * **Two Modes:**
      * `base`: Maximum quality (default).
      * `fast`: Faster processing for quick tasks.
  * **Batch Processing:** Process entire folders of images at once.
  * **Smart Output:** Automatically saves as PNG with transparency.
  * **Hardware Acceleration:** Supports CUDA GPU acceleration (falls back to CPU automatically).
  * **Dependency Check:** Auto-detects missing libraries.

## 🛠️ Prerequisites

  * **Python 3.10 - 3.12** (Recommended).
      * *Note: Python 3.14 (Pre-release) is not currently supported due to missing wheel files.*
  * **Visual C++ Build Tools** (Only if you are compiling dependencies from source).

## 📦 Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Sodaguyprime/Background-Remover-using-python.git
    cd bg-remover
    ```

2.  **Install dependencies:**

    ```bash
    pip install transparent-background pillow numpy
    ```

    *If you have issues with torch, install it directly first:*

    ```bash
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    ```

## 🚀 Usage

Run the script from your terminal using `python`.

### 1\. Process a Single Image

```bash
python bg_remove_script.py -i input.jpg -o output.png
```

### 2\. Batch Process a Folder

Remove backgrounds from all images in a folder:

```bash
python bg_remove_script.py -i ./my_photos -o ./processed_photos -b
```

### 3\. Use Fast Mode

For lower memory usage and faster speeds:

```bash
python bg_remove_script.py -i input.jpg -o output.png -m fast
```

### 4\. Advanced Options

  * **Threshold (`-t`):** Adjust transparency cutoff (0-255). Lower = more transparent pixels kept.
  * **GPU (`--gpu`):** Force usage of CUDA if available.

<!-- end list -->

```bash
python bg_remove_script.py -i input.jpg -o output.png -t 100 --gpu
```

## ⚙️ Troubleshooting

**"Microsoft Visual C++ 14.0 is required" error:**
This usually happens if you use a Python version that is too new (like 3.13 or 3.14).

  * **Fix:** Uninstall Python and install the stable **Python 3.12**.

**First Run Delay:**
On the very first run, the script will download the AI model (\~170MB). This is normal. Please ensure you have an internet connection.

**Permission Denied Errors:**
If you see permission errors regarding `~/.transparent-background`, ensure you are not passing a custom checkpoint path in the code unless you know the specific file path. The script works best using the library's default model path.

## 📄 License

This project uses the `transparent-background` library. Please refer to their repository for underlying model licenses.

-----

**Made with ❤️ by Ammar, aka sodaguy**
