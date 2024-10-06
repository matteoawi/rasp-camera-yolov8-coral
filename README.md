# rasp-camera-yolov8-coral
Setup instructions for running YOLOv8 with COCO on Raspberry Pi 4 with Coral USB at 75 FPS

## Running YOLOv8 at 75 FPS on Raspberry Pi 4 with Google Coral Edge TPU

This guide will help you set up YOLOv8 on a Raspberry Pi 4 with Coral USB Accelerator for high-performance object detection at 75 FPS. **Note:** This tutorial works well with USB webcams but may not function perfectly with Pi Cameras.

**Recommended OS:** Raspberry Pi OS Bookworm 64-bit. If you encounter issues, it’s advisable to start with a freshly flashed OS on the SD card.

## Prerequisites

- **Raspberry Pi 4**
- **Google Coral USB Accelerator**
- **YOLOv8 model**
- **USB Webcam** (recommended)

---

Download my repository to access all the files needed for the initial tests!

## Step 1: Install Python 3.9.12

1. **Update and upgrade system packages:**

   ```bash
   sudo apt-get update
   sudo apt-get upgrade
   ```

2. **Create a new folder and navigate into it:**

   ```bash
   mkdir yourfoldername
   cd yourfoldername
   ```

3. **Install pyenv:**

   ```bash
   curl https://pyenv.run | bash
   ```

4. **Set environment variables:**

   ```bash
   echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
   echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
   echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
   exec "$SHELL"
   ```

5. **Install system dependencies:**

   ```bash
   sudo apt-get install --yes libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
   libsqlite3-dev llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev \
   libgdbm-dev lzma lzma-dev tcl-dev libxml2-dev libxmlsec1-dev libffi-dev \
   liblzma-dev wget curl make build-essential openssl
   ```

6. **Install Python 3.9.12:**

   > **Note:** Python 3.9.12 is required as Coral is currently incompatible with newer versions.

   ```bash
   pyenv install 3.9.12
   pyenv local 3.9.12
   ```

8. **Verify Python installation:**

   ```bash
   python --version
   ```

9. **Create and activate a virtual environment:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

---

## Step 2: Set up Google Coral USB Accelerator

1. **Connect the Coral USB Accelerator to the Raspberry Pi USB 3.0.** If it doesn’t work when running the Python script at the end, disconnect and reconnect the device.

2. **Install PyTorch and related libraries (always in .venv yourfoldername):**

   ```bash
   pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
   ```

   > **Note:** This step is crucial to ensure compatibility with Coral Edge TPU after recent Raspberry Pi OS updates.

3. **Install the Edge TPU library:**

   ```bash
   pip install edge-tpu-silva
   ```

4. **Run the Coral setup script:**
   For standard inference performance:

   ```bash
   silvatpu-linux-setup
   ```

   For maximum inference performance (you might need a fan for safety during operation):

   ```bash
   silvatpu-linux-setup --speed max
   ```
   
**Reboot your Raspberry Pi:**

```bash
sudo reboot
```

**Go back in the folder and reactivate the created virtual environment**
```bash
cd yourfoldername
```
```bash
source .venv/bin/activate
```

**Open Thonny from venv command**

```bash
thonny
```

6. **Change Python version in Thonny:**

   - Go to the **Run** menu.
   - Select **Select interpreter...** and choose the Python version from the virtual environment (`venv` with Python 3.9).
  

7. **Open Thonny and install cvzone:**

   - Open the **Thonny IDE**.
   - If you don't see the Run button, switch the view mode by pressing the button on the far right.
   - Go to **Tools > Manage Packages...**
   - Search for **cvzone** and install it.


8. **Downgrade numpy if necessary:**

   In some cases, numpy version 2.0 may cause issues. You can downgrade it:

   - In Thonny, go to **Tools > Manage Packages...**
   - Search for **numpy**.
   - Click on the three dots (options) next to it and select **Choose specific version**.
   - Choose a version below 2.0 and install it.

---

Congratulations! 🚀

Now run the script `YoloV8_onCOCO.py` on Thonny or use:

```bash
python3 YoloV8_onCOCO.py
```

I used the `--speed max` inference setup for 75 FPS with this script, achieving 18 inferences per second.

---

## Converting YOLOv8 Models for Edge TPU with Google Colab

To convert the YOLOv8 models from PyTorch (.pt) to TensorFlow Lite (.tflite) optimized for Edge TPU

1. Open the Google Colab notebook.
2. Run the script to install Ultralytics and convert the model:

   ```python
   !pip install ultralytics
   from ultralytics import YOLO

   # Load the YOLOv8 model
   model = YOLO('yolov8n.pt')

   # Convert the model to TensorFlow Lite optimized for Edge TPU
   model.export(format='tflite', optimize_for='edgetpu')
   ```

3. Download the converted file and use it in your project with Coral TPU.

---

You should now be able to run YOLOv8 with Coral USB Accelerator at 75 FPS on your Raspberry Pi 4.

---

**Note:** If you encounter any issues or have questions, feel free to contribute or open an issue in the repository.
