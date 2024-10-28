### **StyleTransfer with VGG19**

**A Python project demonstrating style transfer using the VGG19 model.**

**What is style transfer?**

Neural style transfer is a technique that allows you to take two images - a content image and a style image - and create a new image that combines the content of the first with the style of the second.


**Prerequisites:**
* Python 3.x
* TensorFlow
* Pillow (PIL)
* NumPy
* argparse


**Installation:**
1. Clone this repository:
   ```bash
   git clone https://github.com/Likgiltig/Style-Transfer-VGG19.git
   ```
2. Install the required dependencies:
   ```bash
   pip install tensorflow pillow numpy argparse
   ```
   
**Basic Usage:**
   ```bash
    python style_transfer.py --content path/to/content.jpg --style path/to/style.jpg
   ```

This will run with default settings:
    - 10 epochs
    - 512px maximum image dimension
    - Output saved to 'output' directory
    - Intermediate results saved every epoch
    - Final image named 'final_stylized.png'

**Advanced usage examples:**

High-quality output with more epochs and larger size:
   ```bash
    python style_transfer.py \
        --content photos/portrait.jpg \
        --style art/vangogh.jpg \
        --epochs 20 \
        --image-size 1024 \
        --output-dir vangogh_style \
        --final-name vangogh_portrait.png
   ```

Quick draft with fewer epochs and smaller size:
   ```bash
    python style_transfer.py \
        --content photos/landscape.jpg \
        --style art/monet.jpg \
        --epochs 5 \
        --image-size 256 \
        --save-freq 1
   ```

**Recommendation:**

Using a virtual environment is highly recommended to isolate project dependencies and avoid conflicts with other Python projects. This ensures a cleaner and more predictable development environment.
