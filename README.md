
# Retinal Vessel Segmentation with PyTorch U-Net

This project implements a U-Net-based neural network in PyTorch for the segmentation of retinal blood vessels from medical images. It is designed for accuracy and robustness using best deep learning practices and supports automated dataset handling.

## Usage

1. **Clone this repository:**

```bash
git clone https://github.com/saman-emami/retina-unet
cd retina-unet
```

2. **Create virtual environment:**

```bash
python -m venv .venv
source .venv/bin/activate
```

3. **Install requirements:**

```bash
pip install -r requirements.txt
```

4. **Train the model**

    - Run the train.ipynb notebook

5. **Model outputs:**

    * Training/validation metrics are logged to the console.
    * Trained model weights saved as best_model.pth.



## Dataset

- **Source:** [Kaggle: Retina Blood Vessel Dataset](https://www.kaggle.com/datasets/abdallahwagih/retina-blood-vessel)
- The dataset will be automatically handled by the script.
