# 🧠 Gender Detection using Deep Learning (InceptionV3 + Transfer Learning)

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?logo=keras)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-green?logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow)

A deep learning project that detects gender (**Male / Female**) from face images using a fine-tuned **InceptionV3** model pre-trained on ImageNet. Trained on the **CelebA** dataset with over **12,000 celebrity face images**, achieving a peak validation accuracy of **~95.25%** over 30 epochs.

---

## 📁 Project Structure

```
Gender_Detection/
│
├── Gender_detection.ipynb     # Main training notebook (data loading → model → evaluation)
├── Main.py                    # Inference script for single image prediction
├── training_.csv              # Training logs (epoch, accuracy, loss, val_accuracy, val_loss)
├── model.h5                   # Saved trained Keras model (generated after training)
├── dataset.txt                # Link to the CelebA dataset on Kaggle
│
└── img_align_celeba/
    └── img_align_celeba/      # CelebA aligned face images (downloaded separately)
```

---

## 📊 Dataset

- **Source:** [CelebA — Large-scale CelebFaces Attributes Dataset](https://www.kaggle.com/jessicali9530/celeba-dataset)
- **Images used:** 12,000 aligned celebrity face images
- **Train / Validation split:** 10,000 train | 2,000 validation
- **Label column:** `Male` from `list_attr_celeba.csv`
  - Original values: `+1` (Male), `-1` (Female) → Remapped to `1` / `0`
- **Class distribution:** Imbalanced (more Female samples than Male in full CelebA)

---

## 🏗️ Model Architecture

Built using **Transfer Learning** on top of **InceptionV3**:

```
InceptionV3 (ImageNet weights, input: 150×150×3, top excluded)
    ↓
GlobalAveragePooling2D
    ↓
Dense(512, activation='relu')
    ↓
Dropout(0.4)
    ↓
Dense(256, activation='relu')
    ↓
Dense(2, activation='softmax')   ← [Female, Male]
```

- **Frozen layers:** First 52 layers of InceptionV3 are frozen (not trainable)
- **Optimizer:** Adam
- **Loss:** Categorical Crossentropy
- **Output:** One-hot encoded — `[1, 0]` = Female, `[0, 1]` = Male

---

## 📈 Training Results

Model trained for **30 epochs** with a `CSVLogger` callback logging all metrics:

| Metric              | Value    |
|---------------------|----------|
| Best Val Accuracy   | **95.25%** |
| Final Val Accuracy  | **95.05%** |
| Best Train Accuracy | **99.67%** |
| Total Epochs        | 30       |

Training and validation accuracy/loss plots are generated inside the notebook.

---

## ⚙️ Requirements

Install the required dependencies:

```bash
pip install tensorflow opencv-python numpy pandas matplotlib tqdm
```

| Library        | Purpose                            |
|----------------|------------------------------------|
| `tensorflow`   | Model building, training, loading  |
| `keras`        | High-level neural network API      |
| `opencv-python`| Image reading and resizing         |
| `numpy`        | Array manipulation                 |
| `pandas`       | CSV reading (attributes + logs)    |
| `matplotlib`   | Accuracy/loss visualization        |
| `tqdm`         | Progress bar during data loading   |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/gender-detection.git
cd gender-detection
```

### 2. Download the Dataset
Download the CelebA dataset from Kaggle:
```
https://www.kaggle.com/jessicali9530/celeba-dataset
```
Place the `img_align_celeba/` folder and `list_attr_celeba.csv` in the project root.

### 3. Train the Model
Open and run the Jupyter notebook:
```bash
jupyter notebook Gender_detection.ipynb
```
This will:
- Load and preprocess 12,000 face images (resized to 150×150)
- Build the InceptionV3 transfer learning model
- Train for 30 epochs and log metrics to `training.csv`
- Save the trained model as `model.h5`

### 4. Run Inference on a New Image
```bash
python Main.py
```

Update the image path inside `Main.py` before running:
```python
image = cv2.imread('img_align_celeba/img_align_celeba/000007.jpg')
```

**Sample output:**
```
[0.03 0.97]
Predicted class is : Male
```

---

## 🔍 How Prediction Works (`Main.py`)

```python
# 1. Read and resize image to 150×150
image = cv2.imread('path/to/image.jpg')
image = cv2.resize(image, (150, 150))
image = np.array(image).reshape(1, 150, 150, 3)

# 2. Load the saved model
model = tf.keras.models.load_model('model.h5')

# 3. Predict
out_arr = model.predict(image)[0]   # e.g. [0.03, 0.97]

# 4. Decode: argmax(0) → Female, argmax(1) → Male
dict = {1: 'Male', 0: 'Female'}
print(f'Predicted class is : {dict[np.argmax(out_arr)]}')
```

---

## 📉 Training Curves

The notebook generates plots for:
- **Training vs. Validation Accuracy** across 30 epochs
- **Training vs. Validation Loss** across 30 epochs

These visualizations help identify overfitting (training accuracy ~99.67% vs val ~95.05% suggests mild overfitting at later epochs).

---

## 🧩 Key Design Decisions

| Decision | Reason |
|----------|--------|
| InceptionV3 over custom CNN | Faster convergence, better feature extraction with limited data |
| Input size 150×150 (not 299×299) | Memory efficiency; InceptionV3 supports flexible input shapes when `include_top=False` |
| Freeze first 52 layers | Preserve low-level ImageNet features; only fine-tune higher-level layers |
| Dropout(0.4) | Regularization to reduce overfitting |
| One-hot encoding via `np_utils.to_categorical` | Required for `categorical_crossentropy` loss |
| `CSVLogger` callback | Persistent training history for post-training analysis |

---

## ⚠️ Known Limitations

- Model is trained on **celebrity faces** (CelebA) — performance may degrade on non-celebrity or diverse demographic inputs.
- Mild **overfitting** observed after epoch 20 (train acc ~99.67% vs val ~95.05%).
- Input images must be **face-cropped and aligned** for best results, similar to the CelebA preprocessing.
- Binary gender classification only — does not account for non-binary gender identities.

---

## 🔮 Future Improvements

- [ ] Add real-time webcam inference using OpenCV
- [ ] Apply data augmentation to reduce overfitting
- [ ] Use a larger training subset (full CelebA has 202,599 images)
- [ ] Add early stopping callback to prevent overfitting
- [ ] Export model to TensorFlow Lite for mobile deployment
- [ ] Build a simple Flask/FastAPI web interface for demo

---

## 📜 License

This project is licensed under the **MIT License** — free to use, modify, and distribute.

---

## 🙏 Acknowledgements

- **Dataset:** [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) by MMLab, CUHK
- **Model:** [InceptionV3](https://keras.io/api/applications/inceptionv3/) — "Rethinking the Inception Architecture for Computer Vision" (Szegedy et al., 2015)
- **Kaggle Host:** [Jessica Li](https://www.kaggle.com/jessicali9530)
