# Plant Leaf Disease Detection with PyTorch
Open the notebook here: https://nbviewer.org/github/the-noble-analyst/Plant-Leaf-Disease-Detection-with-PyTorch/blob/main/plant_disease_detector.ipynb

## Description

This project delivers an end-to-end computer vision pipeline for detecting plant leaf diseases from images using deep learning. It leverages transfer learning with ResNet18 to classify five plant conditions with high accuracy and minimal training cost. The system includes data preparation, augmentation, model training, error-driven fine-tuning, evaluation, and a production-ready Streamlit web application for real-time inference.

The final model achieves approximately 97–98% test accuracy, with targeted improvements that reduce misclassification between visually similar tomato diseases.

---

## Features

* End-to-end ML pipeline from raw images to deployment
* CNN built with ResNet18 and transfer learning
* Five-class plant disease classifier
* Data augmentation to reduce overfitting
* Clean train/validation/test separation
* Confusion-matrix-driven error analysis
* Targeted fine-tuning of the last ResNet block
* Measurable improvement in hard classes
* Streamlit web app for live predictions
* Reproducible and beginner-friendly workflow

---

## Tech Stack

* Python 3.10
* PyTorch
* torchvision
* NumPy
* Matplotlib
* scikit-learn
* Jupyter Notebook
* Streamlit
* Git
* Kaggle (dataset source)

---

## Installation

Create and activate an isolated environment:

```bash
conda create -n pytorch-env python=3.10
conda activate pytorch-env
```

Install dependencies:

```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn notebook streamlit pillow
```

Launch Jupyter:

```bash
jupyter notebook
```

---

## Configuration

No external API keys are required.

Expected files:

* Trained model: `best_model_ft.pth`
* Dataset structure:

```text
data/
├── train/
│   ├── tomato_early_blight/
│   ├── tomato_late_blight/
│   ├── tomato_healthy/
│   ├── potato_early_blight/
│   └── pepper_healthy/
├── val/
└── test/
```

All folders under `train`, `val`, and `test` must match class names exactly.

---

## Usage

### Training

Run all cells in the notebook or execute the training script:

```bash
python train.py
```

The pipeline:

* Loads and augments images
* Builds a ResNet18 model with transfer learning
* Trains only the classifier head
* Evaluates on validation data
* Saves the best model by validation accuracy

### Fine-Tuning

* Add targeted images to `data/train/tomato_early_blight` and `data/train/tomato_late_blight`
* Restart the environment
* Unfreeze `layer4`
* Train for 4 epochs with `lr=1e-4`
* Save the improved model as `best_model_ft.pth`

### Inference (Streamlit App)

```bash
streamlit run app.py
```

Upload a leaf image to receive:

* Predicted class
* Confidence score
* Visual preview

---

## Project Structure

```text
Plant_Desease_Detector/
├── assets/                    # Screenshots for README
├── data/
│   ├── train/
│   ├── val/
│   └── test/
├── new_data/
│   ├── tomato_early_blight/
│   └── tomato_late_blight/
├── app.py                     # Streamlit app
├── best_model_ft.pth          # Final trained model
├── plant_disease_detector.ipynb
├── requirements.txt
├── runtime.txt
├── README.md
├── LICENSE

```

---

## Screenshots / Demo

### Training Curves
![Training Curves](assets/Accuracy_vs_epoch_1.png)
![Training Curves](assets/Train_loss_vs_epoch_1.png)

### Confusion Matrix Before Fine tuning
![Confusion Matrix](assets/confusion_matrix_1.png)

### Confusion Matrix After Fine tuning
![Confusion Matrix](assets/confusion_matrix_2_after_fine_tuning.png)

### Streamlit App
![Streamlit App](assets/streamlit_plant_app.png)


---

## Roadmap

* Expand dataset for tomato disease classes
* Add Grad-CAM visualization for explainability
* Convert model to TorchScript for faster inference
* Deploy Streamlit app to cloud
* Add batch prediction support
* Introduce class weighting for larger imbalances

---

## Contributing

Contributions are welcome.

Guidelines:

1. Fork the repository
2. Create a feature branch
3. Follow existing coding style
4. Add clear commit messages
5. Submit a pull request with a concise description

Focus areas:

* Data handling improvements
* Model optimization
* UI enhancements
* Documentation clarity

---

## License

This project is released under the MIT License.

---

## Author

Nabeel Siddiqui
Data Science and Machine Learning Practitioner

---

## Acknowledgements

* PlantVillage Dataset (Kaggle)
* PyTorch and torchvision teams
* Open-source contributors in the ML community
