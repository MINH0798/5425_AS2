# Pneumonia Chest X-ray Retrieval System

This project is a deep learning-based medical image retrieval system for chest X-rays. It allows users to upload an image and retrieve the most visually similar cases from a preprocessed training set. The system combines classification (via KNN) and image retrieval (via FAISS), and supports result explanation through Grad-CAM heatmaps.

## 🛠️ Features

- Hybrid feature extractor using EfficientNet-B3 + ViT
- Cosine similarity retrieval using FAISS
- KNN classification with majority voting
- Streamlit-based UI for interaction
- Grad-CAM heatmaps for visual explanation
- Evaluation metrics: KNN accuracy, Top-1 accuracy, Precision@5

## 📂 Dataset

To test the system or rerun feature extraction, you can download the processed dataset (including balanced train/test folders) from the following Google Drive link:

🔗 [Download Dataset from Google Drive](https://drive.google.com/drive/folders/1eFSVz13dNekJwYDYUT0QX1ECfHe1AiJQ?usp=sharing)

## 🧪 Example Output

![interface](./retrieval_vis/1.jpeg)

## 📁 Project Structure

```bash
├── app_streamlit_gam_final.py      # Main Streamlit UI
├── balance_train.py                # Oversample NORMAL class
├── main.py                         # Full pipeline (extract + build index)
├── model.py                        # Hybrid feature extractor (EffNet + ViT)
├── retrieval.py                    # FAISS/KNN + retrieval functions
├── gradcam.py                      # Grad-CAM visualization
├── evaluate.py                     # Batch evaluation metrics
├── batch_test.py                   # Top-K CSV export
├── top.PY                          # t-SNE and distance bar charts
├── utils.py                        # Preprocessing functions
├── test_gradcam_visual.py         # Grad-CAM testing script
├── retrieval_result.txt           # Retrieval logs
├── environment.yml                 # Conda environment dependencies
├── evaluation_report/             # Saved metrics and charts
└── retrieval_vis/                 # Feature files, Grad-CAMs, result images
```

## ⚙️ Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourname/chest-xray-retrieval.git
cd chest-xray-retrieval
```

2. Create the environment:
```bash
conda env create -f environment.yml
conda activate cs20
```

3. (Optional) Run feature extraction if `retrieval_vis/` is empty:
```bash
python main.py
```

4. Launch the app:
```bash
streamlit run app_streamlit_gam_final.py
```

## 📊 Evaluation Results

| Metric                  | Value    |
|-------------------------|----------|
| KNN Accuracy            | 88.30%   |
| Top-1 Retrieval Accuracy| 82.69%   |
| Precision@5             | 86.09%   |

## 📌 Future Improvements

- Add more diverse training data
- Improve Grad-CAM interpretability
- Support multimodal inputs (e.g., clinical data)

## 🧑‍💻 Contributors

Min Huang

---

*This project was developed for COMP5425 - Multimedia Retrieval at the University of Sydney.*
