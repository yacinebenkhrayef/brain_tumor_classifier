# 🧠 Brain Tumor Classifier

## 📋 Project Overview
This project is a deep learning solution for detecting brain tumors from MRI scans. It utilizes **Transfer Learning** (ResNet50) to classify images into four categories: **Glioma, Meningioma, Pituitary, and No Tumor**.

Beyond classification, this project focuses on **Explainable AI (XAI)** using **Grad-CAM** to visualize the regions of the brain driving the model's predictions, addressing ethical transparency requirements in medical AI.

The entire pipeline is deployed using a robust **MLOps** architecture involving Docker, FastAPI, Streamlit, and MLflow.

---

## 🏗️ Architecture

### Software Stack
* **Deep Learning:** TensorFlow/Keras (ResNet50)
* **Backend API:** FastAPI (Python)
* **Frontend UI:** Streamlit
* **MLOps/Tracking:** MLflow (with Postgres & MinIO)
* **Containerization:** Docker & Docker Compose

### Project Structure 
```text
Brain_Tumor_Project/
├── api/
│   ├── __init__.py
│   ├── main.py             
│   ├── utils.py           
│   └── explainability.py  
├── docker/
│   ├── Dockerfile.api      
│   └── Dockerfile.ui      
├── models/
│   └── best_model_finetuned.keras  
├── streamlit_app/
│   └── app.py              
├── tests/
│   ├── __init__.py
│   └── test_api.py         
├── docker-compose.yml      
├── requirements.txt        
└── README.md               
```
## 🚀 Getting Started

### Prerequisites
* Docker Desktop installed and running.
* Git installed.

### Installation & Execution

Clone the repository:

```
git clone <your-repo-url>
cd Brain_Tumor_Project
```

Download the Model:  
Place your trained `best_model_finetuned.keras` file inside the `models/` folder.

Launch the Application:  
Run the following command to build and start the container network:
```
docker-compose up --build
```

### Access Points

Once the containers are running, access the services via your browser:

| Service            | URL                     | Description                                         |
|--------------------|-------------------------|-----------------------------------------------------|
| Frontend (Streamlit)| http://localhost:8501   | Upload MRI scans and view predictions + Heatmaps.  |
| API Documentation  | http://localhost:8000/docs | Swagger UI to test raw API endpoints.               |
| MLflow UI          | http://localhost:5000   | Track experiments, metrics, and logs.               |
| MinIO Console      | http://localhost:9001   | Object storage (S3) browser (User: minio_user, Pass: minio_password).|

---

## 🔬 Methodology

1. **Dataset**  
Source: Brain Tumor MRI Dataset  
Preprocessing: Images resized to 224x224 and pixel values rescaled to [0, 1]. No data augmentation during validation/testing to ensure consistent metrics.

2. **Model Architecture: ResNet50**  
We employed Transfer Learning using the ResNet50 architecture pre-trained on ImageNet.  
- Phase 1 (Feature Extraction): Frozen backbone, trained only the custom top layers (Dense 512 -> Dropout -> Softmax).  
- Phase 2 (Fine-Tuning): Unfroze the top 30 layers of the backbone (approx. last 2 blocks) and retrained with a very low learning rate (1e-5) to adapt feature maps to MRI textures without catastrophic forgetting.

3. **Explainability (Grad-CAM)**  
To ensure the model is not "cheating" (e.g., looking at watermarks or bones), we implemented Gradient-weighted Class Activation Mapping (Grad-CAM). This generates a heatmap overlay showing which pixels most influenced the prediction.

---

## 📊 Results & Performance

(Update this section with your specific results from the notebook)  
Test Accuracy: ~99% (Example)  
Loss: Categorical Crossentropy  
Confusion Matrix: See `mlruns/` or notebook outputs.

---

## 🛠️ Development (Local)

To run tests locally without Docker:

```
pip install -r requirements.txt
pytest tests/
```

---

## 👥 Authors

Student Name 1  
Student Name 2 (Optional)  

Project realized for the Deep Learning Module - Terminale Data Science

---


