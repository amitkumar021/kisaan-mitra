cat > README.md << 'EOF'
# 🌿 किसान मित्र (Kisaan Mitra)
### AI Based Plant Disease Detection System

A final year project that uses deep learning to detect plant diseases from leaf images.

## Features
- Upload any plant leaf image
- AI instantly detects the disease
- Shows confidence percentage
- Gives treatment recommendations
- Supports 38 plant disease classes

## Tech Stack
- **AI Model:** MobileNetV2 (Transfer Learning)
- **Dataset:** PlantVillage (70,295 images, 38 classes)
- **Backend:** Python, Flask
- **Frontend:** HTML, CSS, JavaScript
- **Accuracy:** 97.65%

## Project Structure
plant_disease_project/
├── model/
│   ├── train_model.py      # Model training script
│   └── class_names.json    # Disease class names
├── backend/
│   ├── app.py              # Flask API
│   └── requirements.txt
└── frontend/
├── index.html
├── style.css
└── script.js
## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/kisaan-mitra.git
cd kisaan-mitra
```

### 2. Create virtual environment
```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r backend/requirements.txt
```

### 4. Download trained model
Download `plant_model.h5` and place it in the `model/` folder.

### 5. Run Flask backend
```bash
cd backend
python3 app.py
```

### 6. Run frontend
```bash
cd frontend
python3 -m http.server 8080
```

### 7. Open in browser
Go to http://localhost:8080

## Model Details
| Detail | Value |
|--------|-------|
| Architecture | MobileNetV2 |
| Dataset | PlantVillage |
| Training Images | 70,295 |
| Disease Classes | 38 |
| Validation Accuracy | 97.65% |
| Validation Loss | 0.0738 |

## Developer
Made with ❤️ for Final Year Project
EOF