# 🌾 Crop Recommendation System

This project predicts the most suitable crop to grow based on soil and environmental conditions such as Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, and Rainfall.

---

## 🚀 Features
- Machine Learning model using **Random Forest Classifier**
- Dataset preprocessing with **SMOTE** for balancing classes
- Interactive **Flask web interface**
- Real-time prediction from user inputs
- Clean and modern **UI (HTML + CSS)**

---

## 🧠 Tech Stack
**Frontend:** HTML, CSS, JavaScript  
**Backend:** Flask (Python)  
**ML Model:** Scikit-learn  
**Database:** SQLite/MySQL (optional)  
**Version Control:** Git & GitHub  
**Deployment:** Render / PythonAnywhere / AWS

---

## 📂 Folder Structure
CropRecommendation/
│
├── app.py # Flask web app
├── model.py # Machine learning model training script
├── crop_model.pkl # Saved trained model
│
├── data/
│ └── Crop_recommendation.csv
│
├── templates/
│ └── index.html
│
├── static/
│ └── style.css
│
└── README.md
## ⚙️ How to Run Locally

### **Step 1:** Clone the repository
```bash
git clone https://github.com/bhairava009/CropRecommendation.git
cd CropRecommendation
Step 2: Create virtual environment
bash
Copy code
python -m venv venv
venv\Scripts\activate
Step 3: Install dependencies
bash
Copy code
pip install -r requirements.txt
Step 4: Train the ML model
bash
Copy code
python model.py
Step 5: Run the Flask web app
bash
Copy code
python app.py
Then open:
👉 http://127.0.0.1:5000/

🧾 Dataset
The dataset used is Crop_recommendation.csv, containing soil nutrients, environmental parameters, and crop labels.

📊 Accuracy
With RandomForestClassifier and SMOTE balancing, the model achieves ~96–98% accuracy.

👨‍💻 Author
Sonu Kumar
supriya kumari
gaurav raj
vishal kumar
ayush pandey
anuj yadav
B.Tech CSE (Minor in Data Science)
Lovely Professional University
🌐 GitHub: bhairava009

⭐ If you like this project, give it a star on GitHub!

yaml
Copy code

---

### **3️⃣ Save the file.**

---

### **4️⃣ Add and commit it to GitHub**
Run these commands:

```bash
git add README.md
git commit -m "Added README file with full project details"
git push
