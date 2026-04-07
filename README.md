🚧 Framework for Construction Areas: Safety and Quality Control using Deep Learning and CNN Models
AI-based system for PPE detection, crack detection, and predictive maintenance using YOLOv8, CNN, and LSTM.

## ⚠️ Problem Statement
Construction sites are among the most hazardous work environments, with frequent safety violations, structural defects, and unexpected equipment failures. Traditional monitoring methods rely heavily on manual inspection, which is time-consuming, error-prone, and lacks real-time responsiveness.  
There is a strong need for an intelligent system that can automate safety compliance, detect structural issues early, and predict equipment failures to prevent accidents and reduce operational risks.

## 💡 Proposed Solution
This project presents an AI-based framework that integrates multiple deep learning models to ensure safety and quality control in construction environments. The system combines computer vision and time-series analysis to provide real-time monitoring and predictive insights.

The framework consists of:
- PPE Detection using **YOLOv8**
- Crack Detection using **CNN**
- Predictive Maintenance using **LSTM**

All modules are integrated into a single dashboard for centralized monitoring and decision-making.

## 🎯 Objectives
- Automate detection of safety equipment compliance (helmet, vest)
- Identify structural defects such as cracks in concrete
- Predict equipment failure using sensor data
- Enable real-time monitoring through webcam and image inputs
- Reduce manual inspection and improve safety efficiency

## 🧠 System Architecture
The system follows a layered architecture:
- **Input Layer:** Image upload, webcam, sensor data  
- **Preprocessing Layer:** Resizing, normalization, augmentation  
- **Model Layer:** YOLOv8, CNN, LSTM  
- **Decision Layer:** Alerts, predictions, visualization  

This architecture ensures scalability, modularity, and efficient real-time processing.

## ⚙️ Modules

### 🦺 PPE Detection (YOLOv8)
- Detects helmets and safety vests
- Supports image upload and live webcam
- Identifies safety violations in real time

### 🧱 Crack Detection (CNN)
- Classifies concrete surfaces as crack / no crack
- Provides confidence score and severity
- Helps in early structural defect detection

### ⚙️ Predictive Maintenance (LSTM)
- Uses multivariate time-series sensor data
- Predicts failure probability
- Estimates Remaining Useful Life (RUL)

## 🖥️ Key Features
- Real-time monitoring system  
- Multi-module integration  
- Centralized dashboard using Streamlit  
- Visual outputs (bounding boxes, graphs, alerts)  
- Supports both offline and real-time inputs  

## 🛠️ Tech Stack
- **Programming Language:** Python  
- **Deep Learning Models:** YOLOv8, CNN, LSTM  
- **Libraries:** OpenCV, NumPy, Pandas, Scikit-learn  
- **Frameworks:** TensorFlow / PyTorch  
- **Interface:** Streamlit  

## 📊 Performance Results
| Module | Accuracy |
|--------|--------|
| PPE Detection | 85.3% |
| Crack Detection | 92.5% |
| Predictive Maintenance | 88.7% |

✔ High accuracy across all modules  
✔ Reliable performance in real-time scenarios  

## 📁 Project Structure
AI-Construction-Safety-Monitoring-System/
│
├── dataset/
├── models/
├── src/
├── results/
├── report/
├── requirements.txt
├── README.md

## Overview Architecture
![Uploading Screenshot 2026-04-03 162150.png…]()


## ▶️ How to Run

```bash
git clone https://github.com/your-username/AI-Construction-Safety-Monitoring-System.git
cd AI-Construction-Safety-Monitoring-System
pip install -r requirements.txt
streamlit run app.py

```
## 📸 Sample Outputs

PPE detection with bounding boxes
![Uploading Screenshot 2026-04-04 165614.png…]()
![Uploading Screenshot 2025-12-08 205740.png…]()

Crack detection results
![Uploading Screenshot 2025-12-08 210255.png…]()

Failure risk prediction
![Uploading Screenshot 2025-12-08 210800.png…]()
![Uploading Screenshot 2025-12-08 210818.png…]()


Dashboard visualization
![Uploading Screenshot 2025-12-08 205607.png…]()
![Uploading Screenshot 2025-12-08 205646.png…]()


```
## 🚀 Applications
Construction site safety monitoring
Structural health monitoring
Industrial predictive maintenance
Smart infrastructure systems

```
## 🔮 Future Scope
Deployment on edge devices
Explainable AI integration
Multi-modal data fusion
Large-scale real-time deployment

```
## 👩‍💻 Team Members
Muthyala Radhika
Mannem Narsi Reddy
Munagala Maheswar Reddy
Kamalesh Krishnasamy

```
## 🎓 Guide
Dr. R. Sumathi
Department of Computer Science and Engineering

```
##⭐ Support
If you found this project useful, consider giving it a ⭐ on GitHub!
