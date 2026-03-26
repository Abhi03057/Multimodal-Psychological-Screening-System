# 🧠 Multimodal Psychological Screening System

## 📌 Overview

This project is an AI-based system designed to perform **psychological screening** by combining multiple data sources:

* 🎤 Audio (speech emotion + behavioral analysis)
* 📋 Questionnaire (PHQ-9, GAD-7 scoring)
* 🎥 Facial emotion detection (planned)

The system generates a **structured psychological report** based on combined inputs.

---

## 🚨 Problem Statement

Traditional psychological assessments:

* rely on manual evaluation
* are subjective and inconsistent
* are not scalable

Existing AI systems:

* focus on only one modality (audio or facial)
* lack integration of behavioral signals

---

## 🎯 Objective

To build a **multimodal AI system** that:

* analyzes speech patterns
* evaluates questionnaire responses
* integrates results
* generates a meaningful psychological report

---

## 🧠 System Architecture

```
User Input →
    Audio
    Questionnaire
    (Future: Video)

→ AI Models Process Inputs

→ Results Combined

→ Report Generated
```

---

## 🔧 Tech Stack

### Backend

* Python
* FastAPI
* TensorFlow / Keras
* Librosa

### Frontend

* Next.js
* React

---

## 📂 Project Structure

```
Minor/
├── backend/
│   ├── routes/
│   ├── services/
│   ├── models/
│
├── frontend/
│
├── README.md
```

---

## 🎤 Audio Model

### Features Used:

* Mel Spectrogram (CNN input)
* MFCC
* RMS Energy
* Zero Crossing Rate

### Model:

* Hybrid architecture (CNN + Dense)
* Outputs emotion + behavioral indicators

---

## 📋 Questionnaire Module

* PHQ-9 (Depression scoring)
* GAD-7 (Anxiety scoring)

---

## 🚀 How to Run

### 1. Clone Repository

```
git clone https://github.com/Abhi03057/Multimodal-Psychological-Screening-System
cd Minor
```

---

### 2. Backend Setup

```
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

---

### 3. Frontend Setup

```
cd frontend
npm install
npm run dev
```

---

## ⚠️ Notes

* Datasets are not included (due to size)
* Model file may need to be added manually if not present

---

## 🚧 Current Status

### ✅ Completed

* Audio model training
* Questionnaire system
* Backend API

### 🚧 In Progress

* Model integration
* Facial emotion detection
* Full pipeline connection

---

## 🔥 Key Strength

This project focuses on:

* multimodal integration
* behavioral analysis
* real-world applicability

---

## ⚠️ Limitations

* Dataset is acted (not real patients)
* Not a clinical diagnostic tool
* Moderate model accuracy

---

## 📌 Future Work

* Real-world dataset collection
* Improved model accuracy
* Real-time monitoring system


