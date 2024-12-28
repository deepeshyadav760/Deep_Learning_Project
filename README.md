# Sentiment Analysis on Ear Wearable Product Reviews

## Overview

This project focuses on conducting sentiment analysis on reviews of ear wearable products sourced from Amazon. The goal is to classify user sentiments into **positive (1)** and **negative (0)** categories using **deep learning techniques**. By leveraging an **LSTM model**, the system offers insights into user experiences and preferences while maintaining a user-friendly interface for interaction.

---

## Features
- **Web Scraping:** Python scraper using Selenium to collect 21,825 product reviews from Amazon.
- **Data Preprocessing:** Includes text cleaning, stopword removal, and lemmatization to prepare data.
- **Deep Learning Model:** LSTM-based architecture optimized for sequential text data.
- **Performance Metrics:** Evaluation using accuracy, precision, recall, and F1-score.
- **Interactive UI:** Built with Flask, featuring a clean interface for sentiment prediction.

---

## Key Components

### 1. **Data Collection & Preprocessing**
- **Data Source:** Scraped Amazon reviews of ear wearables (Boat, JBL, Apple, etc.).
- **Data Cleaning:** Removed punctuation, emojis, and stopwords; applied lemmatization.
- **Labeling:** Converted review ratings into binary sentiment labels.
- **Split:** Data split into 80% training and 20% testing sets.

### 2. **Model Architecture**
- **Embedding Layer:** Converts word indices into dense vector representations.
- **LSTM Layer:** Captures temporal dependencies in text sequences.
- **Dropout Layer:** Reduces overfitting.
- **Dense Layer:** Outputs binary classification using a sigmoid activation function.
- **Optimization:** Binary cross-entropy loss function with the Adam optimizer.

### 3. **Performance**
- Achieved robust classification results with balanced accuracy and interpretability.
- Metrics like precision, recall, and F1-score validate model efficacy.

### 4. **User Interface**
- **Frontend:** HTML, CSS, and JavaScript for a responsive experience.
- **Backend:** Flask handles the model and user inputs.
- **Features:**
  - Input field for user reviews.
  - Sentiment prediction with confidence scores.

---

## Project Setup

### Requirements
- Python 3.x
- Libraries: TensorFlow, Flask, Selenium, NumPy, Pandas, Scikit-learn, and Matplotlib.

### Steps to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/deepeshyadav760/Deep_Learning_Project.git
