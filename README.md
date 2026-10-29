# 🤖 SupportAI — Customer Support Intelligence

> **An AI-powered customer support intelligence platform for automated complaint analysis, issue detection, priority prediction, escalation risk assessment, and intelligent resolution assistance.**

---

## 📌 Project Overview

**SupportAI** is an intelligent customer support system that combines **Machine Learning, Natural Language Processing (NLP), semantic analysis, and AI-powered assistance** to analyze customer complaints and support teams in making faster and more informed decisions.

The platform transforms unstructured customer complaints into actionable insights by identifying the underlying issue, predicting its priority, assessing escalation risk, and providing recommended support actions.

The system also includes an **AI Support Assistant** designed to interact with users and provide contextual assistance.

---

## 🎯 Objectives

The primary objectives of SupportAI are to:

* Automate customer complaint analysis
* Identify the underlying issue from natural-language complaints
* Predict complaint priority
* Detect potentially critical or high-risk cases
* Assess escalation risk
* Recommend appropriate resolutions
* Provide an AI-powered customer support assistant
* Reduce manual effort in customer support operations
* Improve response time and support decision-making

---

## 🚀 Key Features

### 📊 Complaint Analysis

Analyzes customer complaints and extracts meaningful information from unstructured text.

**Capabilities:**

* Complaint classification
* Issue identification
* Priority analysis
* Resolution recommendation

---

### ⚡ Semantic Issue Detection

Uses NLP and semantic analysis to understand the meaning of customer complaints rather than relying only on exact keyword matching.

The system can identify relevant issue categories from natural-language descriptions.

---

### 🎯 Priority Prediction

Predicts the priority level of a customer complaint based on its content and characteristics.

Possible priority levels include:

* Low
* Medium
* High
* Critical

This helps support teams focus on the most important complaints first.

---

### 🚨 Escalation Risk Engine

Evaluates the severity and escalation potential of a complaint.

The engine considers factors such as:

* Complaint priority
* Security-related language
* Financial impact
* Service impact
* Urgency
* Legal threats
* Data loss
* Unauthorized access

The resulting risk level helps determine whether immediate human intervention is required.

---

### 🧠 ML Intelligence

The Machine Learning module provides automated predictions from customer complaint text.

The project uses **TF-IDF-based text representation** and machine learning models for classification tasks.

---

### 💬 AI Support Assistant

SupportAI includes an AI-powered conversational assistant that can:

* Understand customer queries
* Respond to support-related questions
* Provide contextual assistance
* Suggest appropriate actions
* Handle general customer-support interactions

---

## 🏗️ System Architecture

```text
                    Customer Complaint
                           │
                           ▼
                  ┌─────────────────┐
                  │   Text Input    │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ NLP Preprocessing│
                  └────────┬────────┘
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
       Issue Detection  Priority      Semantic
          Model         Prediction     Analysis
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                  ┌─────────────────┐
                  │ Escalation Risk │
                  │     Engine      │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │   Resolution    │
                  │ Recommendation  │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ SupportAI UI /  │
                  │ AI Assistant    │
                  └─────────────────┘
```

---

## 🧠 Machine Learning Approach

### Text Representation

Customer complaints are converted into numerical representations using:

**TF-IDF (Term Frequency–Inverse Document Frequency)**

The training pipeline generated a feature matrix of:

```text
Training data : 6775 × 5000
Testing data  : 1694 × 5000
```

The dataset was split using a **stratified train-test split** to maintain class distribution.

### Machine Learning Algorithm

The primary classification algorithm used in the project is:

**Logistic Regression**

It is used for classification tasks involving customer support complaint data.

---

## 📊 Model Components

| Component                     | Approach                            |
| ----------------------------- | ----------------------------------- |
| Ticket / Issue Classification | Logistic Regression + TF-IDF        |
| Root Cause Classification     | Logistic Regression + TF-IDF        |
| Priority Prediction           | Logistic Regression + text features |
| Semantic Issue Detection      | Semantic text analysis              |
| Escalation Risk               | Rule-based risk scoring             |
| Resolution Recommendation     | Support intelligence layer          |
| AI Assistant                  | AI-powered conversational interface |

---

## 📈 Model Results

The current experimental results include:

| Prediction Task            | Accuracy |
| -------------------------- | -------: |
| Ticket Type Classification |   19.19% |
| Root Cause Classification  |   98.35% |
| Priority Prediction        |   25.74% |

> **Note:** Model performance varies significantly across classification tasks because of differences in class distribution, text ambiguity, and dataset characteristics. Further dataset balancing, feature engineering, and model optimization can improve the weaker classification components.

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Machine Learning & NLP

* Scikit-learn
* NumPy
* Pandas
* TF-IDF
* Logistic Regression
* Sentence Transformers

### Application Development

* Streamlit

### AI

* AI-powered conversational assistant
* Semantic text analysis

### Model & Data Handling

* Joblib
* Pickle
* CSV datasets

### Development Tools

* Visual Studio Code
* Jupyter Notebook
* Git
* GitHub

---

## 📁 Project Structure

```text
customer-support-intelligence/
│
├── app/
│   └── Streamlit application files
│
├── data/
│   └── Dataset files
│
├── models/
│   └── Trained ML models
│
├── notebooks/
│   └── Data analysis and model development notebooks
│
├── screenshots/
│   ├── Chatbot/
│   ├── Escalation - Engine/
│   ├── Priority - Prediction/
│   ├── Semantic - Issue/
│   ├── dashboard/
│   └── ml-intelligence/
│
├── .gitignore
└── README.md
```

---

# 📸 Application Screenshots

## 🏠 Main Dashboard

The SupportAI dashboard provides access to the major customer-support intelligence modules.

![Main Dashboard](screenshots/dashboard/Main%20-%20Dashboard.png)
)

---

## 📊 Complaint Analysis

The complaint analysis module processes customer complaints and provides automated insights.

![Dashboard 1](screenshots/dashboard/Dashboard%20-%201.png)

![Dashboard 2](https://raw.githubusercontent.com/aswin-tech-01/customer-support-intelligence/main/screenshots/dashboard/Dashboard%20-%202.png)

---

## 💬 AI Support Assistant

The AI Support Assistant provides conversational support and helps users interact with the system naturally.

![AI Support Assistant](<img width="1841" height="917" alt="Chatbot" src="https://github.com/user-attachments/assets/e318a779-07c1-43e8-b0b3-2b1faba7cce3" />
)

---

## 🧠 ML Intelligence

The ML Intelligence module provides machine-learning-driven analysis of customer complaints.

![ML Intelligence 1](<img width="1850" height="907" alt="Machine - Intelligence - 1" src="https://github.com/user-attachments/assets/97490444-a5eb-4a1a-9462-f7e284a6bb2b" />
)

![ML Intelligence 2](<img width="1860" height="923" alt="Machine - Intelligence - 2" src="https://github.com/user-attachments/assets/0740b96a-c361-4cfe-8f66-badb7721035a" />
)

![ML Intelligence 3](<img width="1851" height="921" alt="Machine - Intelligence - 3" src="https://github.com/user-attachments/assets/f82cf0dc-d441-4673-ac2e-bd135cb09988" />
)

---

## ⚡ Semantic Issue Detection

Semantic analysis identifies the underlying issue from the meaning of the customer's complaint.

![Semantic Issue Detection](<img width="1851" height="923" alt="Semantic - Issue - Prediction" src="https://github.com/user-attachments/assets/15636e15-d8ca-4cb2-8b6d-fc5149e52515" />
)

---

## 🎯 Priority Prediction

The priority prediction module identifies the urgency of customer complaints and helps support teams prioritize cases.

![Priority Prediction](<img width="1845" height="911" alt="Priority - prediction" src="https://github.com/user-attachments/assets/7c767fa2-8d79-49d3-83d2-1d5c32425c43" />
)

---

## 🚨 Escalation Risk Engine

The escalation engine evaluates complaint severity and determines whether a case requires immediate human intervention.

![Escalation Risk 1](<img width="1838" height="912" alt="Escalation - Engine - 1" src="https://github.com/user-attachments/assets/ee33f789-3cb5-4c21-b8f0-45db2e2db747" />
)

![Escalation Risk 2](<img width="1853" height="917" alt="Escalation - Engine - 2 " src="https://github.com/user-attachments/assets/e99b8ac4-11d4-4f2a-a27b-48387a3936e1" />
)

---

## ▶️ How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/aswin-tech-01/customer-support-intelligence.git
```

### 2. Navigate to the Project

```bash
cd customer-support-intelligence
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Environment

**Windows:**

```bash
.venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

> If a `requirements.txt` file is not currently included in the repository, add one containing the project's required Python packages before publishing the final version.

### 6. Run the Streamlit Application

```bash
streamlit run app/app.py
```

The application will then open in your browser.

---

## 🔄 Project Workflow

```text
Customer Complaint
        ↓
Text Processing
        ↓
Feature Extraction
        ↓
Issue Detection
        ↓
Priority Prediction
        ↓
Semantic Analysis
        ↓
Escalation Risk Assessment
        ↓
Resolution Recommendation
        ↓
AI Support Assistance
```

---

## 🔮 Future Enhancements

Future versions of SupportAI can include:

* Improved classification models
* Better handling of class imbalance
* Transformer-based NLP models
* Multilingual complaint analysis
* Real-time customer-support analytics
* Customer sentiment analysis
* Automated ticket routing
* Agent performance dashboards
* Conversation history
* Database integration
* Authentication and role-based access
* Cloud deployment
* Real-time monitoring and logging

---

## 👨‍💻 Author

**Aswin P S**

B.Tech Computer Science & Engineering

### Project

**SupportAI — Customer Support Intelligence**

---

## ⭐ Project Highlights

* 🤖 AI-powered customer support
* 🧠 Machine Learning-based complaint analysis
* ⚡ Semantic issue detection
* 🎯 Priority prediction
* 🚨 Escalation risk assessment
* 💬 AI Support Assistant
* 📊 Interactive Streamlit interface
* 🔍 NLP-based text analysis
* 🛠️ End-to-end ML application

---

> **SupportAI transforms unstructured customer complaints into actionable intelligence for faster, smarter, and more effective customer support.**
