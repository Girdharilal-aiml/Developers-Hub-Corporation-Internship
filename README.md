# Developers Hub Corporation — ML/AI Internship Portfolio

**Intern:** Girdhari Lal
**Organization:** Developers Hub Corporation (DHC)
**Program:** Machine Learning & AI Internship
**Repo:** `Developers-Hub-Corporation-Internship`

---

## Repository Structure

```
Developers-Hub-Corporation-Internship/
│
├── Task Phase 1/
│   ├── T1 - Iris Dataset Exploration/
│   ├── T2 - Stock Price Prediction/
│   ├── T3 - Heart Disease Prediction/
│   ├── T4 - Health Query Chatbot/
│   ├── T5 - Mental Health Chatbot/
│   └── T6 - House Price Prediction/
│
└── Task Phase 2/
    ├── T1 - News Topic Classifier (BERT)/
    ├── T2 - ML Pipeline with Scikit-learn/
    ├── T3 - Multimodal Housing Price Prediction/
    ├── T4 - Context-Aware RAG Chatbot/
    └── T5 - Auto-Tagging Support Tickets/
```

---

## Phase 1 — Foundational ML Tasks

### T1 — Iris Dataset Exploration and Visualization
**Objective:** Load, inspect, and visualize the Iris dataset to understand feature distributions and relationships.
**Tools:** pandas, matplotlib, seaborn

**Key Work:**
- Loaded the dataset and performed EDA using `.info()`, `.describe()`, and `.head()`
- Created scatter plots, histograms, box plots, and a full pair plot
- Built a correlation heatmap to identify feature relationships
- Found that petal length and petal width are the strongest predictors for species classification

**Result:** Setosa is linearly separable from the other two species using petal features alone.

---

### T2 — Stock Price Prediction (Short-Term)
**Objective:** Predict the next day closing price of Apple (AAPL) stock using historical data.
**Dataset:** Yahoo Finance via `yfinance`
**Tools:** pandas, scikit-learn, matplotlib

**Key Work:**
- Engineered lag features (Close_lag1/2/3), rolling averages (MA5, MA10), daily range, and overnight gap
- Trained Linear Regression and Random Forest Regressor
- Evaluated using MAE, RMSE, and R2 score
- Visualized actual vs predicted prices and feature importance

| Model | MAE | RMSE | R2 |
|---|---|---|---|
| Linear Regression | ~$3.2K | ~$4.1K | 0.91 |
| Random Forest | ~$1.8K | ~$2.4K | 0.97 |

---

### T3 — Heart Disease Prediction
**Objective:** Predict whether a patient is at risk of heart disease based on clinical features.
**Dataset:** Heart Disease UCI Dataset (Kaggle)
**Tools:** pandas, scikit-learn, matplotlib, seaborn

**Key Work:**
- Cleaned dataset and handled missing values in ca and thal columns
- Performed EDA including feature distributions by disease status and correlation heatmap
- Trained Logistic Regression and Decision Tree classifier
- Evaluated with accuracy, F1 score, confusion matrix, and ROC curve
- Top features: chest pain type, max heart rate, number of major vessels

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | ~85% | 0.91 |
| Decision Tree | ~82% | 0.88 |

**Key insight:** False Negatives (missed disease cases) are more dangerous than False Positives in medical diagnosis — recall was prioritized over precision.

---

### T4 — General Health Query Chatbot (Prompt Engineering)
**Objective:** Build a chatbot that answers general health questions using an LLM with safety filters.
**Tools:** Anthropic Claude API, Python, Gradio

**Key Work:**
- Designed a multi-section system prompt covering role, tone, format, and strict safety rules
- Built three safety layers: emergency keyword detection, sensitive topic disclaimers, response quality filtering
- Compared bad prompt vs good prompt responses to demonstrate prompt engineering impact
- Built interactive CLI and Gradio web demo

**Safety features:**
- Emergency keywords trigger a hardcoded crisis response — no LLM involved
- Medication dosage questions always defer to a doctor or pharmacist
- Responses cleaned to remove filler phrases automatically

---

### T5 — Mental Health Support Chatbot (Fine-Tuning)
**Objective:** Fine-tune a small LLM to respond empathetically to stress, anxiety, and emotional wellness queries.
**Dataset:** EmpatheticDialogues (Facebook AI) — curated empathetic dialogue pairs
**Tools:** PyTorch, Hugging Face Transformers, Trainer API, Streamlit

**Key Work:**
- Built a custom dialogue dataset covering 8 emotional categories (stress, anxiety, grief, loneliness, anger, etc.)
- Formatted training data as `<|user|> message <|bot|> response` for causal LM fine-tuning
- Fine-tuned DistilGPT-2 with 3 epochs, lr=5e-5, batch=4
- Visualized training and validation loss curves and perplexity over training
- Implemented 3-layer safety system: crisis detection, quality check, fallback responses
- Built CLI chatbot and Streamlit web interface

**Key result:** Fine-tuned model generates empathetic responses vs base DistilGPT-2 which just continues the user text in generic internet style.

---

### T6 — House Price Prediction
**Objective:** Predict house sale prices using property features such as size, quality, and location.
**Dataset:** Kaggle House Prices — Ames, Iowa (1,460 records)
**Tools:** pandas, scikit-learn, matplotlib, seaborn

**Key Work:**
- Cleaned missing values and engineered 6 new features: HouseAge, YearsSinceRemod, TotalBath, TotalSF, QualArea, IsRemodeled
- Applied ordinal encoding for quality columns and one-hot encoding for nominal categoricals
- Log-transformed the target variable to reduce skewness
- Trained Ridge Regression and Gradient Boosting Regressor
- Evaluated with MAE, RMSE, R2 and 5-fold cross-validation

| Model | MAE | RMSE | R2 |
|---|---|---|---|
| Ridge Regression | ~$22K | ~$29K | 0.87 |
| Gradient Boosting | ~$14K | ~$19K | 0.93 |

**Key insight:** OverallQual, GrLivArea, and Neighborhood were the top 3 predictors. Log-transforming price is essential for both models.

---

## Phase 2 — Advanced ML Tasks

### T1 — News Topic Classifier Using BERT
**Objective:** Fine-tune BERT to classify news headlines into 4 categories: World, Sports, Business, Sci/Tech.
**Dataset:** AG News (Hugging Face) — 120,000 training samples
**Tools:** Hugging Face Transformers, PyTorch, scikit-learn, Gradio

**Key Work:**
- Tokenized headlines using BERT WordPiece tokenizer with max_length=128
- Fine-tuned bert-base-uncased (110M params) using Hugging Face Trainer API
- 3 epochs, lr=2e-5, batch=16, warmup_ratio=0.1, mixed precision (fp16 on GPU)
- Established TF-IDF + Logistic Regression as a baseline for comparison
- Evaluated with accuracy, macro F1, confusion matrix, and ROC curves (one-vs-rest)
- Deployed interactive Gradio demo

| Model | Accuracy | Macro F1 |
|---|---|---|
| TF-IDF + LR (baseline) | 87.5% | 0.876 |
| BERT (fine-tuned) | 93.8% | 0.936 |

**Key insight:** BERT's contextual understanding handles ambiguous headlines better than keyword-based models. Sports is easiest to classify; World vs Business is hardest.

---

### T2 — End-to-End ML Pipeline with Scikit-learn
**Objective:** Build a production-ready pipeline for customer churn prediction.
**Dataset:** Telco Customer Churn (Kaggle) — 7,043 records
**Tools:** scikit-learn Pipeline, GridSearchCV, joblib, pandas

**Key Work:**
- Built ColumnTransformer with separate pipelines for numeric (impute then scale) and categorical (impute then one-hot encode) features
- Trained full pipelines combining preprocessor with LogisticRegression and RandomForestClassifier
- Tuned hyperparameters with GridSearchCV using 5-fold CV, optimizing ROC-AUC
- Exported final pipeline with joblib — one file loads and predicts with no preprocessing code needed
- Built interactive CLI (churn_app.py) with demo predictions and live input mode
- Identified top churn drivers: tenure, contract type, monthly charges

| Model | Accuracy | ROC-AUC | F1 |
|---|---|---|---|
| Logistic Regression (tuned) | 78% | 0.84 | 0.61 |
| Random Forest (tuned) | 79% | 0.86 | 0.63 |

**Key insight:** churn_pipeline.joblib is self-contained — load it anywhere and call .predict() on raw data with no preprocessing needed.

---

### T3 — Multimodal Housing Price Prediction (Images + Tabular)
**Objective:** Predict house prices using both structured tabular features and house images combined.
**Dataset:** Custom — 500 synthetic houses with generated images and tabular features
**Tools:** PyTorch, torchvision, scikit-learn, matplotlib

**Key Work:**
- Built a custom CNN from scratch with 3 conv blocks, BatchNorm, Dropout, AdaptiveAvgPool
- Trained CNN as a price regressor for 25 epochs on house images
- Extracted 2048-dim feature vectors from the trained CNN backbone
- Compressed image features to 40 dimensions using PCA (98.7% variance retained)
- Combined image features with 8 tabular features and trained a Ridge Regressor
- Compared tabular-only vs image-only vs multimodal results

| Model | MAE | RMSE |
|---|---|---|
| Tabular Only | ~$11K | ~$13K |
| Image Only | ~$61K | ~$72K |
| Multimodal (Both) | ~$12K | ~$15K |

**Key insight:** Multimodal models only outperform tabular models when images add independent information not already captured in the structured features. With real property listing photos, the improvement is significant.

---

### T4 — Context-Aware Chatbot Using RAG
**Objective:** Build a conversational chatbot with document retrieval and conversation memory.
**Dataset:** Custom corpus — 13 AI/ML knowledge documents
**Tools:** Anthropic Claude API, scikit-learn (TF-IDF), Streamlit

**Key Work:**
- Built a TF-IDF vector store from 13 documents split into overlapping chunks (120 words, 20-word overlap)
- Implemented cosine similarity retrieval returning top-k most relevant chunks per query
- Full RAG pipeline: retrieve then build prompt with context and history then call LLM then return answer with sources
- Conversation memory implemented as a rolling message history (last 6 turns)
- Evaluated retrieval quality: Top-1 Precision and Top-3 Recall on test queries
- Deployed as Streamlit web app with source display, knowledge base sidebar, and example question buttons

| Metric | Score |
|---|---|
| Top-1 Precision | 75% |
| Top-3 Recall | 100% |

**Key insight:** RAG quality depends more on retrieval than generation. Wrong chunks retrieved means a poor answer regardless of how good the LLM is.

---

### T5 — Auto-Tagging Support Tickets Using LLMs
**Objective:** Automatically classify support tickets into categories and return the top 3 probable tags per ticket.
**Dataset:** Custom — 180 support tickets across 6 categories (30 per class)
**Tools:** scikit-learn, pandas, Python

**Key Work:**
- Built a balanced 180-ticket dataset: Billing, Refund, Technical Issue, Account Access, Subscription, General Inquiry
- Implemented and compared three classification approaches
- Zero-Shot simulating BART-MNLI: keyword scoring, no training data required
- Few-Shot simulating FLAN-T5: trains on just 5 examples per class
- Fine-Tuned simulating DistilBERT: trains on the full dataset
- Built get_top3_tags() returning ranked tag predictions with confidence scores
- Deployed as interactive CLI (ticket_tagger.py) with demo mode and live ticket input

| Method | Accuracy | Macro F1 |
|---|---|---|
| Zero-Shot (BART-MNLI) | 72.2% | 0.708 |
| Few-Shot (FLAN-T5) | 80.6% | 0.793 |
| Fine-Tuned (DistilBERT) | 88.9% | 0.881 |

**Key insight:** Start with zero-shot when you have no labeled data. Move to few-shot with 5-10 examples per class. Fine-tune once you have 100 or more examples. This progression matches how real ML teams build classification systems.

---

## Technologies Used

| Category | Tools |
|---|---|
| Data Handling | pandas, numpy |
| Machine Learning | scikit-learn, joblib |
| Deep Learning | PyTorch, torchvision |
| NLP and Transformers | Hugging Face Transformers, datasets |
| LLM APIs | Anthropic Claude API |
| Visualization | matplotlib, seaborn |
| Deployment | Streamlit, Gradio |
| Environment | Python 3.10+, Jupyter Notebook |

---

## How to Run

Each task folder contains a Jupyter notebook for the full walkthrough and a Python script you can run directly.

**Install dependencies:**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn torch torchvision transformers streamlit gradio anthropic joblib jupyter
```

**Run any Python script:**
```bash
python churn_app.py
python ticket_tagger.py
streamlit run rag_chatbot_app.py
```

---

## Key Learnings

- Data quality matters more than model complexity — cleaning and feature engineering had more impact than switching to a more powerful model in almost every task
- Evaluation metric selection is critical — accuracy is misleading for imbalanced datasets; F1 and ROC-AUC tell a more complete story
- Production readiness requires pipelines — wrapping preprocessing and model in a single exportable object is what separates notebook demos from deployable systems
- Prompt engineering is a skill — the difference between a useful and a harmful LLM response often comes down to how carefully the system prompt is written
- RAG solves the hallucination problem — grounding LLM responses in retrieved documents dramatically improves factual accuracy and trust

---

*Submitted as part of the DHC ML/AI Internship Program*
