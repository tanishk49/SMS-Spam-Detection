# 📩 Spam SMS Detection with Naive Bayes

 
 ## Objective
 Build a machine learning model that classifies SMS messages as 'spam' or 'ham' (non-spam) 
 using the Naive Bayes algorithm and Natural Language Processing techniques.

 ## Dataset
 Source: Kaggle - SMS Spam Collection Dataset
 URL: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
 Description: 5,574 English SMS messages labeled as 'spam' or 'ham'

 ## Project Structure
 spam-sms-detection/ </br>
 ├── spam_sms_detection.ipynb    Jupyter Notebook with full implementation </br>
 ├── dataset/                      </br>
   └── spam.csv                Downloaded dataset from Kaggle     </br>
 ├── README.md                   Project documentation              </br>
 └── requirements.txt            Required Python packages           </br>

 ## Technologies Used
 - Language: Python 3.x
 - Libraries: pandas, numpy, nltk, sklearn, matplotlib, seaborn

 ## Data Preprocessing Steps
 1. Lowercasing
 2. Removing punctuation and digits
 3. Tokenization
 4. Stopword removal using NLTK
 5. Stemming using PorterStemmer

 ## Feature Extraction
 - Using Bag of Words model with CountVectorizer
 - Converts messages into a matrix of token counts (sparse matrix)

 ## Model Used
 - Multinomial Naive Bayes
 - Suitable for text classification problems with word frequency as features

 ## Model Evaluation
 - Split data into 80% training and 20% testing
 - Evaluation Metrics:
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - Confusion Matrix

# Example Metrics:
 Accuracy: 98%
 Precision: 97-98%
 Recall: 95-98%
 F1 Score: 96-98%

 ## Insights
 - Spam messages often contain keywords like "free", "win", "cash"
 - Ham messages are typically conversational and shorter
 - Naive Bayes performs well with clean text and frequency-based features

 ## How to Run
# 1. Clone the repository
    git clone https://github.com/tanishk49/SMS-Spam-Detection.git
    cd SMS-Spam-Detection

# 2. Create a virtual environment (optional)
    python -m venv venv
    source venv/bin/activate  (on Windows: venv\Scripts\activate)

# 3. Install dependencies
    pip install -r requirements.txt

# 4. Download the dataset from Kaggle and place `spam.csv` in the `dataset/` folder

# 5. Run the notebook
    jupyter notebook SMS-Spam-Detection.ipynb

 ## Requirements
 - pandas
 - numpy
 - nltk
 - scikit-learn
 - matplotlib
 - seaborn

 ## Future Improvements
 - Use TF-IDF instead of Bag of Words
 - Try other ML models: Logistic Regression, SVM, Random Forest
 - Deploy using Streamlit or Flask
 - Add real-time SMS classification functionality

 ## License
 MIT License

