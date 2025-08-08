# plp-final-project-mind-link
🏥 Smart Health Access – Random Forest Classifier
Smart Health Access is a machine learning–based classification system designed to predict health access levels (e.g., High, Medium, Low) based on socio-economic and demographic data.
The model uses Random Forest Classification with hyperparameter tuning and class weighting to handle imbalanced datasets.

📌 Features
Automatic Categorical Encoding – Converts Location and Condition into numeric labels.

Feature Engineering – Adds derived features like Income_per_Age to improve model accuracy.

Class Imbalance Handling – Uses scikit-learn's compute_class_weight for balanced training.

Hyperparameter Tuning – Uses GridSearchCV to find the best Random Forest parameters.

Detailed Evaluation – Outputs accuracy score and classification report for all classes.

🛠 Tech Stack
Language: Python 3.10+

Libraries:

pandas

scikit-learn

numpy

📂 Project Structure
bash
Copy
Edit
smart-health-access/
│
├── data/
│   └── train.csv       # Dataset
├── home.py             # Main Python script
├── README.md           # Project documentation
└── requirements.txt    # Python dependencies
📊 Dataset
The dataset should be stored in:

bash
Copy
Edit
data/train.csv
It must contain the following columns:

Income (numeric) – Monthly or annual income.

Location (categorical) – Geographic location of the individual.

Age (numeric) – Age in years.

Condition (categorical) – Health condition category.

HealthAccess (categorical) – Target variable (High, Medium, Low).

⚙️ How It Works
Data Preprocessing

Label encodes categorical features (Location, Condition).

Adds a derived feature Income_per_Age.

Model Training

Splits the dataset into training (70%) and test (30%) sets with stratification.

Applies class weights to reduce bias toward majority classes.

Hyperparameter Tuning

Runs GridSearchCV over parameters:

n_estimators = [100, 200, 300]

max_depth = [None, 10, 15, 20]

min_samples_split = [2, 5, 10]

Evaluation

Prints best parameters, accuracy, and a classification report.

🚀 Installation & Usage
1️⃣ Clone the Repository

bash
Copy
Edit
git clone https://github.com/yourusername/smart-health-access.git
cd smart-health-access
2️⃣ Install Dependencies

bash
Copy
Edit
pip install -r requirements.txt
3️⃣ Add the Dataset
Place your train.csv file inside the data/ folder.

4️⃣ Run the Script

bash
Copy
Edit
python home.py
📈 Example Output
yaml
Copy
Edit
Best Parameters: {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 200}
Model Accuracy: 0.87

Classification Report:
               precision    recall  f1-score   support
        High       0.88      0.90      0.89        50
         Low       0.86      0.85      0.85        45
      Medium       0.87      0.86      0.86        55
⚖️ License
This project is licensed under the MIT License – feel free to use, modify, and distribute.

