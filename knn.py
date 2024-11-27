import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Step 1: Read All Documents
folder_path = "Collection 3 for stage 2/Collection 3/Dataset1"  # Replace with your folder path
doc_texts = []  # List to store processed content of each document

for i in range(1, 137):
    file_path = os.path.join(folder_path, f"{i}.txt")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Read and normalize the file content
            content = file.readlines()
            processed_lines = [
                line.replace('\n', '')
                    .replace(' ', '_')
                    .replace('[', '')
                    .replace(']', '')
                    .replace("'", '')
                    .replace(',_', ' ')
                    .strip()
                for line in content if line.strip()
            ]
            # Combine processed lines into a single string for the document
            doc_texts.append(" ".join(processed_lines))
    except FileNotFoundError:
        continue



# Step 2: Generate TF-IDF Matrix
vectorizer = TfidfVectorizer(stop_words='english', use_idf=True)
tfidf_matrix = vectorizer.fit_transform(doc_texts)

# Step 3: Convert to DataFrame for Analysis
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# Save to CSV (Optional)
output_path = "tfidf_vectors.csv"
tfidf_df.to_csv(output_path, index=False)
print(f"TF-IDF vectors saved to {output_path}")

### run a KNN

#gather grades
grades_df = pd.read_csv("Collection 3 for stage 2/Collection 3/ERD_grades.csv", sep='\t')
grades = []
pd.set_option("display.max_rows", None)
print(grades_df.sort_values(by = ["ERD_No"]))
for i in range(1,104):
    grade = grades_df.loc[grades_df['ERD_No'] == i, 'dataset1_grade'].values
    if grade.size > 0:  # Ensure grade exists
        grades.append(grade[0])
    else:
        grades.append(None)  # Handle missing grade
print(grades)


##################### THIS IS CHAT GPT
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Step 1: Split Data into Labeled and Unlabeled
grades = [85, 90, 78, 92, ...]  # Replace with grades for Rows 1–100
X_labeled = tfidf_matrix[:100]  # TF-IDF vectors for labeled documents
y_labeled = grades              # Grades for labeled documents
X_unlabeled = tfidf_matrix[100:]  # TF-IDF vectors for unlabeled documents

# Step 2: Train-Test Split for Validation
X_train, X_val, y_train, y_val = train_test_split(X_labeled, y_labeled, test_size=0.2, random_state=42)

# Step 3: Train KNN Model
knn = KNeighborsRegressor(n_neighbors=5, metric='cosine')  # Experiment with k and metrics
knn.fit(X_train, y_train)

# Step 4: Validate the Model
y_val_pred = knn.predict(X_val)
mse = mean_squared_error(y_val, y_val_pred)
print(f"Validation MSE: {mse}")

# Step 5: Predict Grades for Unlabeled Documents
y_unlabeled_pred = knn.predict(X_unlabeled)

# Save Predictions
import pandas as pd
output = pd.DataFrame({
    "Document": [f"Document {i+101}" for i in range(len(y_unlabeled_pred))],
    "Predicted Grade": y_unlabeled_pred
})
output.to_csv("predicted_grades.csv", index=False)
print("Predicted grades saved to predicted_grades.csv.")
