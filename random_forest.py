import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Step 1: Read All Documents
folder_path = "Collection 3/Dataset1"  # Replace with your folder path
doc_texts = []  # List to store processed content of each document
doc_names = []

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
            doc_names.append(f"{i}")
    except FileNotFoundError:
        continue




# dataset 2

folder_path_2 = "Collection 3/Dataset2"  # Replace with your folder path
doc_texts_2 = []  # List to store processed content of each document

for i in range(1, 137):
    file_path_2 = os.path.join(folder_path_2, f"{i}.txt")
    try:
        with open(file_path_2, 'r', encoding='utf-8') as file:
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
            doc_texts_2.append(" ".join(processed_lines))
    except FileNotFoundError:
        continue

######


# print("Lenght of folders")
# for i in range(101,len(doc_names)):
#     print(doc_names[i])

# Step 2: Generate TF-IDF Matrix
vectorizer = TfidfVectorizer(stop_words='english', use_idf=True)
tfidf_matrix = vectorizer.fit_transform(doc_texts)

# Step 3: Convert to DataFrame for Analysis
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# Save to CSV (Optional)
output_path = "tfidf_vectors.csv"
tfidf_df.to_csv(output_path, index=False)
# print(f"TF-IDF vectors saved to {output_path}")




###### Dataset 2


# Step 2: Generate TF-IDF Matrix
vectorizer_2 = TfidfVectorizer(stop_words='english', use_idf=True)
tfidf_matrix_2 = vectorizer_2.fit_transform(doc_texts)

# Step 3: Convert to DataFrame for Analysis
tfidf_df_2 = pd.DataFrame(tfidf_matrix_2.toarray(), columns=vectorizer_2.get_feature_names_out())

# Save to CSV (Optional)
output_path = "tfidf_vectors_2.csv"
tfidf_df_2.to_csv(output_path, index=False)
# print(f"TF-IDF vectors saved to {output_path}")




### run a KNN

#gather grades
grades_df = pd.read_csv("Collection 3/ERD_grades.csv", sep='\t')
grades_1 = []
grades_2 = []

pd.set_option("display.max_rows", None)
# print(grades_df.sort_values(by = ["ERD_No"]))

grades_df = grades_df.sort_values(by = ["ERD_No"])
# print(grades_df.tail())


for i in range(1,104):
    grade = grades_df.loc[grades_df['ERD_No'] == i, 'dataset1_grade'].values
    if grade.size > 0:  # Ensure grade exists
        grades_1.append(grade[0])


for i in range(1,104):
    grade = grades_df.loc[grades_df['ERD_No'] == i, 'dataset2_grade'].values
    if grade.size > 0:  # Ensure grade exists
        grades_2.append(grade[0])

print(len(grades_1))

print(len(grades_2))



######## Model creation
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Step 1: Split Data into Labeled and Unlabeled
# grades = [85, 90, 78, 92, ...]  # Replace with grades for Rows 1–100
X_labeled = tfidf_matrix[:101]  # TF-IDF vectors for labeled documents
y_labeled = grades_1              # Grades for labeled documents
X_unlabeled = tfidf_matrix[101:]  # TF-IDF vectors for unlabeled documents

# Step 2: Train-Test Split for Validation
X_train, X_val, y_train, y_val = train_test_split(X_labeled, y_labeled, test_size=0.2, random_state=42)

# Step 3: Train KNN Model
model = RandomForestRegressor(random_state=42)



#knn = KNeighborsRegressor(n_neighbors=10, metric='cosine')  # Experiment with k and metrics
model.fit(X_train, y_train)

# Step 4: Validate the Model
y_val_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_val_pred)
print(f"Validation MSE: {mse}")

# Step 5: Predict Grades for Unlabeled Documents
y_unlabeled_pred = model.predict(X_unlabeled)




X_labeled_2 = tfidf_matrix_2[:101]  # TF-IDF vectors for labeled documents
y_labeled_2 = grades_2              # Grades for labeled documents
X_unlabeled_2 = tfidf_matrix_2[101:]  # TF-IDF vectors for unlabeled documents

# Step 2: Train-Test Split for Validation
X_train_2, X_val_2, y_train_2, y_val_2 = train_test_split(X_labeled_2, y_labeled_2, test_size=0.2, random_state=42)

# Step 3: Train KNN Model
model_2 = RandomForestRegressor(random_state=42)



#knn = KNeighborsRegressor(n_neighbors=10, metric='cosine')  # Experiment with k and metrics
model_2.fit(X_train_2, y_train_2)

# Step 4: Validate the Model
y_val_pred_2 = model_2.predict(X_val_2)
mse_2 = mean_squared_error(y_val_2, y_val_pred_2)
print(f"Validation MSE 2: {mse_2}")

# Step 5: Predict Grades for Unlabeled Documents
y_unlabeled_pred_2 = model_2.predict(X_unlabeled_2)







# Save Predictions
import pandas as pd
output = pd.DataFrame({
    "ERD_No": [f"{doc_names[i]}" for i in range(101,len(doc_names))],
    "dataset1_grade": y_unlabeled_pred,
    "dataset2_grade":y_unlabeled_pred_2
})
output["dataset1_grade"] = output["dataset1_grade"].round(2)
output["dataset2_grade"] = output["dataset2_grade"].round(2)

output.to_csv("ERD_grades.csv", index=False)
