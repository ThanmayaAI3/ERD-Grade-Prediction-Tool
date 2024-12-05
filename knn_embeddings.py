import os
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

base_path = "/Users/ishaanpathania/Downloads/Collection 3"
datasets = ["Dataset1", "Dataset2"]

model = SentenceTransformer('all-MiniLM-L6-v2')
all_embeddings = {}
all_grades = {}

for dataset in datasets:
    dataset_path = os.path.join(base_path, dataset)
    doc_texts = []
    doc_names = []
    
    for i in range(1, 137):
        file_path = os.path.join(dataset_path, f"{i}.txt")
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                doc_texts.append(file.read().strip())
                doc_names.append(f"{i}")
        except FileNotFoundError:
            continue

    embeddings = model.encode(doc_texts)
    all_embeddings[dataset] = embeddings
    
    grades_data = pd.read_csv(os.path.join(base_path, "ERD_grades.csv"), sep="\t")
    grades = []
    grade_col = f"{dataset.lower()}_grade"
    for i in range(1, len(doc_texts) + 1):
        grade = grades_data.loc[grades_data['ERD_No'] == i, grade_col].values
        if grade.size > 0:
            grades.append(grade[0])
        else:
            grades.append(None)

    all_grades[dataset] = np.array(grades)

predicted_grades = {"ERD_No": [int(doc_names[i]) for i in range(101, len(doc_names))], "dataset1_grade": [], "dataset2_grade": []}

for dataset in datasets:
    grades = all_grades[dataset]
    embeddings = all_embeddings[dataset]
    labeled_mask = ~pd.isnull(grades)
    X_labeled = embeddings[labeled_mask]
    y_labeled = grades[labeled_mask]
    X_unlabeled = embeddings[~labeled_mask]
    unlabeled_indices = np.where(~labeled_mask)[0]
    
    if len(X_labeled) > 0:
        X_train, X_val, y_train, y_val = train_test_split(X_labeled, y_labeled, test_size=0.2, random_state=42)
        knn = KNeighborsRegressor(n_neighbors=10, metric="cosine")
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        print(f"{dataset}: Mean Absolute Error: {mae}")
        
        if len(X_unlabeled) > 0:
            knn = KNeighborsRegressor(n_neighbors=10, metric="cosine")
            knn.fit(X_labeled, y_labeled)
            y_unlabeled_pred = knn.predict(X_unlabeled)
        else:
            y_unlabeled_pred = []
    else:
        y_unlabeled_pred = []
    
    col = f"{dataset.lower()}_grade"
    for pred in y_unlabeled_pred:
        predicted_grades[col].append(round(pred, 2))
        
output_data = pd.DataFrame(predicted_grades)
output_data.to_csv("ERD_grades.csv", index=False)
print("Predicted grades saved to ERD_grades.csv.")

# # Read documents
# erds = "/Users/ishaanpathania/Downloads/Collection 3/Dataset2"
# doc_texts = []

# for i in range(1, 137):
#     file_path = os.path.join(erds, f"{i}.txt")
#     try:
#         with open(file_path, 'r', encoding='utf-8') as file:
#             doc_texts.append(file.read().strip())
#     except FileNotFoundError:
#         continue

# # Generate embeddings based on method
# model = SentenceTransformer('all-MiniLM-L6-v2')
# embeddings = model.encode(doc_texts)

# # Load grades
# grades_data = pd.read_csv("/Users/ishaanpathania/Downloads/Collection 3/ERD_grades.csv", sep="\t")
# grades = []
# for i in range(1, len(doc_texts) + 1):
#     grade = grades_data.loc[grades_data['ERD_No'] == i, 'dataset1_grade'].values
#     if grade.size > 0:
#         grades.append(grade[0])
#     else:
#         grades.append(None)

# # Filter out documents without grades and perform KNN
# grades = np.array(grades)
# labeled_mask = ~pd.isnull(grades)
# X_labeled = embeddings[labeled_mask]
# y_labeled = grades[labeled_mask]
# X_unlabeled = embeddings[~labeled_mask]

# # kf = KFold(n_splits=5, shuffle=True, random_state=42)
# # mae_scores = []

# # for train_idx, test_idx in kf.split(X_labeled):
# #     X_train, X_val = X_labeled[train_idx], X_labeled[test_idx]
# #     y_train, y_val = y_labeled[train_idx], y_labeled[test_idx]
# #     knn = KNeighborsRegressor(n_neighbors=10, metric="cosine")
# #     knn.fit(X_train, y_train)
# #     y_pred = knn.predict(X_val)
# #     mae_scores.append(mean_absolute_error(y_val, y_pred))

# # print(f"Cross-Validated Mean Absolute Error: {np.mean(mae_scores):.4f}")

# X_train, X_val, y_train, y_val = train_test_split(X_labeled, y_labeled, test_size=0.2, random_state=42)
# knn = KNeighborsRegressor(n_neighbors=10, metric="cosine")
# knn.fit(X_train, y_train)

# y_pred = knn.predict(X_val)
# mae = mean_absolute_error(y_val, y_pred)
# print(f"Mean Absolute Error: {mae}")

# if len(X_unlabeled) > 0:
#     knn = KNeighborsRegressor(n_neighbors=10, metric="cosine")
#     knn.fit(X_labeled, y_labeled)
#     y_unlabeled_pred = knn.predict(X_unlabeled)
#     output = pd.DataFrame({
#         "Document": [f"Document {i+1}" for i in range(len(doc_texts)) if i not in labeled_mask.nonzero()[0]],
#         "Predicted Grade": y_unlabeled_pred
#     })
#     output.to_csv("predicted_grades_embeddings.csv", index=False)
#     print("Predicted grades saved to predicted_grades_embeddings.csv.")
# else:
#     print("No unlabeled documents found.")