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

# Save to CSV
output_path = "tfidf_vectors.csv"
tfidf_df.to_csv(output_path, index=False)
print(f"TF-IDF vectors saved to {output_path}")

### run a KNN yet to implement


