import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Load the data
file_path = 'learning_pace_factors.csv'
df = pd.read_csv(file_path)

# Adjust the column selection to focus on the actual responses
response_column_name = df.columns[1]  # The second column contains the actual responses
df_cleaned = df.dropna(subset=[response_column_name])

# Preprocess the text using TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = tfidf_vectorizer.fit_transform(df_cleaned[response_column_name].tolist())

# Use KMeans clustering with a suitable number of clusters
kmeans_refined_simple = KMeans(n_clusters=10, random_state=0)
kmeans_refined_simple.fit(X_tfidf)
clusters_refined_simple = kmeans_refined_simple.labels_

# Add the clusters back to the cleaned dataframe
df_cleaned['cluster'] = clusters_refined_simple

# Save each cluster to a separate sheet in an Excel file
output_path = '/mnt/data/clustered_responses.xlsx'
with pd.ExcelWriter(output_path) as writer:
    for cluster_num in range(10):
        cluster_data = df_cleaned[df_cleaned['cluster'] == cluster_num]
        cluster_data.to_excel(writer, sheet_name=f'Cluster_{cluster_num}', index=False)

print(f"Clustered data has been saved to {output_path}")