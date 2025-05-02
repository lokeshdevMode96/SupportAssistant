def build_index():
    import os
    import pandas as pd
    import numpy as np
    import json
    from sentence_transformers import SentenceTransformer
    from sklearn.preprocessing import normalize
    from sklearn.neighbors import NearestNeighbors
    import joblib

    # Load ticket CSV
    csv_path = 'data/support_tickets.csv'
    df = pd.read_csv(csv_path)

    # Combine issue and resolution
    texts = df['issue'].tolist()

    # Load model
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    vectors = normalize(model.encode(texts)).astype('float32')

    # Save vectors + metadata using joblib
    metadata = df.to_dict(orient='records')
    os.makedirs("vector_store", exist_ok=True)
    joblib.dump((vectors, metadata), "vector_store/knn_index.pkl")

    print("✅ KNN index and metadata saved!")


# Optional: run on script call
if __name__ == "__main__":
    build_index()
