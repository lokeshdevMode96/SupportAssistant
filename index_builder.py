def build_index():
    import os
    import pandas as pd
    import faiss
    import json
    from sentence_transformers import SentenceTransformer
    from sklearn.preprocessing import normalize

    # Load ticket CSV
    csv_path = 'data/support_tickets.csv'
    df = pd.read_csv(csv_path)

    # Combine issue and resolution
    texts = df['issue'].tolist()

    # Load model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = normalize(model.encode(texts)).astype('float32')

    # Create index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    os.makedirs('vector_store', exist_ok=True)
    faiss.write_index(index, 'vector_store/index.faiss')

    metadata = df.to_dict(orient='records')
    with open('vector_store/ticket_metadata.json', 'w') as f:
        json.dump(metadata, f)

    print("✅ FAISS index and metadata saved!")

# Optional: run on script call
if __name__ == "__main__":
    build_index()
