import streamlit as st
import os
import json
import pandas as pd
import requests
import joblib
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from feedback_logger import init_db, save_feedback
from auth import login_user

init_db()

# Load or rebuild index
index_path = "vector_store/knn_index.pkl"
metadata_path = "vector_store/ticket_metadata.json"

# If both files exist, load them
if os.path.exists(index_path) and os.path.exists(metadata_path):
    vectors, metadata = joblib.load(index_path)
else:
    # Build index only if missing
    from index_builder import build_index
    build_index()
    vectors, metadata = joblib.load(index_path)

#model = SentenceTransformer('all-MiniLM-L6-v2')
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')


# Function to call Mistral via Ollama locally
def get_ollama_response(prompt):
    response = requests.post(
        'http://localhost:11434/api/generate',
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        }
    )
    try:
        return response.json()['response']
    except Exception as e:
        print("⚠️ Ollama response error:", response.text)
        raise

# Search helper
def get_similar_tickets(query, top_k=3):
    query_vector = normalize(model.encode([query])).astype('float32')
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=top_k, metric='cosine')
    nn.fit(vectors)
    distances, indices = nn.kneighbors(query_vector)
    return [metadata[i] for i in indices[0]]

# UI
st.set_page_config(page_title="Support Assistant", layout="centered")
user = login_user()
st.session_state.user = user
st.markdown(f"👤 Logged in as: **{user['name']}** ({user['email']})")

if "mode" not in st.session_state:
    st.session_state.mode = "search"
if "uploaded" in st.session_state and not st.session_state.uploaded:
    del st.session_state.uploaded
if "rerun_once" in st.session_state:
    del st.session_state.rerun_once

st.title("🛠️ Prismforce Support Assistant")

if "menu" not in st.session_state:
    st.session_state.menu = "🔍 Search"
menu = st.selectbox(
    "Select an option:",
    ["🔍 Search", "📁 Upload training data"],
    index=["🔍 Search", "📁 Upload training data"].index(st.session_state.menu)
)
st.session_state.menu = menu
st.session_state.mode = "search" if menu == "🔍 Search" else "upload"

if st.session_state.mode == "upload":
    st.markdown("### 📁 Upload New Ticket Data")
    if "uploaded" not in st.session_state:
        uploaded_file = st.file_uploader("Upload a CSV file with support tickets", type="csv")
        if uploaded_file:
            os.makedirs("data", exist_ok=True)
            with open("data/support_tickets.csv", "wb") as f:
                f.write(uploaded_file.getbuffer())
            from index_builder import build_index
            build_index()
            vectors, metadata = joblib.load("vector_store/knn_index.pkl")
            st.session_state.uploaded = True
            st.rerun()

    if st.session_state.get("uploaded"):
        df_preview = pd.read_csv("data/support_tickets.csv")
        st.success("✅ Index updated successfully!")
        st.markdown("### 🔍 Uploaded File Preview")
        st.dataframe(df_preview.head(), use_container_width=True)
        if st.button("🔄 Return to search"):
            st.session_state.uploaded = False
            st.session_state.mode = "search"
            st.session_state.menu = "🔍 Search"
            st.rerun()

if st.session_state.mode == "search":
    query = st.text_input("Enter your support query:", placeholder="e.g., Certifications not syncing for LTI")
    if query:
        with st.spinner("🔍 Searching similar tickets..."):
            results = get_similar_tickets(query)

        table_data = [{
            "Match #": f"{i+1}",
            "Ticket ID": r['ticket_id'],
            "Product": r['product'],
            "Issue": r['issue'],
            "Resolution": r['resolution']
        } for i, r in enumerate(results)]

        st.markdown("### 🧾 Matched Tickets")
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)

        if results:
            context = "\n\n".join([f"Issue: {r['issue']}\nResolution: {r['resolution']}" for r in results])
            prompt = f"""
            A user raised this issue: "{query}".
            Based on the similar past tickets and resolutions below, suggest a helpful resolution.

            {context}

            Provide a clear and brief recommendation.
            """

            with st.spinner("🤖 Generating resolution using Mistral..."):
                try:
                    if 'suggestion' not in st.session_state:
                        st.session_state.suggestion = get_ollama_response(prompt).strip()
                    st.success("✅ Suggested Resolution:")
                    st.markdown(st.session_state.suggestion)
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if st.button("👍 Helpful"):
                            ticket_ids = ", ".join([r['ticket_id'] for r in results])
                            save_feedback(query, ticket_ids, st.session_state.suggestion, "positive")
                            st.success("Thanks for your feedback! 👍")
                    with col2:
                        if st.button("👎 Not helpful"):
                            ticket_ids = ", ".join([r['ticket_id'] for r in results])
                            save_feedback(query, ticket_ids, st.session_state.suggestion, "negative")
                            st.info("Got it — we'll use this to improve.")
                except Exception as e:
                    st.error(f"⚠️ Failed to generate response from Mistral: {e}")
        else:
            st.warning("No similar tickets found. Try a more general query.")
