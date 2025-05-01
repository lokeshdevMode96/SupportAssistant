import streamlit as st
import faiss
import json
import os
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from feedback_logger import init_db, save_feedback
from auth import login_user

init_db()

# Load model, index, metadata
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index('vector_store/index.faiss')

with open('vector_store/ticket_metadata.json', 'r') as f:
    metadata = json.load(f)

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


# Helper to retrieve top K similar tickets
def get_similar_tickets(query, top_k=3):
    query_vector = normalize(model.encode([query])).astype('float32')
    distances, indices = index.search(query_vector, top_k)
    results = []
    for idx in indices[0]:
        if idx < len(metadata):
            results.append(metadata[idx])
    return results

# Streamlit UI
st.set_page_config(page_title="Support Assistant", layout="centered")

user = login_user()
st.session_state.user = user  # store for later use

st.markdown(f"👤 Logged in as: **{user['name']}** ({user['email']})")


if "mode" not in st.session_state:
    st.session_state.mode = "search"

if "uploaded" in st.session_state and not st.session_state.uploaded:
    del st.session_state.uploaded

# Reset rerun flag if it exists (used to prevent upload loop)
if "rerun_once" in st.session_state:
    del st.session_state.rerun_once


st.title("🛠️ Prismforce Support Assistant")


# Set default menu selection in session state
if "menu" not in st.session_state:
    st.session_state.menu = "🔍 Search"

# Render dropdown with session-selected index
menu = st.selectbox(
    "Select an option:",
    ["🔍 Search", "📁 Upload training data"],
    index=["🔍 Search", "📁 Upload training data"].index(st.session_state.menu)
)

# Update mode and menu in sync
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

            # Reload index + metadata
            index = faiss.read_index('vector_store/index.faiss')
            with open('vector_store/ticket_metadata.json', 'r') as f:
                metadata = json.load(f)

            st.session_state.uploaded = True
            st.rerun()

    # Show preview after rerun
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


        # Create compressed table view
        table_data = []
        for i, ticket in enumerate(results, 1):
            table_data.append({
                "Match #": f"{i}",
                "Ticket ID": ticket['ticket_id'],
                "Product": ticket['product'],
                "Issue": ticket['issue'],
                "Resolution": ticket['resolution']
            })

        st.markdown("### 🧾 Matched Tickets")
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)

        if results:
            # Build prompt using retrieved tickets
            context = "\n\n".join([
                f"Issue: {r['issue']}\nResolution: {r['resolution']}" for r in results
            ])

            prompt = f"""
            A user raised this issue: "{query}".
            Based on the similar past tickets and resolutions below, suggest a helpful resolution.

            {context}

            Provide a clear and brief recommendation.
            """

            with st.spinner("🤖 Generating resolution using Mistral..."):
                try:
                    
                    # Store generated suggestion in session state
                    if 'suggestion' not in st.session_state:
                        suggestion = get_ollama_response(prompt)
                        st.session_state.suggestion = suggestion.strip()

                    st.success("✅ Suggested Resolution:")
                    st.markdown(st.session_state.suggestion.strip())

                    # Feedback buttons
                    col1, col2 = st.columns([1, 1]) 
                    with col1:
                        if st.button("👍 Helpful"):
                            ticket_ids = ", ".join([r['ticket_id'] for r in results])
                            save_feedback(query, ticket_ids, st.session_state.suggestion.strip(), "positive")
                            st.success("Thanks for your feedback! 👍")

                    with col2:
                        if st.button("👎 Not helpful"):
                            ticket_ids = ", ".join([r['ticket_id'] for r in results])
                            save_feedback(query, ticket_ids, st.session_state.suggestion.strip(), "negative")
                            st.info("Got it — we'll use this to improve.")

                except Exception as e:
                    st.error(f"⚠️ Failed to generate response from Mistral: {e}")


        if not results:
            st.warning("No similar tickets found. Try a more general query.")
