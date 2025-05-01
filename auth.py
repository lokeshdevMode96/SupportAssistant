import streamlit as st
from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow
from google.auth.transport import requests as grequests
import os
import json

# Firebase domain restriction
ALLOWED_DOMAIN = "prismforce.ai"

# Path to your downloaded Google client_secret.json
GOOGLE_CLIENT_SECRETS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "client_secret.json"))

# OAuth flow settings
REDIRECT_URI = "http://localhost:8501"

SCOPES = [
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "openid"
]


# Store user in session
if "user" not in st.session_state:
    st.session_state.user = None

def login_user():
    if st.session_state.get("user"):
        return st.session_state.user

    query_params = st.query_params
    has_code = "code" in query_params and "state" in query_params

    client_config = json.loads(st.secrets["client_config"])


    flow = Flow.from_client_config(
        client_config,
        scopes=SCOPES
    )
    flow.redirect_uri = REDIRECT_URI

    if not has_code:
        # Show login UI only if no auth code is present
        st.title("🔐 Sign in to use Support Assistant")
        auth_url, _ = flow.authorization_url(prompt='consent')
        st.markdown(f"[Click here to sign in with Google]({auth_url})")
        st.stop()

    # Handle Google redirect with code
    code = query_params["code"]
    state = query_params["state"]

    # Rebuild flow and fetch token
    flow.fetch_token(code=code)
    credentials = flow.credentials

    # Verify ID token
    request = grequests.Request()
    client_id = client_config["web"]["client_id"]
    idinfo = id_token.verify_oauth2_token(credentials._id_token, request, audience=client_id)

    email = idinfo.get("email")
    domain = email.split("@")[1] if email else ""

    if domain != ALLOWED_DOMAIN:
        st.error(f"Access denied: Use your @{ALLOWED_DOMAIN} email")
        st.stop()

    user = {
        "email": email,
        "name": idinfo.get("name"),
        "is_authenticated": True
    }

    st.session_state.user = user
    st.rerun()
