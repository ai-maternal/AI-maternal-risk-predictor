import streamlit as st
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

st.set_page_config(page_title="Login", page_icon="🔐", layout="centered")
st.title("🔐 Login Page")

with open("config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)

# ✅ Version-proof login call (tries the right signature automatically)
try:
    # Most common signature in newer versions:
    # login(location='main', ...)
    name, authentication_status, username = authenticator.login(location="main")
except TypeError:
    # Other common signature:
    # login(form_name, location)
    name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status:
    st.success(f"Welcome {name} 👋")
    st.session_state["logged_in"] = True
    st.session_state["username"] = username

    # ✅ Version-proof logout
    try:
        authenticator.logout(location="sidebar")
    except TypeError:
        authenticator.logout("Logout", "sidebar")

    st.info("Now open your Predictor page from the sidebar ✅")

elif authentication_status is False:
    st.error("❌ Username/password is incorrect")

else:
    st.warning("Please enter your username and password")