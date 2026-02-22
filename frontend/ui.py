import streamlit as st
from backend.embed import result, embed, clear_db

# Streamlit page setup
st.set_page_config(page_title="PapyrusAI", layout="wide")

st.markdown(
"""
<style>
    .st-key-chat {
        padding-bottom: 10px;
    }

    .st-key-new {
        margin-left: auto;
        margin-right: auto;
    }

    @media (max-width: 1200px) {
        .st-key-new {
            transform: translate(65%, 35%);
        }
    }

    @media (max-width: 992px) {
        .st-key-new {
            transform: translate(65%, 35%);
        }
    }

    @media (max-width: 768px) {
        .st-key-new {
            transform: translate(65%, 35%);
        }
    }

    @media (max-width: 480px) {
        .st-key-new {
            transform: translate(65%, 35%);
        }
    }

</style>
""",
unsafe_allow_html=True
)


# -------------------- Session State --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}
# if "embedded_files" not in st.session_state:
#     st.session_state.embedded_files = []
if "user_query" not in st.session_state:
    st.session_state.user_query = ""
if "new_chat" not in st.session_state:
    st.session_state.new_chat = True
# if "processing" not in st.session_state:
#     st.session_state.processing = False
# -------------------- Header --------------------
left, right = st.columns([0.50, 0.50])
with left:
    st.title("PapyrusAI🍀")
    st.caption("Add your PDFs and chat with them with the power of LLaMa3")
with right:
    if st.button("🆕 New Chat", help="Start new chat", key="new"):
        st.session_state.messages = []
        st.session_state.new_chat = True
# -------------------- Right Panel (Tools) --------------------

with st.sidebar:
    st.title("⚙️ Upload your documents")
    with st.container(border=True):
        st.markdown("**Add Files**")
        files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help="You can upload multiple PDF files.")
        if files:
            for f in files:
                st.session_state.uploaded_files[f.name] = f.getvalue()

        if len(st.session_state.uploaded_files)>=1:
            if st.button("📥 Embed", type="primary", use_container_width=True):
                selected_files = {file: st.session_state.uploaded_files[file] for file in st.session_state.uploaded_files.keys()}
                embed(selected_files)
                st.session_state.uploaded_files = {}
                st.success("Embedding complete!")
        else:
            st.info("No files uploaded yet.")

    with st.container(border=True):
        st.markdown("**Embedded Docs**")
        st.caption("(List embedded docs here)") # st.session_state.embedded_files = []
        cols = st.columns(2)
        with cols[0]:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        with cols[1]:
            if st.button("🧹 Clear DB", use_container_width=True):
                clear_db()
# -------------------- Chat (Main) --------------------
user_query = st.chat_input("Ask about your PDFs...", key="input", disabled=st.session_state.processing)
with st.container(key="chat"):
    st.subheader("💬 Chat")        
    # if user_query and not st.session_state.processing :
    if user_query:
        # st.session_state.processing = True
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.spinner("Thinking..."):
            try:
                response = result(user_query, st.session_state.new_chat)
                if st.session_state.new_chat:
                    st.session_state.new_chat = False
            except Exception as e:
                response = f"Error calling result(): {e}"
        st.session_state.messages.append({"role": "assistant", "content": response})
        # st.session_state.processing = False

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])