import streamlit as st


DEFAULTS = {
    # AI flow
    "ai_generated": False,
    "ai_report_md": "",
    "ai_prompt_used": "",
    "prompt_locked": False,  # lock after first analyze
    "approved": False,
    "revise_nonce": 0,

    # UI messages
    "last_status": "",
    "last_error": "",

    # filter signature (auto reset AI when filters change)
    "filters_sig": None,
}


def init_state() -> None:
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_ai_state() -> None:
    """Call BEFORE rendering AI widgets."""
    st.session_state.ai_generated = False
    st.session_state.ai_report_md = ""
    st.session_state.ai_prompt_used = ""
    st.session_state.prompt_locked = False
    st.session_state.approved = False
    st.session_state.revise_nonce = 0
    st.session_state.last_status = ""
    st.session_state.last_error = ""
