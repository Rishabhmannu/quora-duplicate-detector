import streamlit as st
import helper
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
_models_dir = _project_root / "models"

st.set_page_config(page_title="Quora Duplicate Detector", page_icon="🔍")
st.title("Duplicate Question Pairs")
st.caption("Enter two questions to check if they are semantically duplicate.")

# Model selector
available = helper.get_available_models()
if not available:
    st.error("No models found. Run scripts/04_train_and_save.py first.")
    st.stop()

inference_times = helper.get_inference_times()
model_options = {helper.get_model_display_name(m): m for m in available}

with st.sidebar:
    st.subheader("Model")
    selected_label = st.selectbox(
        "Choose model",
        options=list(model_options.keys()),
        index=0,
        key="model_select",
    )
    selected_model = model_options[selected_label]

    # Show inference time if benchmarked
    if inference_times:
        key = "classical" if selected_model == "classical" else "transformer"
        if key in inference_times:
            mean_ms = inference_times[key].get("mean_ms", 0)
            st.caption(f"~{mean_ms:.0f} ms per prediction")

# Apply prefill from example buttons (must run before text_input widgets are created)
if "_prefill_q1" in st.session_state:
    st.session_state.q1 = st.session_state.pop("_prefill_q1", "")
    st.session_state.q2 = st.session_state.pop("_prefill_q2", "")

q1 = st.text_input("Enter question 1", placeholder="e.g. What is the capital of India?", key="q1")
q2 = st.text_input("Enter question 2", placeholder="e.g. Which city is India's capital?", key="q2")

# Example pairs
def _set_duplicate_example():
    st.session_state["_prefill_q1"] = "How do I learn Python?"
    st.session_state["_prefill_q2"] = "What is the best way to learn Python programming?"

def _set_not_duplicate_example():
    st.session_state["_prefill_q1"] = "What is the capital of France?"
    st.session_state["_prefill_q2"] = "How do I cook pasta?"

with st.expander("Try example pairs"):
    col1, col2 = st.columns(2)
    with col1:
        st.button("Duplicate: Python learning", on_click=_set_duplicate_example)
    with col2:
        st.button("Not duplicate: Different topics", on_click=_set_not_duplicate_example)

if st.button("Check", type="primary"):
    q1_clean = (q1 or "").strip()
    q2_clean = (q2 or "").strip()

    if not q1_clean or not q2_clean:
        st.warning("Please enter both questions.")
    elif len(q1_clean) < 3 or len(q2_clean) < 3:
        st.warning("Questions should be at least 3 characters.")
    else:
        try:
            pred, proba = helper.predict(q1_clean, q2_clean, selected_model)

            st.metric("Probability of Duplicate", f"{proba:.1%}")

            if pred:
                st.success("**Duplicate** — These questions likely have the same meaning.")
            else:
                st.info("**Not Duplicate** — These questions appear to be different.")

            st.progress(float(proba))
        except Exception as e:
            st.error(f"Error: {str(e)}")

st.divider()
with st.expander("About"):
    st.markdown("""
    This app predicts whether two Quora questions are duplicates (same meaning).

    **Models:**
    - **Classical**: Random Forest or XGBoost on 25 handcrafted features + TF-IDF
    - **DistilBERT**: Fine-tuned transformer for sentence-pair classification

    Run `python scripts/06_benchmark_inference.py` to measure inference times.
    """)
