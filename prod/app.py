# ==============================================================================
# app.py - Steam Review Classifier (Streamlit Web App)
# ==============================================================================
# Two modes:
#   1. Manual Input: paste/type a review and get a sentiment prediction.
#   2. Steam App Analysis: enter a Steam App ID, fetch real reviews from the
#      Steam API, classify them with the fine-tuned BERT model, and display
#      an overall recommendation ratio with sample predictions.
#
# Run locally: streamlit run app.py
# ==============================================================================

import streamlit as st
from utils import load_model, predict_sentiment, fetch_steam_reviews, analyze_reviews

# ---- Page config ----
st.set_page_config(
    page_title="Steam Review Classifier",
    page_icon="🎮",
    layout="centered",
)

# ---- Load model (cached so it's only loaded once) ----
@st.cache_resource
def get_model():
    model, tokenizer, device = load_model("model_files")
    return model, tokenizer, device


st.title("🎮 Steam Review Classifier")
st.caption(
    "Sentiment analysis of video game reviews powered by a fine-tuned BERT model."
)

# Load model
with st.spinner("Loading BERT model..."):
    model, tokenizer, device = get_model()

st.divider()

# ---- Tabs ----
tab_manual, tab_steam = st.tabs(["✍️ Manual Input", "🔍 Analyze by App ID"])

# ==========================================================================
# TAB 1 — Manual text input
# ==========================================================================
with tab_manual:
    st.subheader("Classify a single review")
    st.markdown(
        "Type or paste a video game review and the model will predict whether "
        "the opinion is **positive** or **negative**."
    )

    review_text = st.text_area(
        "Write a review:",
        placeholder="e.g. This game is amazing! Great graphics, smooth gameplay...",
        height=150,
        key="manual_input",
    )

    if st.button("🔍 Classify", type="primary", key="classify_btn"):
        text = st.session_state.get("manual_input", "")
        if not text or len(text.strip()) < 10:
            st.warning("Please enter a longer review to classify.")
        else:
            with st.spinner("Analyzing..."):
                label, confidence, cleaned = predict_sentiment(
                    text, model, tokenizer, device
                )

            st.divider()

            if "Positive" in label:
                st.success(f"**Prediction: {label} 👍**")
            elif "Negative" in label:
                st.error(f"**Prediction: {label} 👎**")
            else:
                st.warning(f"**Result: {label}**")

            st.metric("Confidence", f"{confidence:.1%}")
            st.progress(confidence)

            with st.expander("View preprocessed text"):
                st.text(cleaned)

# ==========================================================================
# TAB 2 — Analyze by Steam App ID
# ==========================================================================
with tab_steam:
    st.subheader("Analyze reviews from a Steam game")
    st.markdown(
        "Enter a **Steam App ID** and the app will fetch real reviews from the "
        "Steam Store API, run them through the BERT model, and show you a "
        "recommendation breakdown.\n\n"
        "You can find the App ID in the game's Steam URL:  \n"
        "`https://store.steampowered.com/app/`**1850570**`/Death_Stranding_Directors_Cut/`"
    )

    col_id, col_n = st.columns([2, 1])
    with col_id:
        app_id = st.text_input(
            "Steam App ID:",
            placeholder="e.g. 1850570",
            key="app_id_input",
        )
    with col_n:
        num_reviews = st.slider(
            "Reviews to fetch:",
            min_value=10,
            max_value=100,
            value=30,
            step=10,
            key="num_reviews",
        )

    if st.button("🚀 Analyze Game", type="primary", key="analyze_btn"):
        current_app_id = st.session_state.get("app_id_input", "").strip()
        if not current_app_id or not current_app_id.isdigit():
            st.warning("Please enter a valid numeric App ID.")
        else:
            # Fetch reviews from Steam API
            with st.spinner(f"Fetching reviews for App ID {current_app_id}..."):
                steam_data = fetch_steam_reviews(current_app_id, num_reviews)

            if not steam_data["success"]:
                st.error(f"Error: {steam_data['error']}")
            elif len(steam_data["reviews"]) == 0:
                st.warning(
                    "No reviews found after filtering. The game might have very "
                    "few English reviews, or they may be too short/long."
                )
            else:
                # Show Steam's own summary
                st.divider()
                st.markdown(f"### 📊 Steam Summary")
                meta_cols = st.columns(3)
                with meta_cols[0]:
                    st.metric("Steam Rating", steam_data["review_score_desc"])
                with meta_cols[1]:
                    st.metric("Total Reviews", f"{steam_data['total_reviews']:,}")
                with meta_cols[2]:
                    steam_pos = steam_data["total_positive"]
                    steam_total = steam_data["total_reviews"]
                    steam_pct = (steam_pos / steam_total * 100) if steam_total > 0 else 0
                    st.metric("Steam Positive %", f"{steam_pct:.1f}%")

                # Run BERT on fetched reviews
                with st.spinner("Running BERT model on fetched reviews..."):
                    results, summary = analyze_reviews(
                        steam_data["reviews"], model, tokenizer, device
                    )

                # Model recommendation ratio
                st.divider()
                st.markdown("### 🤖 BERT Model Analysis")
                st.markdown(
                    f"Analyzed **{summary['total_analyzed']}** reviews from the "
                    f"Steam API using our fine-tuned BERT model."
                )

                ratio_cols = st.columns(3)
                with ratio_cols[0]:
                    st.metric(
                        "Positive",
                        f"{summary['positive_count']} ({summary['positive_ratio']:.1f}%)",
                    )
                with ratio_cols[1]:
                    st.metric(
                        "Negative",
                        f"{summary['negative_count']} ({summary['negative_ratio']:.1f}%)",
                    )
                with ratio_cols[2]:
                    # Overall recommendation
                    if summary["positive_ratio"] >= 70:
                        verdict = "✅ Recommended"
                    elif summary["positive_ratio"] >= 40:
                        verdict = "⚠️ Mixed"
                    else:
                        verdict = "❌ Not Recommended"
                    st.metric("Verdict", verdict)

                # Visual bar
                st.progress(summary["positive_ratio"] / 100)
                st.caption(
                    f"{summary['positive_ratio']:.1f}% of the analyzed reviews "
                    f"were classified as positive by the model."
                )

                # Agreement rate between Steam labels and BERT
                agree = sum(
                    1 for r in results
                    if (r["model_label"] == "Positive") == (r["steam_label"] == "Recommended")
                )
                agree_pct = agree / len(results) * 100 if results else 0
                st.info(
                    f"**Agreement with Steam labels:** {agree_pct:.1f}% — "
                    f"the model agreed with Steam's own recommendation on "
                    f"{agree} out of {len(results)} reviews."
                )

                # Sample reviews
                st.divider()
                st.markdown("### 📝 Sample Reviews & Predictions")

                # Show up to 6 reviews (3 positive, 3 negative if possible)
                pos_reviews = [r for r in results if r["model_label"] == "Positive"]
                neg_reviews = [r for r in results if r["model_label"] == "Negative"]

                samples = pos_reviews[:3] + neg_reviews[:3]
                if not samples:
                    samples = results[:6]

                for r in samples:
                    is_pos = r["model_label"] == "Positive"
                    icon = "👍" if is_pos else "👎"
                    color = "green" if is_pos else "red"

                    with st.container(border=True):
                        # Header with prediction
                        header_cols = st.columns([3, 1, 1])
                        with header_cols[0]:
                            st.markdown(
                                f"**{icon} {r['model_label']}** "
                                f"(confidence: {r['confidence']:.0%})"
                            )
                        with header_cols[1]:
                            st.caption(f"Steam: {r['steam_label']}")
                        with header_cols[2]:
                            st.caption(f"🕐 {r['playtime_hours']}h played")

                        # Review text (truncated to 500 chars for display)
                        text_display = r["text"]
                        if len(text_display) > 500:
                            text_display = text_display[:500] + "..."
                        st.markdown(f"*\"{text_display}\"*")

# ---- Footer ----
st.divider()
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.8em;'>"
    "Final Project — Deep Neural Networks<br>"
    "UTN — Facultad Regional Mendoza<br>"
    "Model: BERT (bert-base-uncased) fine-tuned on Steam Reviews Dataset"
    "</div>",
    unsafe_allow_html=True,
)
