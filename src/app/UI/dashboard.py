import io
import os
import sys

import altair as alt
import pandas as pd
import requests
import streamlit as st
from PIL import Image

# Ensure the project root is importable when Streamlit runs the file directly.
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.append(root_dir)

try:
    from src.app.DB.database import get_all_sightings
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()


st.set_page_config(
    page_title="Wildlife AI Monitor",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_styles(theme_mode):
    is_dark = theme_mode == "Dark"
    bg = "#0f1418" if is_dark else "#f4efe6"
    bg_end = "#161d22" if is_dark else "#f7f2e9"
    panel = "rgba(24, 31, 36, 0.82)" if is_dark else "rgba(255, 250, 241, 0.82)"
    line = "rgba(214, 223, 228, 0.12)" if is_dark else "rgba(73, 59, 34, 0.12)"
    text = "#edf3f6" if is_dark else "#2f271b"
    muted = "#9fb0b8" if is_dark else "#766a55"
    accent = "#7ca44d" if is_dark else "#6f8a3a"
    accent_deep = "#b7d47a" if is_dark else "#465a24"
    shadow = "rgba(0, 0, 0, 0.28)" if is_dark else "rgba(82, 62, 27, 0.08)"
    grad_a = "rgba(124, 164, 77, 0.18)" if is_dark else "rgba(111, 138, 58, 0.16)"
    grad_b = "rgba(181, 128, 67, 0.16)" if is_dark else "rgba(182, 145, 73, 0.18)"

    st.markdown(
        f"""
        <style>
            :root {{
                --bg: {bg};
                --panel: {panel};
                --line: {line};
                --text: {text};
                --muted: {muted};
                --accent: {accent};
                --accent-deep: {accent_deep};
            }}

            .stApp {{
                background:
                    radial-gradient(circle at top right, {grad_b}, transparent 26%),
                    radial-gradient(circle at top left, {grad_a}, transparent 28%),
                    linear-gradient(180deg, {bg_end} 0%, var(--bg) 100%);
                color: var(--text);
            }}

            header[data-testid="stHeader"] {{
                display: none;
            }}

            div[data-testid="stToolbar"] {{
                display: none;
            }}

            div[data-testid="stDecoration"] {{
                display: none;
            }}

            #MainMenu {{
                display: none;
            }}

            .block-container {{
                padding-top: 1.2rem;
                padding-bottom: 3rem;
            }}

            .hero-card,
            .soft-card,
            .metric-card {{
                background: var(--panel);
                border: 1px solid var(--line);
                border-radius: 24px;
                box-shadow: 0 18px 40px {shadow};
                backdrop-filter: blur(8px);
            }}

            .hero-card {{
                padding: 2rem 2rem 1.6rem 2rem;
                margin-bottom: 1.25rem;
            }}

            .hero-kicker {{
                display: inline-block;
                padding: 0.35rem 0.75rem;
                border-radius: 999px;
                background: rgba(111, 138, 58, 0.12);
                color: var(--accent-deep);
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.04em;
                text-transform: uppercase;
            }}

            .hero-title {{
                font-size: 3rem;
                line-height: 1.02;
                font-weight: 800;
                margin: 0.9rem 0 0.8rem 0;
                color: var(--text);
            }}

            .hero-text {{
                max-width: 54rem;
                font-size: 1.08rem;
                line-height: 1.6;
                color: var(--muted);
                margin: 0;
            }}

            .section-title {{
                font-size: 1.45rem;
                font-weight: 800;
                color: var(--text);
                margin: 0 0 0.35rem 0;
            }}

            .section-copy {{
                color: var(--muted);
                margin: 0 0 1rem 0;
            }}

            .soft-card {{
                padding: 1.15rem 1.25rem;
                margin-bottom: 1rem;
            }}

            .soft-card h4 {{
                margin: 0 0 0.35rem 0;
                font-size: 1rem;
                color: var(--text);
            }}

            .soft-card p,
            .soft-card li {{
                color: var(--muted);
                margin-bottom: 0;
            }}

            .metric-card {{
                padding: 1rem 1.1rem;
                min-height: 120px;
            }}

            .metric-label {{
                color: var(--muted);
                font-size: 0.82rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-bottom: 0.55rem;
            }}

            .metric-value {{
                font-size: 1.6rem;
                font-weight: 800;
                color: var(--text);
                line-height: 1.1;
            }}

            .metric-note {{
                color: var(--muted);
                font-size: 0.92rem;
                margin-top: 0.55rem;
            }}

            .subtle-label {{
                color: var(--muted);
                font-size: 0.85rem;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                margin-bottom: 0.4rem;
            }}

            .stButton > button {{
                border-radius: 999px;
                border: 1px solid transparent;
                background: linear-gradient(135deg, var(--accent) 0%, var(--accent-deep) 100%);
                color: {"#112017" if is_dark else "white"};
                font-weight: 700;
                min-height: 2.9rem;
                padding: 0.2rem 1.1rem;
            }}

            .stRadio > div {{
                background: var(--panel);
                border: 1px solid var(--line);
                border-radius: 18px;
                padding: 0.3rem 0.7rem;
            }}

            .stRadio [role="radiogroup"] label,
            .stRadio [role="radiogroup"] p,
            .stRadio [data-baseweb="radio"] div {{
                color: var(--text) !important;
            }}

            .stRadio label,
            .stMarkdown,
            .stCaption,
            .stText {{
                color: var(--text);
            }}

            .stFileUploader {{
                background: var(--panel);
                border: 1px dashed {line};
                border-radius: 20px;
                padding: 0.65rem;
            }}

            div[data-testid="stFileUploader"] small,
            div[data-testid="stFileUploader"] span,
            div[data-testid="stFileUploader"] label {{
                color: var(--muted);
            }}

            div[data-testid="stImage"] img {{
                border-radius: 22px;
                border: 1px solid var(--line);
            }}

            [data-testid="stMetricValue"] {{
                color: var(--text);
            }}

            [data-testid="stMetricLabel"],
            [data-testid="stMetricDelta"] {{
                color: var(--muted);
            }}

            [data-baseweb="tab-list"] {{
                gap: 0.45rem;
            }}

            button[data-baseweb="tab"] {{
                background: var(--panel);
                border: 1px solid var(--line);
                border-radius: 999px;
                color: var(--muted);
                padding: 0.5rem 1rem;
            }}

            button[data-baseweb="tab"][aria-selected="true"] {{
                color: var(--text);
                border-color: rgba(124, 164, 77, 0.4);
                box-shadow: inset 0 0 0 1px rgba(124, 164, 77, 0.18);
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def latest_sighting():
    rows = get_all_sightings()
    if not rows:
        return None

    row = rows[0]
    return {
        "id": row[0],
        "filename": row[1],
        "species": row[2],
        "quantity": row[3],
        "confidence": row[4],
        "condition": row[5],
        "timestamp": row[6],
    }


def render_metric_card(label, value, note):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_history(df):
    st.markdown('<div class="section-title">Recent Sightings</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">A quick operational view of the latest records captured by the monitor.</p>',
        unsafe_allow_html=True,
    )

    history_df = df.rename(
        columns={
            "filename": "File",
            "species": "Species",
            "quantity": "Count",
            "confidence": "Confidence",
            "condition": "Health",
            "timestamp": "Captured At",
        }
    ).copy()
    history_df["Confidence"] = history_df["Confidence"].apply(
        lambda x: f"{float(x):.1%}" if pd.notna(x) else "N/A"
    )
    st.dataframe(history_df, hide_index=True, use_container_width=True)


def render_statistics(df):
    species_colors = ["#8db45a", "#d29a55", "#a56a46", "#64843b", "#d7bd8c"]
    hourly_color = "#d29a55"

    st.markdown('<div class="section-title">Field Activity</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">Current data distribution by species and by time of day.</p>',
        unsafe_allow_html=True,
    )

    total_animals = df.groupby("species", dropna=False)["quantity"].sum().reset_index()
    total_animals.columns = ["species", "total_count"]
    total_animals["species"] = total_animals["species"].fillna("Unknown")

    activity_df = df.copy()
    activity_df["timestamp"] = pd.to_datetime(activity_df["timestamp"])
    activity_df["hour"] = activity_df["timestamp"].dt.hour
    hourly = activity_df["hour"].value_counts().reset_index()
    hourly.columns = ["hour", "count"]
    hourly = hourly.sort_values("hour")

    stats_col1, stats_col2 = st.columns(2)

    with stats_col1:
        pie = (
            alt.Chart(total_animals)
            .mark_arc(outerRadius=112, innerRadius=42)
            .encode(
                theta=alt.Theta("total_count:Q", stack=True),
                color=alt.Color(
                    "species:N",
                    scale=alt.Scale(range=species_colors),
                ),
                tooltip=["species", "total_count"],
            )
            .properties(height=320)
        )
        st.altair_chart(pie, width="stretch")

    with stats_col2:
        bar = (
            alt.Chart(hourly)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color=hourly_color)
            .encode(
                x=alt.X("hour:O", title="Hour", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("count:Q", title="Sightings"),
                tooltip=["hour", "count"],
            )
            .properties(height=320)
        )
        st.altair_chart(bar, width="stretch")


if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "last_record" not in st.session_state:
    st.session_state["last_record"] = None
if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "Dark"

theme_bar_col, theme_toggle_col = st.columns([1, 0.34])
with theme_bar_col:
    st.markdown(
        '<div class="subtle-label">Appearance</div>',
        unsafe_allow_html=True,
    )
with theme_toggle_col:
    st.session_state["theme_mode"] = st.radio(
        "Theme",
        ["Dark", "Light"],
        index=0 if st.session_state["theme_mode"] == "Dark" else 1,
        horizontal=True,
        label_visibility="collapsed",
    )

inject_styles(st.session_state["theme_mode"])


st.markdown(
    """
    <section class="hero-card">
        <div class="hero-kicker">Wildlife Monitoring Interface</div>
        <h1 class="hero-title">Wildlife AI Monitor</h1>
        <p class="hero-text">
            Monitor wildlife images, review detections, and track sightings in one place.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

all_rows = get_all_sightings()
latest_record = st.session_state["last_record"] or latest_sighting()
total_sightings = len(all_rows)
unique_species = len({row[2] for row in all_rows if row[2]})

summary_col1, summary_col2, summary_col3 = st.columns(3)
with summary_col1:
    st.metric("Sightings Logged", total_sightings)
with summary_col2:
    st.metric("Species Seen", unique_species)
with summary_col3:
    st.metric("Latest Health Status", latest_record["condition"] if latest_record else "No data")

analysis_tab, history_tab, stats_tab, ai_tab = st.tabs(
    ["Image Analysis", "History", "Statistics", "AI Status"]
)

with analysis_tab:
    main_col, side_col = st.columns([1.35, 0.9], gap="large")

    with main_col:
        st.markdown('<div class="section-title">Image Analysis</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-copy">Upload a JPG or PNG from the field and run detection through the local API.</p>',
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Upload wildlife image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key="analysis_uploader",
        )

        preview_col, action_col = st.columns([1.15, 0.85], gap="large")

        with preview_col:
            if uploaded_file is not None:
                preview_image = Image.open(uploaded_file)
                st.markdown('<div class="subtle-label">Selected Image</div>', unsafe_allow_html=True)
                st.image(preview_image, caption=uploaded_file.name, width="stretch")
            else:
                st.markdown(
                    """
                    <div class="soft-card">
                        <h4>Drop in a camera-trap image</h4>
                        <p>This tab is the main workflow. Upload, analyze, and review the latest detection without leaving the page.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with action_col:
            st.markdown(
                """
                <div class="soft-card">
                    <h4>Detection Flow</h4>
                    <p>1. Send image to the local API.</p>
                    <p>2. Receive an annotated wildlife frame.</p>
                    <p>3. Pull the latest saved sighting and show it as summary cards.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if latest_record:
                st.markdown(
                    f"""
                    <div class="soft-card">
                        <h4>Latest Record</h4>
                        <p><strong>Species:</strong> {latest_record['species']}</p>
                        <p><strong>Health:</strong> {latest_record['condition']}</p>
                        <p><strong>Captured:</strong> {latest_record['timestamp']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if uploaded_file is not None:
                analyze = st.button(
                    "Analyze Image",
                    type="primary",
                    use_container_width=True,
                    key="analyze_button",
                )
                if analyze:
                    with st.spinner("Running wildlife detection..."):
                        img_byte_arr = io.BytesIO()
                        preview_image.save(img_byte_arr, format=preview_image.format or "PNG")
                        files = {"file": (uploaded_file.name, img_byte_arr.getvalue(), "image/jpeg")}

                        try:
                            response = requests.post(
                                "http://127.0.0.1:8000/predict",
                                files=files,
                                timeout=90,
                            )
                            if response.status_code == 200:
                                st.session_state["last_result"] = response.content
                                st.session_state["last_record"] = latest_sighting()
                                latest_record = st.session_state["last_record"]
                                st.success("Analysis completed.")
                            else:
                                st.error(f"API returned status code {response.status_code}.")
                        except requests.RequestException as e:
                            st.error(f"Could not reach the detection API: {e}")

    with side_col:
        st.markdown(
            """
            <div class="soft-card">
                <h4>Tab Purpose</h4>
                <p>The analysis tab stays focused on the primary user action: upload an image and review the latest result clearly.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if all_rows:
            st.markdown(
                """
                <div class="soft-card">
                    <h4>System Status</h4>
                    <p>History and statistics are separated into their own tabs so this area can stay focused on the current analysis session.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="soft-card">
                    <h4>No records yet</h4>
                    <p>Once detections are saved, the other tabs will become more useful for browsing history and trends.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if st.session_state["last_result"] is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Latest Analysis Result</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-copy">The annotated image is shown alongside the latest database-backed summary.</p>',
            unsafe_allow_html=True,
        )

        result_col, details_col = st.columns([1.35, 1], gap="large")

        with result_col:
            st.image(st.session_state["last_result"], caption="Annotated detection", width="stretch")

        with details_col:
            record = st.session_state["last_record"]
            if record:
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    render_metric_card("Species", record["species"] or "Unknown", "Primary detected animal")
                with metric_col2:
                    render_metric_card("Count", record["quantity"], "Animals counted in this sighting")

                metric_col3, metric_col4 = st.columns(2)
                with metric_col3:
                    confidence = (
                        f"{float(record['confidence']):.1%}" if record["confidence"] is not None else "N/A"
                    )
                    render_metric_card("Confidence", confidence, "Model confidence score")
                with metric_col4:
                    render_metric_card("Health", record["condition"] or "Unknown", "Latest health classification")

                st.markdown(
                    f"""
                    <div class="soft-card">
                        <h4>Detection Context</h4>
                        <p><strong>File:</strong> {record['filename']}</p>
                        <p><strong>Recorded at:</strong> {record['timestamp']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info(
                    "The image result is available, but there is no matching database record to summarize."
                )

with history_tab:
    st.markdown('<div class="section-title">Sightings History</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">Browse the latest saved detections in a dedicated tab instead of sharing space with the upload workflow.</p>',
        unsafe_allow_html=True,
    )

    if all_rows:
        df = pd.DataFrame(
            all_rows,
            columns=["id", "filename", "species", "quantity", "confidence", "condition", "timestamp"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        render_history(df)
    else:
        st.markdown(
            """
            <div class="soft-card">
                <h4>No history available yet</h4>
                <p>Analyze an image first to start building the sightings log.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

with stats_tab:
    st.markdown('<div class="section-title">Statistics</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">Use this tab for the broader field picture: species distribution and activity over time.</p>',
        unsafe_allow_html=True,
    )

    if all_rows:
        df = pd.DataFrame(
            all_rows,
            columns=["id", "filename", "species", "quantity", "confidence", "condition", "timestamp"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        render_statistics(df)
    else:
        st.markdown(
            """
            <div class="soft-card">
                <h4>No statistics available yet</h4>
                <p>Once sightings are logged, this tab will visualize the main trends.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

with ai_tab:
    st.markdown('<div class="section-title">AI Status</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">This area remains separated from your UI scope and is intentionally left as a status panel only.</p>',
        unsafe_allow_html=True,
    )
    st.info(
        "The AI chat feature is intentionally out of scope for this UI pass. The redesigned dashboard keeps it isolated so it does not interfere with the main user flow."
    )
