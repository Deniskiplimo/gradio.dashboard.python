import gradio as gr
import pandas as pd
import plotly.express as px
import base64
import io
import os
from sklearn import datasets
import requests
from io import StringIO
import sqlite3
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import threading
import pyttsx3  # For voice output (TTS)

DB_NAME = "dashboard.db"

# ==========================
# DATABASE SETUP
# ==========================
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dataset_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_name TEXT,
            summary TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_summary(dataset_name, summary):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO dataset_summary (dataset_name, summary) VALUES (?, ?)",
        (dataset_name, summary)
    )
    conn.commit()
    conn.close()

def fetch_summaries():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id, dataset_name, summary, created_at FROM dataset_summary ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows

init_db()

# ==========================
# FASTAPI BACKEND
# ==========================
api = FastAPI(title="Data Dashboard API")
api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@api.get("/api/summaries")
def get_summaries():
    return JSONResponse(content={"summaries": fetch_summaries()})

def start_api():
    uvicorn.run(api, host="127.0.0.1", port=8000)

threading.Thread(target=start_api, daemon=True).start()

# ==========================
# DATA LOADING & PROCESSING
# ==========================
def load_dataset(name, file=None, url=None):
    try:
        if name == "Upload File" and file is not None:
            ext = os.path.splitext(file.name)[-1].lower()
            if ext == ".csv":
                df = pd.read_csv(file)
            elif ext in [".xlsx", ".xls"]:
                df = pd.read_excel(file)
            elif ext == ".json":
                df = pd.read_json(file)
            elif ext == ".tsv":
                df = pd.read_csv(file, sep="\t")
            elif ext == ".parquet":
                df = pd.read_parquet(file)
            elif ext == ".feather":
                df = pd.read_feather(file)
            else:
                return None, f"Unsupported file format: {ext}"
            return df, None

        if url:
            resp = requests.get(url)
            resp.raise_for_status()
            content_type = resp.headers.get('content-type', '')
            if 'csv' in content_type or url.endswith('.csv'):
                df = pd.read_csv(StringIO(resp.text))
            elif 'json' in content_type or url.endswith('.json'):
                df = pd.read_json(StringIO(resp.text))
            else:
                return None, "Unsupported URL content type or extension."
            return df, None

        if name == "Iris":
            df = datasets.load_iris(as_frame=True).frame
        elif name == "Wine":
            df = datasets.load_wine(as_frame=True).frame
        elif name == "Diabetes":
            df = datasets.load_diabetes(as_frame=True).frame
        else:
            return None, "Unknown dataset name."
        return df, None

    except Exception as e:
        return None, f"Error loading dataset: {e}"

def preview_data(name, file, url):
    df, err = load_dataset(name, file, url)
    if err or df is None:
        return pd.DataFrame()
    return df.head()

def data_summary(name, file, url):
    df, err = load_dataset(name, file, url)
    if err or df is None or df.empty:
        return "No data loaded."
    try:
        desc = df.describe(include='all').transpose()
        notes = []
        for col in df.columns:
            missing_pct = df[col].isna().mean() * 100
            notes.append(f"- Column '{col}': {missing_pct:.1f}% missing values.")
            if df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col]):
                unique_vals = df[col].nunique(dropna=True)
                notes.append(f"  Unique categories: {unique_vals}")
                top_vals = df[col].value_counts(dropna=True).head(3)
                notes.append(f"  Top categories:\n    {top_vals.to_string()}")
        notes_text = "\n".join(notes)
        summary_text = desc.to_string()
        return f"Summary Statistics:\n{summary_text}\n\nData Notes:\n{notes_text}"
    except Exception as e:
        return f"Error generating summary: {e}"

# ==========================
# VOICE OUTPUT (Text-to-Speech)
# ==========================
def generate_voice(summary_text):
    try:
        engine = pyttsx3.init()
        tmp_file = "summary_voice.wav"
        engine.save_to_file(summary_text, tmp_file)
        engine.runAndWait()
        with open(tmp_file, "rb") as f:
            audio_bytes = f.read()
        os.remove(tmp_file)
        return ("audio/wav", audio_bytes)
    except Exception:
        return None

def voice_summary(text, voice_enabled):
    if voice_enabled and text.strip():
        audio = generate_voice(text)
        if audio:
            return audio
    return None

# ==========================
# PLOTS & VISUALIZATIONS
# ==========================
def generate_scatter_plot(name, file, url, x_col, y_col):
    df, err = load_dataset(name, file, url)
    if err or df is None or df.empty:
        return None, "No data loaded or error."
    if x_col not in df.columns or y_col not in df.columns:
        return None, "Selected columns not found."
    try:
        fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
        return fig, ""
    except Exception as e:
        return None, f"Plot error: {e}"

def generate_histograms(name, file, url):
    df, err = load_dataset(name, file, url)
    if err or df is None or df.empty:
        return []
    num_cols = df.select_dtypes(include='number').columns.tolist()
    return [px.histogram(df, x=col, title=f"Histogram of {col}") for col in num_cols]

def generate_box_plots(name, file, url):
    df, err = load_dataset(name, file, url)
    if err or df is None or df.empty:
        return []
    num_cols = df.select_dtypes(include='number').columns.tolist()
    return [px.box(df, y=col, title=f"Box Plot of {col}") for col in num_cols]

def generate_correlation_heatmap(name, file, url):
    df, err = load_dataset(name, file, url)
    if err or df is None or df.empty:
        return None
    corr = df.select_dtypes(include='number').corr()
    fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
    return fig

def generate_scatter_matrix(name, file, url):
    df, err = load_dataset(name, file, url)
    if err or df is None or df.empty:
        return None
    num_df = df.select_dtypes(include='number')
    if num_df.shape[1] < 2:
        return None
    fig = px.scatter_matrix(num_df, title="Scatter Matrix")
    return fig

# ==========================
# REPORT GENERATION & DOWNLOAD
# ==========================
def generate_report(name, file, url, format):
    df, err = load_dataset(name, file, url)
    if err or df is None or df.empty:
        return "No data to generate report."

    meta = f"Dataset: {name}\nRows: {df.shape[0]}\nColumns: {df.shape[1]}"
    desc = df.describe(include='all').transpose()
    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    cat_summary = []
    for col in df.select_dtypes(include=['object', 'category']).columns:
        top_vals = df[col].value_counts(dropna=True).head(5)
        cat_summary.append(f"Top categories for '{col}':\n{top_vals.to_string()}\n")

    if format == "txt":
        buffer = io.StringIO()
        buffer.write(meta + "\n\nSummary Statistics:\n")
        buffer.write(desc.to_string())
        buffer.write("\n\nMissing values (%):\n")
        buffer.write(missing_pct.to_string())
        buffer.write("\n\nCategorical summaries:\n")
        buffer.write("\n".join(cat_summary))
        return buffer.getvalue()

    elif format == "csv":
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        return buffer.getvalue()

    elif format == "pdf":
        # PDF generation requires reportlab or fpdf (not included here to keep dependencies minimal)
        return "PDF report generation not implemented."

    else:
        return "Unsupported report format."

def download_report(name, file, url, format):
    content = generate_report(name, file, url, format)
    if content.startswith("No data") or content.startswith("Unsupported") or content.startswith("PDF"):
        return None, content

    if format == "csv":
        b = content.encode()
        b64 = base64.b64encode(b).decode()
        return f"data:text/csv;base64,{b64}", None

    if format == "txt":
        b = content.encode()
        b64 = base64.b64encode(b).decode()
        return f"data:text/plain;base64,{b64}", None

    # PDF not supported here, return error message
    return None, content

# ==========================
# UI CALLBACK HELPERS
# ==========================
def update_columns_for_scatter(name, file, url):
    df, err = load_dataset(name, file, url)
    if err or df is None or df.empty:
        return gr.Dropdown.update(choices=[]), gr.Dropdown.update(choices=[])
    cols = df.columns.tolist()
    return gr.Dropdown.update(choices=cols, value=cols[0] if cols else None), gr.Dropdown.update(choices=cols, value=cols[1] if len(cols) > 1 else cols[0])

def toggle_upload_mode(mode):
    # mode is True (URL) or False (file upload)
    return (gr.Textbox.update(visible=mode), gr.File.update(visible=not mode))

# ==========================
# GRADIO UI LAYOUT
# ==========================
with gr.Blocks(title="Modern Data Dashboard") as demo:
    gr.Markdown("# Modern Data Dashboard")

    with gr.Row():
        with gr.Column(scale=2):
            dataset_selector = gr.Dropdown(label="Select Dataset",
                                          choices=["Iris", "Wine", "Diabetes", "Upload File"],
                                          value="Iris")

            toggle_upload_checkbox = gr.Checkbox(label="Use URL Input Instead of File Upload?", value=False)

            url_input = gr.Textbox(label="Enter URL (CSV/JSON)", visible=False)
            file_input = gr.File(label="Upload Dataset File", file_types=['.csv', '.xlsx', '.xls', '.json', '.tsv', '.parquet', '.feather'])

            preview_button = gr.Button("Preview Data")
            summary_button = gr.Button("Generate Summary")
            voice_toggle = gr.Checkbox(label="Enable Voice Output for Summary", value=False)

            scatter_x = gr.Dropdown(label="Scatter Plot X Axis")
            scatter_y = gr.Dropdown(label="Scatter Plot Y Axis")
            scatter_button = gr.Button("Generate Scatter Plot")
            scatter_plot = gr.Plot()

            histogram_button = gr.Button("Generate Histograms")
            histogram_gallery = gr.Gallery(label="Histograms")


            boxplot_button = gr.Button("Generate Box Plots")
            boxplot_gallery = gr.Gallery(label="Box Plots")

            corr_heatmap_button = gr.Button("Generate Correlation Heatmap")
            corr_heatmap_plot = gr.Plot()

            scatter_matrix_button = gr.Button("Generate Scatter Matrix")
            scatter_matrix_plot = gr.Plot()

            report_format = gr.Radio(label="Report Format", choices=["txt", "csv"], value="txt")
            generate_report_button = gr.Button("Generate & Download Report")
            download_link = gr.Markdown()

        with gr.Column(scale=3):
            data_preview = gr.DataFrame(label="Data Preview", interactive=False)
            data_summary_text = gr.Textbox(label="Data Summary", lines=20, interactive=False)
            voice_output = gr.Audio(label="Voice Summary", interactive=False)

    # ========== CALLBACKS ===========

    # Toggle URL/file inputs
    toggle_upload_checkbox.change(
        fn=toggle_upload_mode,
        inputs=[toggle_upload_checkbox],
        outputs=[url_input, file_input]
    )

    # Update scatter dropdowns on dataset or file/url change
    def update_scatter_cols(name, file, url):
        return update_columns_for_scatter(name, file, url)
    dataset_selector.change(update_scatter_cols, inputs=[dataset_selector, file_input, url_input], outputs=[scatter_x, scatter_y])
    file_input.change(update_scatter_cols, inputs=[dataset_selector, file_input, url_input], outputs=[scatter_x, scatter_y])
    url_input.change(update_scatter_cols, inputs=[dataset_selector, file_input, url_input], outputs=[scatter_x, scatter_y])

    # Preview data
    preview_button.click(preview_data, inputs=[dataset_selector, file_input, url_input], outputs=data_preview)

    # Generate summary + voice
    def summary_and_voice(name, file, url, voice_on):
        summary = data_summary(name, file, url)
        audio = voice_summary(summary, voice_on)
        return summary, audio
    summary_button.click(
        summary_and_voice,
        inputs=[dataset_selector, file_input, url_input, voice_toggle],
        outputs=[data_summary_text, voice_output]
    )

    # Scatter plot
    def scatter_click(name, file, url, x_col, y_col):
        fig, err = generate_scatter_plot(name, file, url, x_col, y_col)
        if err:
            return None
        return fig
    scatter_button.click(scatter_click, inputs=[dataset_selector, file_input, url_input, scatter_x, scatter_y], outputs=scatter_plot)

    # Histograms
    def histograms_click(name, file, url):
        figs = generate_histograms(name, file, url)
        return figs
    histogram_button.click(histograms_click, inputs=[dataset_selector, file_input, url_input], outputs=histogram_gallery)

    # Box plots
    def boxplots_click(name, file, url):
        figs = generate_box_plots(name, file, url)
        return figs
    boxplot_button.click(boxplots_click, inputs=[dataset_selector, file_input, url_input], outputs=boxplot_gallery)

    # Correlation heatmap
    corr_heatmap_button.click(
        lambda name, file, url: generate_correlation_heatmap(name, file, url),
        inputs=[dataset_selector, file_input, url_input],
        outputs=corr_heatmap_plot,
    )

    # Scatter matrix
    scatter_matrix_button.click(
        lambda name, file, url: generate_scatter_matrix(name, file, url),
        inputs=[dataset_selector, file_input, url_input],
        outputs=scatter_matrix_plot,
    )

    # Generate & Download report link
    def get_report_download_link(name, file, url, fmt):
        data_url, err_msg = download_report(name, file, url, fmt)
        if data_url:
            link = f"[Download {fmt.upper()} Report]({data_url})"
            return link
        return f"Error: {err_msg}"
    generate_report_button.click(
        get_report_download_link,
        inputs=[dataset_selector, file_input, url_input, report_format],
        outputs=download_link,
    )

demo.launch()
