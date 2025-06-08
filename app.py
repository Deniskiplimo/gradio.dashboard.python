import gradio as gr
import pandas as pd
import plotly.express as px
import base64
import io
import os
import socket
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

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def start_fastapi():
    if not is_port_in_use(8000):
        threading.Thread(target=lambda: uvicorn.run(api, host="0.0.0.0", port=8000, log_level="error"), daemon=True).start()

start_fastapi()

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
        save_summary(name, summary_text + "\n\n" + notes_text)
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
# GRADIO INTERFACE
# ==========================
with gr.Blocks(title="📊 Smart Data Dashboard") as demo:
    gr.Markdown("## 📊 Interactive Data Dashboard with Voice Summary & API Access")

    with gr.Row():
        dataset_selector = gr.Dropdown(choices=["Iris", "Wine", "Diabetes", "Upload File"], label="Select Dataset")
        file_input = gr.File(label="Upload CSV/Excel/JSON/TSV", file_types=[".csv", ".xlsx", ".xls", ".json", ".tsv", ".parquet", ".feather"])
        url_input = gr.Textbox(label="Or Enter URL to Dataset")

    with gr.Row():
        preview_btn = gr.Button("🔍 Preview")
        summary_btn = gr.Button("📄 Summary")
        voice_toggle = gr.Checkbox(label="🔈 Voice Summary")

    with gr.Row():
        preview_output = gr.Dataframe(label="Dataset Preview")
        summary_output = gr.Textbox(lines=20, label="Data Summary")
    
    voice_output = gr.Audio(label="Summary Audio")

    preview_btn.click(fn=preview_data, inputs=[dataset_selector, file_input, url_input], outputs=preview_output)
    summary_btn.click(fn=data_summary, inputs=[dataset_selector, file_input, url_input], outputs=summary_output)
    summary_btn.click(fn=voice_summary, inputs=[summary_output, voice_toggle], outputs=voice_output)



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
        b = content.encode("utf-8")
        return b, f"report.{format}"
    else:
        b = content.encode("utf-8")
        return b, f"report.{format}"

# ==========================
# GRADIO INTERFACE
# ==========================
dataset_choices = ["Iris", "Wine", "Diabetes", "Upload File"]

def update_columns(name, file, url):
    df, err = load_dataset(name, file, url)
    if err or df is None or df.empty:
        return [], [], err or "No data loaded"
    cols = df.columns.tolist()
    num_cols = df.select_dtypes(include='number').columns.tolist()
    return cols, cols, ""

with gr.Blocks() as demo:
    gr.Markdown("# Interactive Data Dashboard")

    with gr.Row():
        dataset = gr.Dropdown(label="Select Dataset", choices=dataset_choices, value="Iris")
        upload_file = gr.File(label="Upload CSV/Excel/JSON file", file_types=['.csv', '.xlsx', '.xls', '.json', '.tsv', '.parquet', '.feather'])
        url_input = gr.Textbox(label="Or enter URL for CSV/JSON data (optional)")

    preview_btn = gr.Button("Preview Data")
    preview_output = gr.Dataframe(headers=None, datatype=["str"], interactive=False, label="Data Preview")

    summary_btn = gr.Button("Generate Summary")
    summary_output = gr.Textbox(label="Data Summary", lines=15)

    voice_toggle = gr.Checkbox(label="Enable Voice Output for Summary", value=False)
    audio_output = gr.Audio(label="Summary Audio", interactive=False)

    with gr.Tab("Scatter Plot"):
        scatter_x = gr.Dropdown(label="X-axis Column", choices=[])
        scatter_y = gr.Dropdown(label="Y-axis Column", choices=[])
        plot_scatter_btn = gr.Button("Plot Scatter")
        scatter_plot = gr.Plot()

    with gr.Tab("Histograms"):
        histograms_gallery = gr.Gallery(label="Histograms", elem_id="histograms_gallery")

    with gr.Tab("Box Plots"):
        boxplots_gallery = gr.Gallery(label="Box Plots", elem_id="boxplots_gallery")

    with gr.Tab("Correlation Heatmap"):
        corr_heatmap_plot = gr.Plot()

    with gr.Tab("Scatter Matrix"):
        scatter_matrix_plot = gr.Plot()

    with gr.Tab("Download Report"):
        report_format = gr.Radio(choices=["txt", "csv"], label="Select Report Format", value="txt")
        download_btn = gr.Button("Download Report")
        download_output = gr.File(label="Download your report here")

    # Callbacks
    def preview_callback(name, file, url):
        return preview_data(name, file, url)

    def summary_callback(name, file, url):
        return data_summary(name, file, url)

    def voice_callback(text, enabled):
        if enabled:
            audio = generate_voice(text)
            if audio:
                mime, data = audio
                return (mime, data)
        return None

    def update_scatter_columns(name, file, url):
        cols, _, err = update_columns(name, file, url)
        return cols, cols

    def scatter_plot_callback(name, file, url, x_col, y_col):
        fig, err = generate_scatter_plot(name, file, url, x_col, y_col)
        if fig:
            return fig
        else:
            return None

    def histograms_callback(name, file, url):
        return generate_histograms(name, file, url)

    def boxplots_callback(name, file, url):
        return generate_box_plots(name, file, url)

    def corr_heatmap_callback(name, file, url):
        return generate_correlation_heatmap(name, file, url)

    def scatter_matrix_callback(name, file, url):
        return generate_scatter_matrix(name, file, url)

    def download_callback(name, file, url, fmt):
        content, filename = download_report(name, file, url, fmt)
        if content is None:
            return None
        # Save file temporarily for download
        path = f"/tmp/{filename}"
        with open(path, "wb") as f:
            f.write(content)
        return path

    # Wiring inputs and outputs
    preview_btn.click(preview_callback, inputs=[dataset, upload_file, url_input], outputs=preview_output)
    summary_btn.click(summary_callback, inputs=[dataset, upload_file, url_input], outputs=summary_output)
    summary_btn.click(voice_callback, inputs=[summary_output, voice_toggle], outputs=audio_output)

    dataset.change(update_scatter_columns, inputs=[dataset, upload_file, url_input], outputs=[scatter_x, scatter_y])
    upload_file.change(update_scatter_columns, inputs=[dataset, upload_file, url_input], outputs=[scatter_x, scatter_y])
    url_input.change(update_scatter_columns, inputs=[dataset, upload_file, url_input], outputs=[scatter_x, scatter_y])

    plot_scatter_btn.click(scatter_plot_callback, inputs=[dataset, upload_file, url_input, scatter_x, scatter_y], outputs=scatter_plot)
    plot_scatter_btn.click(histograms_callback, inputs=[dataset, upload_file, url_input], outputs=histograms_gallery)
    plot_scatter_btn.click(boxplots_callback, inputs=[dataset, upload_file, url_input], outputs=boxplots_gallery)
    plot_scatter_btn.click(corr_heatmap_callback, inputs=[dataset, upload_file, url_input], outputs=corr_heatmap_plot)
    plot_scatter_btn.click(scatter_matrix_callback, inputs=[dataset, upload_file, url_input], outputs=scatter_matrix_plot)

    download_btn.click(download_callback, inputs=[dataset, upload_file, url_input, report_format], outputs=download_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

