import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import base64
import io
import os
from sklearn import datasets
import requests
from io import BytesIO, StringIO
import sqlite3
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import threading
import pyttsx3  # For voice output (TTS)
from datetime import datetime

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
    cursor.execute("INSERT INTO dataset_summary (dataset_name, summary) VALUES (?, ?)", (dataset_name, summary))
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

# Start FastAPI in background thread to run alongside Gradio
threading.Thread(target=start_api, daemon=True).start()

# ==========================
# DATA LOADING & PROCESSING
# ==========================
def load_dataset(name, file=None, url=None):
    """
    Load dataset by name, file upload, or URL.
    Returns: (df: pd.DataFrame or None, error: str or None)
    """
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

def preview_data(name, file):
    df, err = load_dataset(name, file)
    if err or df is None:
        return pd.DataFrame()
    return df.head()

def data_summary(name, file):
    df, err = load_dataset(name, file)
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
        # Save to a temp file
        tmp_file = "summary_voice.mp3"
        engine.save_to_file(summary_text, tmp_file)
        engine.runAndWait()
        with open(tmp_file, "rb") as f:
            audio_bytes = f.read()
        os.remove(tmp_file)
        return ("audio/mp3", audio_bytes)
    except Exception as e:
        return None

# ==========================
# PLOTS & VISUALIZATIONS
# ==========================
def generate_scatter_plot(name, file, x_col, y_col):
    df, err = load_dataset(name, file)
    if err or df is None or df.empty:
        return None, "No data loaded or error."
    if x_col not in df.columns or y_col not in df.columns:
        return None, "Selected columns not found."
    try:
        fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
        return fig, ""
    except Exception as e:
        return None, f"Plot error: {e}"

def generate_histograms(name, file):
    df, err = load_dataset(name, file)
    if err or df is None or df.empty:
        return []
    num_cols = df.select_dtypes(include='number').columns.tolist()
    return [px.histogram(df, x=col, title=f"Histogram of {col}") for col in num_cols]

def generate_box_plots(name, file):
    df, err = load_dataset(name, file)
    if err or df is None or df.empty:
        return []
    num_cols = df.select_dtypes(include='number').columns.tolist()
    return [px.box(df, y=col, title=f"Box Plot of {col}") for col in num_cols]

def generate_correlation_heatmap(name, file):
    df, err = load_dataset(name, file)
    if err or df is None or df.empty:
        return None
    corr = df.select_dtypes(include='number').corr()
    fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
    return fig

def generate_scatter_matrix(name, file):
    df, err = load_dataset(name, file)
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
def generate_report(name, file, format):
    df, err = load_dataset(name, file)
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
        buffer.write(meta + "\n\n")
        buffer.write("Summary Statistics:\n")
        buffer.write(desc.to_string())
        buffer.write("\n\nMissing Values %:\n")
        buffer.write(missing_pct.to_string())
        buffer.write("\n\nCategorical Data Summary:\n")
        buffer.write("\n".join(cat_summary))
        report_content = buffer.getvalue()
        buffer.close()
        b64 = base64.b64encode(report_content.encode()).decode()
        href = f'<a href="data:text/plain;base64,{b64}" download="report.txt">Download TXT Report</a>'
        save_summary(name, report_content)
        return href

    elif format == "csv":
        csv_bytes = df.to_csv(index=False).encode()
        b64 = base64.b64encode(csv_bytes).decode()
        href = f'<a href="data:text/csv;base64,{b64}" download="report.csv">Download CSV Report (Raw Data)</a>'
        save_summary(name, "CSV raw data report")
        return href

    elif format == "xlsx":
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Data', index=False)
            desc.to_excel(writer, sheet_name='Summary')
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="report.xlsx">Download XLSX Report</a>'
        save_summary(name, "XLSX data report")
        return href

    else:
        return "Unsupported report format."

# ==========================
# UI CALLBACKS
# ==========================
def update_columns(name, file):
    df, err = load_dataset(name, file)
    if err or df is None or df.empty:
        return [], []
    cols = df.columns.tolist()
    return gr.Dropdown.update(choices=cols, value=cols[0] if cols else None), gr.Dropdown.update(choices=cols, value=cols[1] if len(cols) > 1 else None)

def update_preview(name, file):
    df, err = load_dataset(name, file)
    if err or df is None or df.empty:
        return pd.DataFrame()
    return df.head()

def update_summary(name, file):
    summary_text = data_summary(name, file)
    save_summary(name, summary_text)
    return summary_text

def voice_summary(text, voice_enabled):
    if voice_enabled:
        audio = generate_voice(text)
        if audio:
            return audio
        else:
            return None
    else:
        return None

def run_scatter(name, file, x_col, y_col):
    fig, err = generate_scatter_plot(name, file, x_col, y_col)
    if err:
        return None, err
    return fig, ""

def run_histograms(name, file):
    figs = generate_histograms(name, file)
    return figs

def run_boxplots(name, file):
    figs = generate_box_plots(name, file)
    return figs

def run_corr_heatmap(name, file):
    fig = generate_correlation_heatmap(name, file)
    return fig

def run_scatter_matrix(name, file):
    fig = generate_scatter_matrix(name, file)
    return fig

# ==========================
# GRADIO INTERFACE
# ==========================
with gr.Blocks() as demo:
    gr.Markdown("# 📊 Modern Data Dashboard")
    gr.Markdown("Load a dataset, explore summary statistics, generate visualizations, and download reports.")

    with gr.Row():
        dataset_choice = gr.Dropdown(
            label="Select Dataset or Upload File",
            choices=["Iris", "Wine", "Diabetes", "Upload File"],
            value="Iris",
            interactive=True,
        )
        upload_file = gr.File(label="Upload your dataset file", visible=False, file_types=['.csv', '.xlsx', '.xls', '.json', '.tsv', '.parquet', '.feather'])
        url_input = gr.Textbox(label="Or enter dataset URL (CSV or JSON)", visible=False, placeholder="https://example.com/data.csv")

    def toggle_upload(name):
        if name == "Upload File":
            return gr.update(visible=True), gr.update(visible=True)
        else:
            return gr.update(visible=False), gr.update(visible=False)

    dataset_choice.change(toggle_upload, inputs=dataset_choice, outputs=[upload_file, url_input])

    preview_btn = gr.Button("Preview Data")
    data_preview = gr.Dataframe(headers="auto", interactive=False, max_rows=10, max_cols=20)

    preview_btn.click(update_preview, inputs=[dataset_choice, upload_file], outputs=data_preview)

    gr.Markdown("## Dataset Summary and Notes")
    summary_output = gr.Textbox(label="Summary", lines=15, interactive=False)

    summarize_btn = gr.Button("Generate Summary")
    summarize_btn.click(update_summary, inputs=[dataset_choice, upload_file], outputs=summary_output)

    voice_toggle = gr.Checkbox(label="Enable Voice Output", value=False)
    audio_output = gr.Audio(label="Voice Summary", interactive=False)

    summarize_btn.click(voice_summary, inputs=[summary_output, voice_toggle], outputs=audio_output)

    gr.Markdown("## Visualizations")

    with gr.Tabs():
        with gr.TabItem("Scatter Plot"):
            with gr.Row():
                x_col = gr.Dropdown(label="X Axis")
                y_col = gr.Dropdown(label="Y Axis")
            scatter_btn = gr.Button("Generate Scatter Plot")
            scatter_output = gr.Plot()
            scatter_msg = gr.Textbox(label="Message", interactive=False, lines=1)

            # Update columns on dataset or file change
            dataset_choice.change(update_columns, inputs=[dataset_choice, upload_file], outputs=[x_col, y_col])
            upload_file.change(update_columns, inputs=[dataset_choice, upload_file], outputs=[x_col, y_col])

            scatter_btn.click(run_scatter, inputs=[dataset_choice, upload_file, x_col, y_col], outputs=[scatter_output, scatter_msg])

        with gr.TabItem("Histograms"):
            histograms_output = gr.Gallery(label="Histograms", show_label=False, elem_id="histograms")
            histograms_btn = gr.Button("Generate Histograms")
            histograms_btn.click(run_histograms, inputs=[dataset_choice, upload_file], outputs=histograms_output)

        with gr.TabItem("Box Plots"):
            boxplots_output = gr.Gallery(label="Box Plots", show_label=False, elem_id="boxplots")
            boxplots_btn = gr.Button("Generate Box Plots")
            boxplots_btn.click(run_boxplots, inputs=[dataset_choice, upload_file], outputs=boxplots_output)

        with gr.TabItem("Correlation Heatmap"):
            corr_output = gr.Plot()
            corr_btn = gr.Button("Generate Correlation Heatmap")
            corr_btn.click(run_corr_heatmap, inputs=[dataset_choice, upload_file], outputs=corr_output)

        with gr.TabItem("Scatter Matrix"):
            matrix_output = gr.Plot()
            matrix_btn = gr.Button("Generate Scatter Matrix")
            matrix_btn.click(run_scatter_matrix, inputs=[dataset_choice, upload_file], outputs=matrix_output)

    gr.Markdown("## Download Report")
    report_format = gr.Radio(label="Report Format", choices=["txt", "csv", "xlsx"], value="txt")
    generate_report_btn = gr.Button("Generate & Download Report")
    download_link = gr.HTML()

    generate_report_btn.click(generate_report, inputs=[dataset_choice, upload_file, report_format], outputs=download_link)

    gr.Markdown("---")
    gr.Markdown("**API Endpoint:** GET [http://127.0.0.1:8000/api/summaries](http://127.0.0.1:8000/api/summaries)")

demo.launch(server_name="0.0.0.0", server_port=7860)
