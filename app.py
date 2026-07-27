"""
app.py — GaadiScout Hugging Face Spaces Entry Point
"""

import gradio as gr
from dashboard import build_dashboard, CUSTOM_CSS

demo = build_dashboard()
demo.launch(
    css=CUSTOM_CSS,
    theme=gr.themes.Base(
        primary_hue="indigo",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    ),
)
