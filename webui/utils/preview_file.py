import codecs
import os

import gradio as gr
import pandas as pd


def preview_file(file):
    if file is None:
        return gr.update(visible=False), gr.update(visible=False)

    path = file.name
    ext = os.path.splitext(path)[1].lower()

    try:
        if ext == ".csv":
            df = pd.read_csv(path, nrows=10)
            return gr.update(visible=False), gr.update(value=df, visible=True)
        with codecs.open(path, "r", encoding="utf-8") as f:
            text = f.read(5000)
            if len(text) == 5000:
                text += "\n\n... (truncated at 5000 chars)"
        return gr.update(
            value=text, visible=True, language="json" if ext != ".txt" else None
        ), gr.update(visible=False)
    except Exception as e:  # pylint: disable=broad-except
        return gr.update(
            value=f"Preview failed: {e}", visible=True, language=None
        ), gr.update(visible=False)
