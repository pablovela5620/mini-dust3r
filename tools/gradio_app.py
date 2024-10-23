from mini_dust3r.gradio_ui.dust3r_ui import mini_dust3r_ui
import gradio as gr

title = """# Mini-DUSt3R: Unofficial Demo of Dust3r Geometric 3D Vision Made Easy"""
description1 = """
    <a title="Github" href="https://github.com/pablovela5620/mini-dust3r" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
        <img src="https://img.shields.io/github/stars/pablovela5620/mini-dust3r?label=GitHub%20%E2%98%85&logo=github&color=C8C" alt="badge-github-stars">
    </a>
    <a title="Social" href="https://x.com/pablovelagomez1" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
        <img src="https://www.obukhov.ai/img/badges/badge-social.svg" alt="social">
    </a>
    <a title="Website" href="https://pablovela.dev/" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
        <img src="https://img.shields.io/badge/website-000000?style=for-the-badge&logo=About.me&logoColor=white">
    </a>
    <a title="arXiv" href="https://arxiv.org/abs/2312.14132" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
        <img src="https://www.obukhov.ai/img/badges/badge-pdf.svg">
    </a>
"""
description2 = "Using Rerun to visualize the results of InstantSplat"

with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description1)
    gr.Markdown(description2)
    mini_dust3r_ui.render()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
