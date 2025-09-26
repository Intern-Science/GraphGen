import json
import os
import sys
import tempfile
from importlib.resources import files

import gradio as gr
import pandas as pd
from dotenv import load_dotenv

from graphgen.graphgen import GraphGen
from graphgen.models import OpenAIModel, Tokenizer
from graphgen.models.llm.limitter import RPM, TPM
from graphgen.utils import set_logger
from webui.base import WebuiParams
from webui.i18n import Translate
from webui.i18n import gettext as _
from webui.test_api import test_api_connection
from webui.utils import cleanup_workspace, count_tokens, preview_file, setup_workspace

root_dir = files("webui").parent
sys.path.append(root_dir)


load_dotenv()

css = """
.center-row {
    display: flex;
    justify-content: center;
    align-items: center;
}
"""


def init_graph_gen(config: dict, env: dict) -> GraphGen:
    # Set up working directory
    log_file, working_dir = setup_workspace(os.path.join(root_dir, "cache"))
    set_logger(log_file, if_stream=True)
    os.environ.update({k: str(v) for k, v in env.items()})

    graph_gen = GraphGen(working_dir=working_dir, config=config)
    # Set up LLM clients
    graph_gen.synthesizer_llm_client = OpenAIModel(
        model_name=env.get("SYNTHESIZER_MODEL", ""),
        base_url=env.get("SYNTHESIZER_BASE_URL", ""),
        api_key=env.get("SYNTHESIZER_API_KEY", ""),
        request_limit=True,
        rpm=RPM(env.get("RPM", 1000)),
        tpm=TPM(env.get("TPM", 50000)),
    )

    graph_gen.trainee_llm_client = OpenAIModel(
        model_name=env.get("TRAINEE_MODEL", ""),
        base_url=env.get("TRAINEE_BASE_URL", ""),
        api_key=env.get("TRAINEE_API_KEY", ""),
        request_limit=True,
        rpm=RPM(env.get("RPM", 1000)),
        tpm=TPM(env.get("TPM", 50000)),
    )

    graph_gen.tokenizer_instance = Tokenizer(config.get("tokenizer", "cl100k_base"))

    return graph_gen


# pylint: disable=too-many-statements
def run_graphgen(params: WebuiParams, progress=gr.Progress()):
    def sum_tokens(client):
        return sum(u["total_tokens"] for u in client.token_usage)

    config = {
        "if_trainee_model": params.if_trainee_model,
        "read": {
            "input_file": params.input_file,
        },
        "split": {
            "chunk_size": params.chunk_size,
            "chunk_overlap": params.chunk_overlap,
        },
        "output_data_type": params.output_data_type,
        "output_data_format": params.output_data_format,
        "tokenizer": params.tokenizer,
        "search": {"enabled": False},
        "quiz_and_judge_strategy": {
            "enabled": params.if_trainee_model,
            "quiz_samples": params.quiz_samples,
        },
        "traverse_strategy": {
            "bidirectional": params.bidirectional,
            "expand_method": params.expand_method,
            "max_extra_edges": params.max_extra_edges,
            "max_tokens": params.max_tokens,
            "max_depth": params.max_depth,
            "edge_sampling": params.edge_sampling,
            "isolated_node_strategy": params.isolated_node_strategy,
            "loss_strategy": params.loss_strategy,
        },
    }

    env = {
        "SYNTHESIZER_BASE_URL": params.synthesizer_url,
        "SYNTHESIZER_MODEL": params.synthesizer_model,
        "TRAINEE_BASE_URL": params.trainee_url,
        "TRAINEE_MODEL": params.trainee_model,
        "SYNTHESIZER_API_KEY": params.api_key,
        "TRAINEE_API_KEY": params.trainee_api_key,
        "RPM": params.rpm,
        "TPM": params.tpm,
    }

    # Test API connection
    test_api_connection(
        env["SYNTHESIZER_BASE_URL"],
        env["SYNTHESIZER_API_KEY"],
        env["SYNTHESIZER_MODEL"],
    )
    if config["if_trainee_model"]:
        test_api_connection(
            env["TRAINEE_BASE_URL"], env["TRAINEE_API_KEY"], env["TRAINEE_MODEL"]
        )

    # Initialize GraphGen
    graph_gen = init_graph_gen(config, env)
    graph_gen.clear()

    graph_gen.progress_bar = progress

    try:
        # Process the data
        graph_gen.insert()

        if config["if_trainee_model"]:
            # Generate quiz
            graph_gen.quiz()

            # Judge statements
            graph_gen.judge()
        else:
            graph_gen.traverse_strategy.edge_sampling = "random"

        # Traverse graph
        graph_gen.traverse()

        # Save output
        output_data = graph_gen.qa_storage.data
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as tmpfile:
            json.dump(output_data, tmpfile, ensure_ascii=False)
            output_file = tmpfile.name

        synthesizer_tokens = sum_tokens(graph_gen.synthesizer_llm_client)
        trainee_tokens = (
            sum_tokens(graph_gen.trainee_llm_client)
            if config["if_trainee_model"]
            else 0
        )
        total_tokens = synthesizer_tokens + trainee_tokens

        data_frame = params.token_counter
        try:
            _update_data = [
                [data_frame.iloc[0, 0], data_frame.iloc[0, 1], str(total_tokens)]
            ]
            new_df = pd.DataFrame(_update_data, columns=data_frame.columns)
            data_frame = new_df

        except Exception as e:
            raise gr.Error(f"DataFrame operation error: {str(e)}")

        return output_file, gr.DataFrame(
            label="Token Stats",
            headers=["Source Text Token Count", "Expected Token Usage", "Token Used"],
            datatype="str",
            interactive=False,
            value=data_frame,
            visible=True,
            wrap=True,
        )

    except Exception as e:  # pylint: disable=broad-except
        raise gr.Error(f"Error occurred: {str(e)}")

    finally:
        # Clean up workspace
        cleanup_workspace(graph_gen.working_dir)


with gr.Blocks(title="GraphGen Demo", theme=gr.themes.Glass(), css=css) as demo:
    # Header
    gr.Image(
        value=os.path.join(root_dir, "resources", "images", "logo.png"),
        label="GraphGen Banner",
        elem_id="banner",
        interactive=False,
        container=False,
        show_download_button=False,
        show_fullscreen_button=False,
    )
    lang_btn = gr.Radio(
        choices=[
            ("English", "en"),
            ("简体中文", "zh"),
        ],
        value="en",
        # label=_("Language"),
        render=False,
        container=False,
        elem_classes=["center-row"],
    )

    gr.HTML(
        """
    <div style="display: flex; gap: 8px; margin-left: auto; align-items: center; justify-content: center;">
        <a href="https://github.com/open-sciencelab/GraphGen/releases">
            <img src="https://img.shields.io/badge/Version-v0.1.0-blue" alt="Version">
        </a>
        <a href="https://graphgen-docs.example.com">
            <img src="https://img.shields.io/badge/Docs-Latest-brightgreen" alt="Documentation">
        </a>
        <a href="https://github.com/open-sciencelab/GraphGen/issues/10">
            <img src="https://img.shields.io/github/stars/open-sciencelab/GraphGen?style=social" alt="GitHub Stars">
        </a>
        <a href="https://arxiv.org/abs/2505.20416">
            <img src="https://img.shields.io/badge/arXiv-pdf-yellow" alt="arXiv">
        </a>
    </div>
    """
    )
    with Translate(
        os.path.join(root_dir, "webui", "translation.json"),
        lang_btn,
        placeholder_langs=["en", "zh"],
        persistant=False,  # True to save the language setting in the browser. Requires gradio >= 5.6.0
    ):
        lang_btn.render()

        gr.Markdown(
            value="# "
            + _("Title")
            + "\n\n"
            + "### [GraphGen](https://github.com/open-sciencelab/GraphGen) "
            + _("Intro")
        )

        if_trainee_model = gr.Checkbox(
            label=_("Use Trainee Model"), value=False, interactive=True
        )

        with gr.Accordion(label=_("Model Config"), open=False):
            synthesizer_url = gr.Textbox(
                label="Synthesizer URL",
                value="https://api.siliconflow.cn/v1",
                info=_("Synthesizer URL Info"),
                interactive=True,
            )
            synthesizer_model = gr.Textbox(
                label="Synthesizer Model",
                value="Qwen/Qwen2.5-7B-Instruct",
                info=_("Synthesizer Model Info"),
                interactive=True,
            )
            trainee_url = gr.Textbox(
                label="Trainee URL",
                value="https://api.siliconflow.cn/v1",
                info=_("Trainee URL Info"),
                interactive=True,
                visible=if_trainee_model.value is True,
            )
            trainee_model = gr.Textbox(
                label="Trainee Model",
                value="Qwen/Qwen2.5-7B-Instruct",
                info=_("Trainee Model Info"),
                interactive=True,
                visible=if_trainee_model.value is True,
            )
            trainee_api_key = gr.Textbox(
                label=_("SiliconFlow Token for Trainee Model"),
                type="password",
                value="",
                info="https://cloud.siliconflow.cn/account/ak",
                visible=if_trainee_model.value is True,
            )

        with gr.Accordion(label=_("Generation Config"), open=False):
            chunk_size = gr.Slider(
                label="Chunk Size",
                minimum=256,
                maximum=4096,
                value=1024,
                step=256,
                interactive=True,
            )
            chunk_overlap = gr.Slider(
                label="Chunk Overlap",
                minimum=0,
                maximum=500,
                value=100,
                step=100,
                interactive=True,
            )
            tokenizer = gr.Textbox(
                label="Tokenizer", value="cl100k_base", interactive=True
            )
            output_data_type = gr.Radio(
                choices=["atomic", "multi_hop", "aggregated"],
                label="Output Data Type",
                value="aggregated",
                interactive=True,
            )
            output_data_format = gr.Radio(
                choices=["Alpaca", "Sharegpt", "ChatML"],
                label="Output Data Format",
                value="Alpaca",
                interactive=True,
            )
            quiz_samples = gr.Number(
                label="Quiz Samples",
                value=2,
                minimum=1,
                interactive=True,
                visible=if_trainee_model.value is True,
            )
            bidirectional = gr.Checkbox(
                label="Bidirectional", value=True, interactive=True
            )

            expand_method = gr.Radio(
                choices=["max_width", "max_tokens"],
                label="Expand Method",
                value="max_tokens",
                interactive=True,
            )
            max_extra_edges = gr.Slider(
                minimum=1,
                maximum=10,
                value=5,
                label="Max Extra Edges",
                step=1,
                interactive=True,
                visible=expand_method.value == "max_width",
            )
            max_tokens = gr.Slider(
                minimum=64,
                maximum=1024,
                value=256,
                label="Max Tokens",
                step=64,
                interactive=True,
                visible=(expand_method.value != "max_width"),
            )

            max_depth = gr.Slider(
                minimum=1,
                maximum=5,
                value=2,
                label="Max Depth",
                step=1,
                interactive=True,
            )
            edge_sampling = gr.Radio(
                choices=["max_loss", "min_loss", "random"],
                label="Edge Sampling",
                value="max_loss",
                interactive=True,
                visible=if_trainee_model.value is True,
            )
            isolated_node_strategy = gr.Radio(
                choices=["add", "ignore"],
                label="Isolated Node Strategy",
                value="ignore",
                interactive=True,
            )
            loss_strategy = gr.Radio(
                choices=["only_edge", "both"],
                label="Loss Strategy",
                value="only_edge",
                interactive=True,
            )

        with gr.Row(equal_height=True):
            with gr.Column(scale=3):
                api_key = gr.Textbox(
                    label=_("SiliconFlow Token"),
                    type="password",
                    value="",
                    info="https://cloud.siliconflow.cn/account/ak",
                )
            with gr.Column(scale=1):
                test_connection_btn = gr.Button(_("Test Connection"))

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                with gr.Blocks():
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1):
                            upload_file = gr.File(
                                label=_("Upload File"),
                                file_count="single",
                                file_types=[".txt", ".json", ".jsonl", ".csv"],
                                interactive=True,
                            )
                            examples_dir = os.path.join(root_dir, "webui", "examples")
                            gr.Examples(
                                examples=[
                                    [os.path.join(examples_dir, "txt_demo.txt")],
                                    [os.path.join(examples_dir, "jsonl_demo.jsonl")],
                                    [os.path.join(examples_dir, "json_demo.json")],
                                    [os.path.join(examples_dir, "csv_demo.csv")],
                                ],
                                inputs=upload_file,
                                label=_("Example Files"),
                                examples_per_page=4,
                            )
            with gr.Column(scale=1):
                with gr.Blocks():
                    preview_code = gr.Code(
                        label=_("File Preview"),
                        interactive=False,
                        visible=True,
                        elem_id="preview_code",
                    )
                    preview_df = gr.DataFrame(
                        label=_("File Preview"),
                        interactive=False,
                        visible=False,
                        elem_id="preview_df",
                    )

        with gr.Blocks():
            token_counter = gr.DataFrame(
                label="Token Stats",
                headers=[
                    "Source Text Token Count",
                    "Estimated Token Usage",
                    "Token Used",
                ],
                datatype="str",
                interactive=False,
                visible=False,
                wrap=True,
            )

        with gr.Blocks():
            with gr.Row(equal_height=True):
                with gr.Column():
                    rpm = gr.Slider(
                        label="RPM",
                        minimum=10,
                        maximum=10000,
                        value=1000,
                        step=100,
                        interactive=True,
                        visible=True,
                    )
                with gr.Column():
                    tpm = gr.Slider(
                        label="TPM",
                        minimum=5000,
                        maximum=5000000,
                        value=50000,
                        step=1000,
                        interactive=True,
                        visible=True,
                    )

        with gr.Blocks():
            with gr.Column(scale=1):
                output = gr.File(
                    label=_("Output File"),
                    file_count="single",
                    interactive=False,
                )

        submit_btn = gr.Button(_("Run GraphGen"))

        # Test Connection
        test_connection_btn.click(
            test_api_connection,
            inputs=[synthesizer_url, api_key, synthesizer_model],
            outputs=[],
        )

        if if_trainee_model.value:
            test_connection_btn.click(
                test_api_connection,
                inputs=[trainee_url, api_key, trainee_model],
                outputs=[],
            )

        expand_method.change(
            lambda method: (
                gr.update(visible=method == "max_width"),
                gr.update(visible=method != "max_width"),
            ),
            inputs=expand_method,
            outputs=[max_extra_edges, max_tokens],
        )

        if_trainee_model.change(
            lambda use_trainee: [gr.update(visible=use_trainee)] * 5,
            inputs=if_trainee_model,
            outputs=[
                trainee_url,
                trainee_model,
                quiz_samples,
                edge_sampling,
                trainee_api_key,
            ],
        )

        upload_file.change(
            preview_file, inputs=upload_file, outputs=[preview_code, preview_df]
        ).then(
            lambda x: gr.update(visible=True), inputs=upload_file, outputs=token_counter
        ).then(
            count_tokens,
            inputs=[upload_file, tokenizer, token_counter],
            outputs=token_counter,
        )

        # run GraphGen
        submit_btn.click(
            lambda x: (gr.update(visible=False)),
            inputs=[token_counter],
            outputs=[token_counter],
        )

        submit_btn.click(
            lambda *args: run_graphgen(
                WebuiParams(
                    if_trainee_model=args[0],
                    input_file=args[1],
                    tokenizer=args[2],
                    output_data_type=args[3],
                    output_data_format=args[4],
                    bidirectional=args[5],
                    expand_method=args[6],
                    max_extra_edges=args[7],
                    max_tokens=args[8],
                    max_depth=args[9],
                    edge_sampling=args[10],
                    isolated_node_strategy=args[11],
                    loss_strategy=args[12],
                    synthesizer_url=args[13],
                    synthesizer_model=args[14],
                    trainee_model=args[15],
                    api_key=args[16],
                    chunk_size=args[17],
                    chunk_overlap=args[18],
                    rpm=args[19],
                    tpm=args[20],
                    quiz_samples=args[21],
                    trainee_url=args[22],
                    trainee_api_key=args[23],
                    token_counter=args[24],
                )
            ),
            inputs=[
                if_trainee_model,
                upload_file,
                tokenizer,
                output_data_type,
                output_data_format,
                bidirectional,
                expand_method,
                max_extra_edges,
                max_tokens,
                max_depth,
                edge_sampling,
                isolated_node_strategy,
                loss_strategy,
                synthesizer_url,
                synthesizer_model,
                trainee_model,
                api_key,
                chunk_size,
                chunk_overlap,
                rpm,
                tpm,
                quiz_samples,
                trainee_url,
                trainee_api_key,
                token_counter,
            ],
            outputs=[output, token_counter],
        )


if __name__ == "__main__":
    demo.queue(api_open=False, default_concurrency_limit=2)
    demo.launch(server_name="0.0.0.0")
