import json
import os
import sys
import tempfile
from importlib.resources import files

import gradio as gr
import pandas as pd
from dotenv import load_dotenv

from graphgen.engine import Context, Engine, collect_ops
from graphgen.graphgen import GraphGen
from graphgen.models import OpenAIClient, Tokenizer
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

    tokenizer_instance = Tokenizer(config.get("tokenizer", "cl100k_base"))
    synthesizer_llm_client = OpenAIClient(
        model_name=env.get("SYNTHESIZER_MODEL", ""),
        base_url=env.get("SYNTHESIZER_BASE_URL", ""),
        api_key=env.get("SYNTHESIZER_API_KEY", ""),
        request_limit=True,
        rpm=RPM(env.get("RPM", 1000)),
        tpm=TPM(env.get("TPM", 50000)),
        tokenizer=tokenizer_instance,
    )
    trainee_llm_client = OpenAIClient(
        model_name=env.get("TRAINEE_MODEL", ""),
        base_url=env.get("TRAINEE_BASE_URL", ""),
        api_key=env.get("TRAINEE_API_KEY", ""),
        request_limit=True,
        rpm=RPM(env.get("RPM", 1000)),
        tpm=TPM(env.get("TPM", 50000)),
        tokenizer=tokenizer_instance,
    )

    graph_gen = GraphGen(
        working_dir=working_dir,
        tokenizer_instance=tokenizer_instance,
        synthesizer_llm_client=synthesizer_llm_client,
        trainee_llm_client=trainee_llm_client,
    )

    return graph_gen


# pylint: disable=too-many-statements
def run_graphgen(params: WebuiParams, progress=gr.Progress()):
    def sum_tokens(client):
        return sum(u["total_tokens"] for u in client.token_usage)

    method = params.partition_method
    if method == "dfs":
        partition_params = {
            "max_units_per_community": params.dfs_max_units,
        }
    elif method == "bfs":
        partition_params = {
            "max_units_per_community": params.bfs_max_units,
        }
    elif method == "leiden":
        partition_params = {
            "max_size": params.leiden_max_size,
            "use_lcc": params.leiden_use_lcc,
            "random_seed": params.leiden_random_seed,
        }
    else:  # ece
        partition_params = {
            "max_units_per_community": params.ece_max_units,
            "min_units_per_community": params.ece_min_units,
            "max_tokens_per_community": params.ece_max_tokens,
            "unit_sampling": params.ece_unit_sampling,
        }

    pipeline = [
        {
            "name": "read",
            "params": {
                "input_file": params.upload_file,
                "chunk_size": params.chunk_size,
                "chunk_overlap": params.chunk_overlap,
            },
        },
        {
            "name": "build_kg",
        },
    ]

    if params.if_trainee_model:
        pipeline.append(
            {
                "name": "quiz_and_judge",
                "params": {"quiz_samples": params.quiz_samples, "re_judge": True},
            }
        )
        pipeline.append(
            {
                "name": "partition",
                "deps": ["quiz_and_judge"],
                "params": {
                    "method": params.partition_method,
                    "method_params": partition_params,
                },
            }
        )
    else:
        pipeline.append(
            {
                "name": "partition",
                "params": {
                    "method": params.partition_method,
                    "method_params": partition_params,
                },
            }
        )
    pipeline.append(
        {
            "name": "generate",
            "params": {
                "method": params.mode,
                "data_format": params.data_format,
            },
        }
    )

    config = {
        "if_trainee_model": params.if_trainee_model,
        "read": {"input_file": params.upload_file},
        "pipeline": pipeline,
    }

    env = {
        "TOKENIZER_MODEL": params.tokenizer,
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
        ctx = Context(config=config, graph_gen=graph_gen)
        ops = collect_ops(config, graph_gen)
        Engine(max_workers=config.get("max_workers", 4)).run(ops, ctx)

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

        gr.Markdown(value=_("Title") + _("Intro"))

        if_trainee_model = gr.Checkbox(
            label=_("Use Trainee Model"), value=False, interactive=True
        )

        with gr.Accordion(label=_("Model Config"), open=False):
            tokenizer = gr.Textbox(
                label="Tokenizer", value="cl100k_base", interactive=True
            )
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

        with gr.Row(equal_height=True):
            with gr.Column(scale=3):
                api_key = gr.Textbox(
                    label=_("SiliconFlow Token"),
                    type="password",
                    value="",
                    info=_("SiliconFlow Token Info"),
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

        with gr.Accordion(label=_("Split Config"), open=False):
            gr.Markdown(value=_("Split Config Info"))
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    chunk_size = gr.Slider(
                        label=_("Chunk Size"),
                        minimum=256,
                        maximum=4096,
                        value=1024,
                        step=256,
                        interactive=True,
                        info=_("Chunk Size Info"),
                    )
                with gr.Column(scale=1):
                    chunk_overlap = gr.Slider(
                        label=_("Chunk Overlap"),
                        minimum=0,
                        maximum=500,
                        value=100,
                        step=100,
                        interactive=True,
                        info=_("Chunk Overlap Info"),
                    )

        with gr.Accordion(
            label=_("Quiz & Judge Config"), open=False, visible=False
        ) as quiz_accordion:
            gr.Markdown(value=_("Quiz & Judge Config Info"))
            quiz_samples = gr.Number(
                label=_("Quiz Samples"),
                value=2,
                minimum=1,
                interactive=True,
                info=_("Quiz Samples Info"),
            )

        with gr.Accordion(label=_("Partition Config"), open=False):
            gr.Markdown(value=_("Partition Config Info"))

            partition_method = gr.Dropdown(
                label=_("Partition Method"),
                choices=["dfs", "bfs", "ece", "leiden"],
                value="ece",
                interactive=True,
                info=_("Which algorithm to use for graph partitioning."),
            )

            # DFS method parameters
            with gr.Group(visible=False) as dfs_group:
                gr.Markdown(_("DFS intro"))
                dfs_max_units = gr.Slider(
                    label=_("Max Units Per Community"),
                    minimum=1,
                    maximum=100,
                    value=5,
                    step=1,
                    interactive=True,
                    info=_("Max Units Per Community Info"),
                )
            # BFS method parameters
            with gr.Group(visible=False) as bfs_group:
                gr.Markdown(_("BFS intro"))
                bfs_max_units = gr.Slider(
                    label=_("Max Units Per Community"),
                    minimum=1,
                    maximum=100,
                    value=5,
                    step=1,
                    interactive=True,
                    info=_("Max Units Per Community Info"),
                )

            # Leiden method parameters
            with gr.Group(visible=False) as leiden_group:
                gr.Markdown(_("Leiden intro"))
                leiden_max_size = gr.Slider(
                    label=_("Maximum Size of Communities"),
                    minimum=1,
                    maximum=100,
                    value=20,
                    step=1,
                    interactive=True,
                    info=_("Maximum Size of Communities Info"),
                )
                leiden_use_lcc = gr.Checkbox(
                    label=_("Use Largest Connected Component"),
                    value=False,
                    interactive=True,
                    info=_("Use Largest Connected Component Info"),
                )
                leiden_random_seed = gr.Number(
                    label=_("Random Seed"),
                    value=42,
                    precision=0,
                    interactive=True,
                    info=_("Random Seed Info"),
                )

            # ECE method parameters
            with gr.Group(visible=True) as ece_group:
                gr.Markdown(_("ECE intro"))
                ece_max_units = gr.Slider(
                    label=_("Max Units Per Community"),
                    minimum=1,
                    maximum=100,
                    value=20,
                    step=1,
                    interactive=True,
                    info=_("Max Units Per Community Info"),
                )
                ece_min_units = gr.Slider(
                    label=_("Min Units Per Community"),
                    minimum=1,
                    maximum=100,
                    value=3,
                    step=1,
                    interactive=True,
                    info=_("Min Units Per Community Info"),
                )
                ece_max_tokens = gr.Slider(
                    label=_("Max Tokens Per Community"),
                    minimum=512,
                    maximum=20_480,
                    value=10_240,
                    step=512,
                    interactive=True,
                    info=_("Max Tokens Per Community Info"),
                )
                ece_unit_sampling = gr.Radio(
                    label=_("Unit Sampling Strategy"),
                    choices=["random"],
                    value="random",
                    interactive=True,
                    info=_("Unit Sampling Strategy Info"),
                )

            def toggle_partition_params(method):
                dfs = method == "dfs"
                bfs = method == "bfs"
                leiden = method == "leiden"
                ece = method == "ece"
                return (
                    gr.update(visible=dfs),  # dfs_group
                    gr.update(visible=bfs),  # bfs_group
                    gr.update(visible=leiden),  # leiden_group
                    gr.update(visible=ece),  # ece_group
                )

            partition_method.change(
                fn=toggle_partition_params,
                inputs=partition_method,
                outputs=[dfs_group, bfs_group, leiden_group, ece_group],
            )

        with gr.Accordion(label=_("Generation Config"), open=False):
            gr.Markdown(value=_("Generation Config Info"))
            mode = gr.Radio(
                choices=["atomic", "multi_hop", "aggregated", "CoT"],
                label=_("Mode"),
                value="aggregated",
                interactive=True,
                info=_("Mode Info"),
            )
            data_format = gr.Radio(
                choices=["Alpaca", "Sharegpt", "ChatML"],
                label=_("Output Data Format"),
                value="Alpaca",
                interactive=True,
                info=_("Output Data Format Info"),
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

        if_trainee_model.change(
            lambda use_trainee: [gr.update(visible=use_trainee)] * 4,
            inputs=if_trainee_model,
            outputs=[
                trainee_url,
                trainee_model,
                trainee_api_key,
                quiz_accordion,
            ],
        )

        if_trainee_model.change(
            lambda on: (
                gr.update(
                    choices=["random"]
                    if not on
                    else ["random", "max_loss", "min_loss"],
                    value="random",
                )
            ),
            inputs=if_trainee_model,
            outputs=ece_unit_sampling,
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
                WebuiParams(**dict(zip(WebuiParams.__annotations__, args)))
            ),
            inputs=[
                if_trainee_model,
                upload_file,
                tokenizer,
                synthesizer_model,
                synthesizer_url,
                trainee_model,
                trainee_url,
                api_key,
                trainee_api_key,
                chunk_size,
                chunk_overlap,
                quiz_samples,
                partition_method,
                dfs_max_units,
                bfs_max_units,
                leiden_max_size,
                leiden_use_lcc,
                leiden_random_seed,
                ece_max_units,
                ece_min_units,
                ece_max_tokens,
                ece_unit_sampling,
                mode,
                data_format,
                rpm,
                tpm,
                token_counter,
            ],
            outputs=[output, token_counter],
        )


if __name__ == "__main__":
    demo.queue(api_open=False, default_concurrency_limit=2)
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)
