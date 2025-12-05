import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import ray
from ray.data import Dataset

from graphgen.bases.base_reader import BaseReader
from graphgen.models.reader.txt_reader import TXTReader
from graphgen.utils import logger, pick_device


class PDFReader(BaseReader):
    """
    PDF files are converted using MinerU, see [MinerU](https://github.com/opendatalab/MinerU).
    After conversion, the resulting markdown file is parsed into text, images, tables, and formulas which can be used
    for multi-modal graph generation.
    """

    def __init__(
        self,
        *,
        output_dir: Optional[Union[str, Path]] = None,
        method: str = "auto",  # auto | txt | ocr
        lang: Optional[str] = None,  # ch / en / ja / ...
        backend: Optional[
            str
        ] = None,  # pipeline | vlm-transformers | vlm-sglang-engine | vlm-sglang-client
        device: Optional[str] = "auto",  # cpu | cuda | cuda:0 | npu | mps | auto
        source: Optional[str] = None,  # huggingface | modelscope | local
        vlm_url: Optional[str] = None,  # 当 backend=vlm-sglang-client 时必填
        start_page: Optional[int] = None,  # 0-based
        end_page: Optional[int] = None,  # 0-based， inclusive
        formula: bool = True,
        table: bool = True,
        return_assets: bool = True,
        **other_mineru_kwargs: Any,
    ):
        super().__init__()
        self.output_dir = os.path.join(output_dir, "mineru") if output_dir else None

        if device == "auto":
            device = pick_device()

        self._default_kwargs: Dict[str, Any] = {
            "method": method,
            "lang": lang,
            "backend": backend,
            "device": device,
            "source": source,
            "vlm_url": vlm_url,
            "start_page": start_page,
            "end_page": end_page,
            "formula": formula,
            "table": table,
            **other_mineru_kwargs,
        }
        self._default_kwargs = {
            k: v for k, v in self._default_kwargs.items() if v is not None
        }
        self.return_assets = return_assets
        self.parser = MinerUParser()
        self.txt_reader = TXTReader()

    def read(
        self,
        input_path: Union[str, List[str]],
        **override,
    ) -> Dataset:

        # Ensure input_path is a list
        if isinstance(input_path, str):
            input_path = [input_path]

        paths_ds = ray.data.from_items(input_path)

        def process_pdf(row: Dict[str, Any]) -> List[Dict[str, Any]]:
            try:
                pdf_path = row["item"]
                kwargs = {**self._default_kwargs, **override}
                return self._call_mineru(Path(pdf_path), kwargs)
            except Exception as e:
                logger.error("Failed to process %s: %s", row, e)
                return []

        docs_ds = paths_ds.flat_map(process_pdf)
        docs_ds = docs_ds.filter(self._should_keep_item)

        return docs_ds

    def _call_mineru(
        self, pdf_path: Path, kwargs: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        output_dir: Optional[str] = None
        if self.output_dir:
            output_dir = str(self.output_dir)

        return self.parser.parse_pdf(pdf_path, output_dir=output_dir, **kwargs)

    def _locate_md(self, pdf_path: Path, kwargs: Dict[str, Any]) -> Optional[Path]:
        out_dir = (
            Path(self.output_dir) if self.output_dir else Path(tempfile.gettempdir())
        )
        method = kwargs.get("method", "auto")
        backend = kwargs.get("backend", "")
        if backend.startswith("vlm-"):
            method = "vlm"

        candidate = Path(
            os.path.join(out_dir, pdf_path.stem, method, f"{pdf_path.stem}.md")
        )
        if candidate.exists():
            return candidate
        candidate = Path(os.path.join(out_dir, f"{pdf_path.stem}.md"))
        if candidate.exists():
            return candidate
        return None


class MinerUParser:
    def __init__(self) -> None:
        self._check_bin()

    @staticmethod
    def parse_pdf(
        pdf_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        method: str = "auto",
        device: str = "cpu",
        **kw: Any,
    ) -> List[Dict[str, Any]]:
        pdf = Path(pdf_path).expanduser().resolve()
        if not pdf.is_file():
            raise FileNotFoundError(pdf)

        out = (
            Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="mineru_"))
        )
        out.mkdir(parents=True, exist_ok=True)

        cached = MinerUParser._try_load_cached_result(str(out), pdf.stem, method)
        if cached is not None:
            return cached

        MinerUParser._run_mineru(pdf, out, method, device, **kw)

        cached = MinerUParser._try_load_cached_result(str(out), pdf.stem, method)
        return cached if cached is not None else []

    @staticmethod
    def _try_load_cached_result(
        out_dir: str, pdf_stem: str, method: str
    ) -> Optional[List[Dict[str, Any]]]:
        """
        try to load cached json result from MinerU output.
        :param out_dir:
        :param pdf_stem:
        :param method:
        :return:
        """
        json_file = os.path.join(
            out_dir, pdf_stem, method, f"{pdf_stem}_content_list.json"
        )
        if not os.path.exists(json_file):
            return None

        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Failed to load cached MinerU result: %s", exc)
            return None

        base = os.path.dirname(json_file)
        results = []
        for it in data:
            for key in ("img_path", "table_img_path", "equation_img_path"):
                rel_path = it.get(key)
                if rel_path:
                    it[key] = str(Path(base).joinpath(rel_path).resolve())
            if it["type"] == "text":
                it["content"] = it["text"]
                del it["text"]
            for key in ("page_idx", "bbox", "text_level"):
                if it.get(key) is not None:
                    del it[key]
            results.append(it)
        return results

    @staticmethod
    def _run_mineru(
        pdf: Path,
        out: Path,
        method: str,
        device: str,
        **kw: Any,
    ) -> None:
        cmd = [
            "mineru",
            "-p",
            str(pdf),
            "-o",
            str(out),
            "-m",
            method,
            "-d",
            device,
        ]
        for k, v in kw.items():
            if v is None:
                continue
            if isinstance(v, bool):
                cmd += [f"--{k}", str(v).lower()]
            else:
                cmd += [f"--{k}", str(v)]

        logger.info("Parsing PDF with MinerU: %s", pdf)
        logger.debug("Running MinerU command: %s", " ".join(cmd))

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"MinerU failed: {proc.stderr or proc.stdout}")

    @staticmethod
    def _check_bin() -> None:
        try:
            subprocess.run(
                ["mineru", "--version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise RuntimeError(
                "MinerU is not installed or not found in PATH. Please install it from pip: \n"
                "pip install -U 'mineru[core]'"
            ) from exc
