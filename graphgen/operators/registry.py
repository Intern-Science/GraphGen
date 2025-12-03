from .build_kg import build_kg
from .extract import extract_info
from .generate import generate_qas
from .init import init_llm
from .partition import partition_kg
from .quiz_and_judge import judge_statement, quiz
from .read import read
from .search import search_all
from .split import chunk_documents

operators = {
    "read": read,
    "init_llm": init_llm,
    "chunk_documents": chunk_documents,
    "extract_info": extract_info,
    "search_all": search_all,
    "build_kg": build_kg,
    "partition_kg": partition_kg,
    "generate_qas": generate_qas,
    "quiz": quiz,
    "judge_statement": judge_statement,
}
