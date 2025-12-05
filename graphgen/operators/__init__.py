from .build_kg import BuildKGService
from .chunk import ChunkService
from .extract import extract_info
from .generate import generate_qas
from .partition import partition_kg
from .quiz import QuizService
from .read import read
from .search import search_all

operators = {
    "read": read,
    "chunk": ChunkService,
    "build_kg": BuildKGService,
    "quiz": QuizService,
    "extract_info": extract_info,
    "search_all": search_all,
    "partition_kg": partition_kg,
    "generate_qas": generate_qas,
}
