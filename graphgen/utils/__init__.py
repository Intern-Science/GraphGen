from .calculate_confidence import yes_no_loss_entropy
from .detect_lang import detect_if_chinese, detect_main_language
from .format import (
    format_generation_results,
    handle_single_entity_extraction,
    handle_single_relationship_extraction,
    load_json,
    pack_history_conversations,
    split_string_by_multi_markers,
    write_json,
)
from .hash import compute_args_hash, compute_content_hash
from .help_nltk import NLTKHelper
from .log import logger, parse_log, set_logger
from .loop import create_event_loop
