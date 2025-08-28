from typing import List

from graphgen.models import Chunk, OpenAIModel
from graphgen.templates import COREFERENCE_RESOLUTION_PROMPT
from graphgen.utils import detect_main_language


async def resolute_coreference(
    llm_client: OpenAIModel, chunks: List[Chunk]
) -> List[Chunk]:
    """
    Resolute conference

    :param llm_client: LLM model
    :param chunks: List of chunks
    :return: List of chunks
    """

    if len(chunks) == 0:
        return chunks

    results = [chunks[0]]

    for _, chunk in enumerate(chunks[1:]):
        language = detect_main_language(chunk.content)
        result = await llm_client.generate_answer(
            COREFERENCE_RESOLUTION_PROMPT[language].format(
                reference=results[0].content, input_sentence=chunk.content
            )
        )
        results.append(Chunk(id=chunk.id, content=result))

    return results
