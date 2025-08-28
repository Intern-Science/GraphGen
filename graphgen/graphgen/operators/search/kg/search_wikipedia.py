from tqdm.asyncio import tqdm_asyncio as tqdm_async

from graphgen.models import WikiSearch
from graphgen.utils import logger


async def _process_single_entity(
    entity_name: str,
    wiki_search_client: WikiSearch,
) -> str | None:
    """
    Process single entity by searching Wikipedia
    :param entity_name
    :param wiki_search_client
    :return: summary of the entity or None if not found
    """
    search_results = await wiki_search_client.search(entity_name)
    if not search_results:
        return None

    summary = None
    try:
        summary = await wiki_search_client.summary(search_results[-1])
        logger.info(
            "Entity %s search result: %s summary: %s",
            entity_name,
            str(search_results),
            summary,
        )
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error processing entity %s: %s", entity_name, str(e))

    return summary


async def search_wikipedia(
    wiki_search_client: WikiSearch,
    entities: set[str],
) -> dict:
    """
    Search wikipedia for entities

    :param wiki_search_client: wiki search client
    :param entities: list of entities to search
    :return: nodes with search results
    """
    wiki_data = {}

    async for entity in tqdm_async(
        entities, desc="Searching Wikipedia", total=len(entities)
    ):
        try:
            summary = await _process_single_entity(entity, wiki_search_client)
            if summary:
                wiki_data[entity] = summary
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Error processing entity %s: %s", entity, str(e))
    return wiki_data
