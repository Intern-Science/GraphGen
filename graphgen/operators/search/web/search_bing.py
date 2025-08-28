import trafilatura
from tqdm.asyncio import tqdm_asyncio as tqdm_async

from graphgen.models import BingSearch
from graphgen.utils import logger


async def _process_single_entity(
    entity_name: str, bing_search_client: BingSearch
) -> str | None:
    """
    Process single entity by searching Bing.
    :param entity_name: The name of the entity to search.
    :param bing_search_client: The Bing search client.
    :return: Summary of the entity or None if not found.
    """
    search_results = bing_search_client.search(entity_name)
    if not search_results:
        return None

    # Get more details from the first search result
    first_result = search_results[0]
    content = trafilatura.fetch_url(first_result["url"])
    summary = trafilatura.extract(content, include_comments=False, include_links=False)
    summary = summary.strip()
    logger.info(
        "Entity %s search result: %s",
        entity_name,
        summary,
    )
    return summary


async def search_bing(
    bing_search_client: BingSearch,
    entities: set[str],
) -> dict[str, str]:
    """
    Search with Bing and return the contexts.
    :return:
    """
    bing_data = {}

    async for entity in tqdm_async(
        entities, desc="Searching Bing", total=len(entities)
    ):
        try:
            summary = await _process_single_entity(entity, bing_search_client)
            if summary:
                bing_data[entity] = summary
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Error processing entity %s: %s", entity, str(e))
    return bing_data
