import trafilatura
from tqdm.asyncio import tqdm_asyncio as tqdm_async

from graphgen.models import GoogleSearch
from graphgen.utils import logger


async def _process_single_entity(
    entity_name: str, google_search_client: GoogleSearch
) -> str | None:
    search_results = google_search_client.search(entity_name)
    if not search_results:
        return None

    # Get more details from the first search result
    first_result = search_results[0]
    content = trafilatura.fetch_url(first_result["link"])
    summary = trafilatura.extract(content, include_comments=False, include_links=False)
    summary = summary.strip()
    logger.info(
        "Entity %s search result: %s",
        entity_name,
        summary,
    )
    return summary


async def search_google(
    google_search_client: GoogleSearch,
    entities: set[str],
) -> dict:
    """
    Search with Google and return the contexts.
    :param google_search_client: Google search client
    :param entities: list of entities to search
    :return:
    """
    google_data = {}

    async for entity in tqdm_async(
        entities, desc="Searching Google", total=len(entities)
    ):
        try:
            summary = await _process_single_entity(entity, google_search_client)
            if summary:
                google_data[entity] = summary
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Error processing entity %s: %s", entity, str(e))
    return google_data
