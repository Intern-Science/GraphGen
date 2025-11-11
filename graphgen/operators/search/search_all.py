"""
To use Google Web Search API,
follow the instructions [here](https://developers.google.com/custom-search/v1/overview)
to get your Google searcher api key.

To use Bing Web Search API,
follow the instructions [here](https://www.microsoft.com/en-us/bing/apis/bing-web-search-api)
and obtain your Bing subscription key.
"""


from graphgen.utils import logger, run_concurrent


async def search_all(
    seed_data: dict,
    search_config: dict,
) -> dict:
    """
    Perform searches across multiple search types and aggregate the results.
    :param seed_data: A dictionary containing seed data with entity names.
    :param search_config: A dictionary specifying which data sources to use for searching.
    :return: A dictionary with
    """

    results = {}
    data_sources = search_config.get("data_sources", [])

    for data_source in data_sources:
        if data_source == "uniprot":
            from graphgen.models import UniProtSearch

            uniprot_search_client = UniProtSearch(
                **search_config.get("uniprot_params", {})
            )

            data = list(seed_data.values())
            data = [d["content"] for d in data if "content" in d]
            data = list(set(data))  # Remove duplicates
            uniprot_results = await run_concurrent(
                uniprot_search_client.search,
                data,
                desc="Searching UniProt database",
                unit="keyword",
            )
        else:
            logger.error("Data source %s not supported.", data_source)
            continue

        results[data_source] = uniprot_results

    return results
