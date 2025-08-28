"""
To use Google Web Search API,
follow the instructions [here](https://developers.google.com/custom-search/v1/overview)
to get your Google search api key.

To use Bing Web Search API,
follow the instructions [here](https://www.microsoft.com/en-us/bing/apis/bing-web-search-api)
and obtain your Bing subscription key.
"""

import os

from graphgen.utils import logger


async def search_all(
    search_types: dict, search_entities: set[str]
) -> dict[str, dict[str, str]]:
    """
    :param search_types
    :param search_entities: list of entities to search
    :return: nodes with search results
    """

    results = {}

    for search_type in search_types:
        if search_type == "wikipedia":
            from graphgen.models import WikiSearch
            from graphgen.operators.search.kg.search_wikipedia import search_wikipedia

            wiki_search_client = WikiSearch()

            wiki_results = await search_wikipedia(wiki_search_client, search_entities)
            for entity_name, description in wiki_results.items():
                if description:
                    results[entity_name] = {"wikipedia": description}
        elif search_type == "google":
            from graphgen.models import GoogleSearch
            from graphgen.operators.search.web.search_google import search_google

            google_search_client = GoogleSearch(
                subscription_key=os.environ["GOOGLE_SEARCH_API_KEY"],
                cx=os.environ["GOOGLE_SEARCH_CX"],
            )

            google_results = await search_google(google_search_client, search_entities)
            for entity_name, description in google_results.items():
                if description:
                    results[entity_name] = results.get(entity_name, {})
                    results[entity_name]["google"] = description
        elif search_type == "bing":
            from graphgen.models import BingSearch
            from graphgen.operators.search.web.search_bing import search_bing

            bing_search_client = BingSearch(
                subscription_key=os.environ["BING_SEARCH_API_KEY"]
            )

            bing_results = await search_bing(bing_search_client, search_entities)
            for entity_name, description in bing_results.items():
                if description:
                    results[entity_name] = results.get(entity_name, {})
                    results[entity_name]["bing"] = description
        elif search_type == "uniprot":
            # from graphgen.models import UniProtSearch
            # from graphgen.operators.search.db.search_uniprot import search_uniprot
            #
            # uniprot_search_client = UniProtSearch()
            #
            # uniprot_results = await search_uniprot(
            #     uniprot_search_client, search_entities
            # )
            raise NotImplementedError(
                "Processing of UniProt search results is not implemented yet."
            )

        else:
            logger.error("Search type %s is not supported yet.", search_type)
            continue

    return results
