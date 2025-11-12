# pylint: disable=C0301

TEMPLATE: str = """-Goal-
Please select the most relevant searcher result for the given entity. 
The name and description of the entity are provided. The searcher results are provided as a list.
Please select the most relevant searcher result from the list. If none of the searcher results are relevant, please select 'None of the above'.

Steps:
1. Read the name and description of the entity.

2. Read the searcher results. For each searcher result, compare it with the entity name and description to determine if it is relevant.

3. Select the most relevant searcher result from the list. If none of the searcher results are relevant, select 'None of the above'.

4. Output your selection directly, please do not provide any additional information.

################
-Examples-
################
{input_examples}

################
-Real Data-
################
Entity_name: {entity_name}
Description: {description}
Search Results:
{search_results}
################
Output:
"""

EXAMPLES = [
    """Example 1:
################
Entity_name: Java
Description: Java is a high-level programming language developed by Sun Microsystems. It is used to create web applications, mobile applications, and enterprise software.
Search Results:
1. Java (programming language)
2. Java (island)
3. Java (software platform)
4. Java (drink)
5. Java (disambiguation)
6. None of the above
################
Output:
1
################""",
    """Example 2:
################
Entity_name: Apple
Description: Apple Inc. is an American multinational technology company that designs, manufactures, and sells consumer electronics, computer software, and online services.
Search Results:
1. Apple (fruit)
2. Apple Inc.
3. Apple (disambiguation)
4. None of the above
################
Output:
2
################""",
]

SEARCH_JUDGEMENT_PROMPT = {
    "TEMPLATE": TEMPLATE,
    "EXAMPLES": EXAMPLES,
}
