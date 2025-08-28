TEMPLATE_EN = """You are an NLP expert responsible for generating a comprehensive summary of the data provided below.
Given one entity or relationship, and a list of descriptions, all related to the same entity or relationship.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.
Use {language} as output language.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

TEMPLATE_ZH = """你是一个NLP专家，负责根据以下提供的数据生成综合摘要。
给定一个实体或关系，以及一系列描述，所有描述都与同一实体或关系相关。
请将所有这些描述整合成一个综合描述。确保包含所有描述中收集的信息。
如果提供的描述是矛盾的，请解决这些矛盾并提供一个连贯的总结。
确保以第三人称写作，并包含实体名称，以便我们有完整的上下文。
使用{language}作为输出语言。

#######
-数据-
实体：{entity_name}
描述列表：{description_list}
#######
输出：
"""


KG_SUMMARIZATION_PROMPT = {
    "Chinese": {
        "TEMPLATE": TEMPLATE_ZH
    },
    "English": {
        "TEMPLATE": TEMPLATE_EN
    },
    "FORMAT": {
        "language": "English",
        "tuple_delimiter": "<|>",
        "record_delimiter": "##",
        "completion_delimiter": "<|COMPLETE|>",
    },
}
