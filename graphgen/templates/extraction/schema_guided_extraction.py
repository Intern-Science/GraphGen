TEMPLATE_EN = """You are an expert at extracting information from text based on a given schema.
Extract relevant information about {field} from a given contract document according to the provided schema.

Instructions:
1. Carefully read the entire document provided at the end of this prompt.
2. Extract the relevant information.
3. Present your findings in JSON format as specified below.

Important Notes:
- Extract only relevant information. 
- Consider the context of the entire document when determining relevance.
- Do not be verbose, only respond with the correct format and information.
- Some docs may have multiple relevant excerpts -- include all that apply.
- Some questions may have no relevant excerpts -- just return "".
- Do not include additional JSON keys beyond the ones listed here.
- Do not include the same key multiple times in the JSON.
- Use English for your response.

Expected JSON keys and explanation of what they are:
{schema_explanation}

Expected format:
{{
    "key1": "value1",
    "key2": "value2",
    ...
}}

{examples}

Document to extract from:
{text}
"""

TEMPLATE_ZH = """你是一个擅长根据给定的模式从文本中提取信息的专家。
根据提供的模式，从合同文件中提取与{field}相关的信息。
操作说明：
1. 仔细阅读本提示末尾提供的整份文件。
2. 提取相关信息。
3. 按照下面指定的JSON格式呈现你的发现。

重要注意事项：
- 仅提取相关信息。
- 在确定相关性时，考虑整份文件的上下文。
- 不要冗长，只需以正确的格式和信息进行回应。
- 有些文件可能有多个相关摘录——请包含所有适用的内容。
- 有些问题可能没有相关摘录——只需返回""。
- 不要在JSON中包含除列出的键之外的其他键。
- 不要多次包含同一个键。
- 使用中文回答。

预期的JSON键及其说明：
{schema_explanation}

预期格式：
{{
    "key1": "value1",
    "key2": "value2",
    ...
}}

{examples}
要提取的文件：
{text}
"""

SCHEMA_GUIDED_EXTRACTION_PROMPT = {
    "en": TEMPLATE_EN,
    "zh": TEMPLATE_ZH,
}
