# pylint: disable=C0301
TEMPLATE_SINGLE_EN: str = """The answer to a question is provided. Please generate a question that corresponds to the answer.

################
Answer:
{answer}
################
Question:
"""

TEMPLATE_SINGLE_ZH: str = """下面提供了一个问题的答案，请生成一个与答案对应的问题。

################
答案：
{answer}
################
问题：
"""

TEMPLATE_SINGLE_QA_EN: str = """You are given a text passage. Your task is to generate a question and answer (QA) pair based on the content of that text.
The answer should be accurate and directly derived from the text. Make sure the QA pair is relevant to the main theme or important details of the given text. 
For example:
Question: What is the effect of overexpressing the BG1 gene on grain size and development?
Answer: Overexpression of the BG1 gene leads to significantly increased grain size, demonstrating its role in grain development.

Question: What role does TAC4 play in the gravitropism of rice shoots?
Answer: TAC4 is a key regulator of gravitropism in rice shoots, promoting the bending of shoots towards the gravity vector.

Here is the text passage you need to generate a QA pair for:
{doc}
"""

TEMPLATE_SINGLE_QA_ZH: str = """给定一个文本段落。你的任务是根据该文本的内容生成一个问答（QA）对。
答案应准确且直接从文本中得出。确保QA对与给定文本的主题或重要细节相关。
例如：
问题：过表达BG1基因对谷粒大小和发育有什么影响？
答案：BG1基因的过表达显著增加了谷粒大小，表明其在谷物发育中的作用。

问题：TAC4在水稻茎的重力性状中扮演什么角色？
答案：TAC4是水稻茎重力性状的关键调节因子，促进茎向重力矢量弯曲。

以下是你需要为其生成QA对的文本段落：
{doc}
"""

# TODO: 修改这里的prompt
TEMPLATE_MULTI_EN = """You are an assistant to help read a article and then rephrase it in a question answering format. The user will provide you with an article with its content. You need to generate a paraphrase of the same article in question and answer format with one tag of "Question: ..." followed by "Answer: ...". Remember to keep the meaning and every content of the article intact.

Here is the format you should follow for your response:
Question: <Question>
Answer: <Answer>

Here is the article you need to rephrase:
{doc}
"""

TEMPLATE_MULTI_ZH = """你是一位助手，帮助阅读一篇文章，然后以问答格式重述它。用户将为您提供一篇带有内容的文章。你需要以一个标签"问题：..."为开头，接着是"答案：..."，生成一篇与原文章相同的问答格式的重述。请确保保持文章的意义和每个内容不变。

以下是你应该遵循的响应格式：
问题： <问题>
答案： <答案>

以下是你需要重述的文章：
{doc}
"""

QUESTION_GENERATION_PROMPT = {
    "English": {
        "SINGLE_TEMPLATE": TEMPLATE_SINGLE_EN,
        "SINGLE_QA_TEMPLATE": TEMPLATE_SINGLE_QA_EN,
        "MULTI_TEMPLATE": TEMPLATE_MULTI_EN
    },
    "Chinese": {
        "SINGLE_TEMPLATE": TEMPLATE_SINGLE_ZH,
        "SINGLE_QA_TEMPLATE": TEMPLATE_SINGLE_QA_ZH,
        "MULTI_TEMPLATE": TEMPLATE_MULTI_ZH
    }
}
