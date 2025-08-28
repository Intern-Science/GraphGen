TEMPLATE_ZH: str = """请根据参考文本识别并消解文本中的指代词，明确每个代词所指代的具体实体，并直接输出消解后的文本。

-示例-
输入：
小明和小红一起去公园。她们玩得很开心。之后，他们去吃冰淇淋。
输出：
小明和小红一起去公园。小明和小红玩得很开心。之后，小明和小红去吃冰淇淋。

-真实数据-
参考文本：
{reference}
输入：
{input_sentence}
请直接输出改写后的句子，不要输出任何额外信息。
输出：
"""

TEMPLATE_EN: str = """Please identify and resolve the pronouns in the reference text, \
specify the specific entities referred to by each pronoun, and directly output the resolved text.

-Example-
Input:
John and Mary went to the park. They had a great time. Later, they went to eat ice cream.
Output:
John and Mary went to the park. John and Mary had a great time. Later, John and Mary went to eat ice cream.

-Real Data-
Reference text:
{reference}
Input:
{input_sentence}
Please directly output the rewritten sentence without any additional information.
Output:
"""

COREFERENCE_RESOLUTION_PROMPT = {"en": TEMPLATE_EN, "zh": TEMPLATE_ZH}
