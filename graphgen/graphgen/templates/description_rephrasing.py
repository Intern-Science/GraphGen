ANTI_TEMPLATE_EN: str = """-Goal-
Transform the input sentence into its opposite meaning while:

1. Preserving most of the original sentence structure
2. Changing only key words that affect the core meaning
3. Maintaining the same tone and style
4. The input sentence provided is a right description, and the output sentence should be a wrong description
5. The output sentence should be fluent and grammatically correct

################
-Examples-
################
Input:
The bright sunshine made everyone feel energetic and happy.

Output:
The bright sunshine made everyone feel tired and sad.

################
-Real Data-
################
Input: 
{input_sentence}
################
Please directly output the rewritten sentence without any additional information.
Output:
"""

ANTI_TEMPLATE_ZH: str = """-目标-
将输入句子转换为相反含义的句子，同时：

1. 保留大部分原始句子结构
2. 仅更改影响核心含义的关键词
3. 保持相同的语气和风格
4. 提供的输入句子是一个正确的描述，输出句子应该是一个错误的描述
5. 输出句子应该流畅且语法正确

################
-示例-
################
输入：
明亮的阳光让每个人都感到充满活力和快乐。

输出：
明亮的阳光让每个人都感到疲惫和悲伤。

################
-真实数据-
################
输入：
{input_sentence}
################
请直接输出改写后的句子，不要输出任何额外信息。
输出：
"""

TEMPLATE_ZH: str = """-目标-
将输入句子转换为相同含义的句子，同时：

1. 保留大部分原始句子结构
2. 仅更改影响核心含义的关键词
3. 保持相同的语气和风格
4. 输出句子应该流畅且语法正确

################
-示例-
################
输入：
明亮的阳光让每个人都感到充满活力和快乐。

输出：
明媚的阳光让每个人都感受到活力与快乐。

################
-真实数据-
################
输入：
{input_sentence}
################
请直接输出改写后的句子，不要输出任何额外信息。
输出：
"""

TEMPLATE_EN: str = """-Goal-
Transform the input sentence into a sentence with the same meaning while:

1. Preserving most of the original sentence structure
2. Changing only key words that affect the core meaning
3. Maintaining the same tone and style
4. The output sentence should be fluent and grammatically correct

################
-Examples-
################
Input:
The bright sunshine made everyone feel energetic and happy.

Output:
The bright sunshine made everyone feel energetic and joyful.

################
-Real Data-
################
Input:
{input_sentence}
################
Please directly output the rewritten sentence without any additional information.
Output:
"""


DESCRIPTION_REPHRASING_PROMPT= {
    "English": {
        "ANTI_TEMPLATE": ANTI_TEMPLATE_EN,
        "TEMPLATE": TEMPLATE_EN
    },
    "Chinese": {
        "ANTI_TEMPLATE": ANTI_TEMPLATE_ZH,
        "TEMPLATE": TEMPLATE_ZH
    }
}
