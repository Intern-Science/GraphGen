TEMPLATE_ZH = """根据给定的知识图谱原始信息及已生成的推理路径，产出一条符合模板要求、可直接用于下游训练或推理的 CoT 数据。\
CoT（Chain-of-Thought，思维链）指在回答复杂问题时，把中间推理步骤一步一步显式写出来，使推理过程透明、可追溯，而不是直接给出最终答案。

-输入格式-
[Entities:]
(实体名:实体描述)
...

[Relationships:]
(来源实体)-[关系描述]->(目标实体)
...

[Question and Reasoning Path:]
(问题)
(推理路径)

-输出要求-
1. 每一步只完成一个不可分割的子任务，并用自然语言衔接，但是要避免生硬的连接词。
2. 使用中文。
3. 不要使用有序列表或编号。
4. 请直接给出答案，不要生成无关信息。

-真实数据-
输入:
[Entities:]:
{entities}

[Relationships:]:
{relationships}

[Question:]:
{question}

[Reasoning_Template:]:
{reasoning_template}

输出：

"""

TEMPLATE_EN = """Given the raw knowledge graph information and the provided reasoning-path, \
produce one Chain-of-Thought (CoT) sample that strictly follows the template \
and can be directly used for downstream training or inference.
CoT (Chain-of-Thought) means that when answering a complex question, the intermediate reasoning steps are \
explicitly written out one by one, making the reasoning process transparent and traceable instead of giving \
only the final answer.

-Input Format-
[Entities:]:
(ENTITY_NAME: ENTITY_DESCRIPTION)
...

[Relationships:]:
(ENTITY_SOURCE)-[RELATIONSHIP_DESCRIPTION]->(ENTITY_TARGET)
...

[Question and Reasoning Path:]:
(QUESTION)
(REASONING_PATH)

-Output Requirements-
1. Each step completes a single, indivisible sub-task and is naturally connected, avoiding abrupt transition words.
2. Use English.
3. Do not use ordered lists or numbering.
4. Do not generate extraneous information, just provide the answer.

-Real Data-
Input:
[Entities:]:
{entities}

[Relationships:]:
{relationships}

[Question:]:
{question}

[Reasoning_Template:]:
{reasoning_template}

Output:
"""

COT_GENERATION_PROMPT = {
    "Chinese": {"TEMPLATE": TEMPLATE_ZH},
    "English": {"TEMPLATE": TEMPLATE_EN},
}
