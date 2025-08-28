TEMPLATE_ZH = """你是一位“元推理架构师”。你的任务不是回答问题，\
而是根据给定的知识图谱中的实体和关系的名称以及描述信息，设计一条可复用、可泛化的 CoT 推理路径模板。\

-步骤-
1. 实体识别
- 准确地识别[Entities:]章节中的实体信息，包括实体名、实体描述信息。
- 实体信息的一般格式为:
(实体名:实体描述)

2. 关系识别
- 准确地识别[Relationships:]章节中的关系信息，包括来源实体名、目标实体名、关系描述信息。
- 关系信息的一般格式为:
(来源实体名)-[关系描述]->(目标实体名)

3. 图结构理解
- 正确地将关系信息中的来源实体名与实体信息关联。
- 根据提供的关系信息还原出图结构。

4. 问题设计
- 围绕知识图谱所表达的“核心主题”设计一个问题。
- 问题必须能在图谱内部通过实体、关系或属性直接验证；避免主观判断。
- 问题应该能够模型足够的思考，充分利用图谱中的实体和关系，避免过于简单或无关的问题。

5. 推理路径生成
- 根据问题设计一个**可被后续模型直接执行的推理蓝图**。
- 保持步骤最小化：每一步只解决一个“不可分割”的子问题。 

-约束条件-
1. 不要在回答中描述你的思考过程，直接给出回复，只给出问题和推理路径设计，不要生成无关信息。
2. 如果提供的描述信息相互矛盾，请解决矛盾并提供一个单一、连贯的逻辑。
3. 避免使用停用词和过于常见的词汇。
4. 不要出现具体数值或结论，不要出现“识别实体”、“识别关系”这类无意义的操作描述。
5. 使用中文作为输出语言。
6. 输出格式为：
问题：
推理路径设计：

-真实数据-
输入:
[Entities:]:
{entities}

[Relationships:]:
{relationships}

输出:
"""


TEMPLATE_EN = """You are a “meta-reasoning architect”. \
Your task is NOT to answer the question, but to design a reusable, generalizable CoT reasoning-path \
template based solely on the names and descriptions of entities and \
relationships in the provided knowledge graph.

- Steps -
1. Entity Recognition
- Accurately recognize entity information in the [Entities:] section, including entity names and descriptions.
- The general formats for entity information are:
(ENTITY_NAME: ENTITY_DESCRIPTION)

2. Relationship Recognition
- Accurately recognize relationship information in the [Relationships:] section, including source_entity_name, target_entity_name, and relationship descriptions.
- The general formats for relationship information are:
(SOURCE_ENTITY_NAME)-[RELATIONSHIP_DESCRIPTION]->(TARGET_ENTITY_NAME)

3. Graph Structure Understanding
- Correctly associate the source entity name in the relationship information with the entity information.
- Reconstruct the graph structure based on the provided relationship information.

4. Question Design
- Design a question around the "core theme" expressed by the knowledge graph.
- The question must be verifiable directly within the graph through entities, relationships, or attributes; avoid subjective judgments.
- The question should allow the model to think sufficiently, fully utilizing the entities and relationships in the graph, avoiding overly simple or irrelevant questions.

5. Reasoning-Path Design 
- Output a **blueprint that any later model can directly execute**.
- Keep steps minimal: each step solves one indivisible sub-problem.


- Constraints -
1. Do NOT describe your thinking; output only the reasoning-path design.
2. If the provided descriptions are contradictory, resolve conflicts and provide a single coherent logic.
3. Avoid using stop words and overly common words.
4. Do not include specific numerical values or conclusions, \
and DO NOT describing meaningless operations like "Identify the entity" or "Identify the relationship".
5. Use English as the output language.
6. The output format is:
Question:
Reasoning-Path Design:

Please summarize the information expressed by the knowledge graph based on the following [Entities:] and [Relationships:] provided.

- Real Data -
Input:
[Entities:]:
{entities}

[Relationships:]:
{relationships}

Output:
"""

COT_TEMPLATE_DESIGN_PROMPT = {
    "Chinese": {"TEMPLATE": TEMPLATE_ZH},
    "English": {"TEMPLATE": TEMPLATE_EN},
}
