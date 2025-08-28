TEMPLATE: str = """Please determine if the following statement is correct.

Note:
1. If the statement is correct, please reply with 'yes', otherwise reply with 'no'.
2. The answer should be either 'yes' or 'no', do not output any other content.

Statement:
{statement}
Judgement: """

STATEMENT_JUDGEMENT_PROMPT = {
    "TEMPLATE": TEMPLATE
}
