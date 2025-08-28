from openai import OpenAI
import gradio as gr

def test_api_connection(api_base, api_key, model_name):
    client = OpenAI(api_key=api_key, base_url=api_base)
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1
        )
        if not response.choices or not response.choices[0].message:
            raise gr.Error(f"{model_name}: Invalid response from API")
        gr.Success(f"{model_name}: API connection successful")
    except Exception as e:
        raise gr.Error(f"{model_name}: API connection failed: {str(e)}")
