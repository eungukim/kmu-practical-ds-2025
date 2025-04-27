from tools import DescribeArgs, CorrelationArgs

def pydantic_to_openai_schema(model_cls):
    """Pydantic 모델 ➜ OpenAI function-tool JSON"""
    return {
        "type": "function",
        "function": {
            "name": model_cls.__name__.replace("Args", "").lower(),
            "description": model_cls.__doc__ or "",
            "parameters": model_cls.schema()
        }
    }

TOOLS = [
    pydantic_to_openai_schema(DescribeArgs),
    pydantic_to_openai_schema(CorrelationArgs),
]