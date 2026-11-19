from gandolf.llms.Qwen3_30B_A3B_Thinking_2507 import Qwen3_30B_A3B_Thinking_2507

def create_LLM_instance(model_name, url, concurrency, max_tokens, response_checker, n=1, apply_chat_template=False, timeout=1800.0):
    if model_name == "Qwen3-30B-A3B-Thinking-2507":
        return Qwen3_30B_A3B_Thinking_2507(url, concurrency, max_tokens, response_checker, n, timeout, apply_chat_template)
    else:
        raise ValueError(f"{model_name} is not supported")

