from gandolf.llms.BaseTransformer import BaseTransformer

class Qwen3_30B_A3B_Thinking_2507(BaseTransformer):

    def __init__(self, url, concurrency, max_tokens, response_checker, n, timeout, apply_chat_template):
        super().__init__(
            n=n,
            url=url,
            concurrency=concurrency,
            payload={
                "model": "Qwen3-30B-A3B-Thinking-2507",
                "presence_penalty": 1.05,
                "temperature": 0.6,
                "top_p": 0.95,
                "min_p": 0.0,
                "top_k": 20,
                "max_tokens": max_tokens
            },
            response_checker=response_checker,
            timeout=timeout,
            apply_chat_template=apply_chat_template
        )

