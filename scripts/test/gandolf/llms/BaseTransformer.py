from typing import Any
import asyncio
import httpx
import copy


class BaseTransformer:

    def __init__(self, n, url, concurrency, payload, response_checker, timeout, apply_chat_template):
        self.n = n
        self.payload = payload
        self.url = url
        self.concurrency = concurrency
        self.response_checker = response_checker
        self.timeout = timeout
        self.apply_chat_template = apply_chat_template

    def inference(self, text_inputs) -> Any:
        return asyncio.run(self.send_requests(text_inputs))

    def set_n(self, n):
        self.n = n

    async def send_one_request(self, client, prompt):
        payload = copy.deepcopy(self.payload)  # deep copy, avoiding concurrency issues
        if self.apply_chat_template:
            payload["messages"] = [{"role": "user", "content": prompt}]
        else:
            payload["prompt"] = prompt
        payload["n"] = self.n  # support sampling multiple times
        r = await client.post(self.url, json=payload)
        return r.json()

    def process_response(self, res):
        """
        Return:
            n=1:
                str | None

            n>1:
                List[str | None]
        """

        if res is None:
            return None if self.n == 1 else [None] * self.n

        if "choices" not in res:
            return None if self.n == 1 else [None] * self.n

        if len(res["choices"]) != self.n:
            return None if self.n == 1 else [None] * self.n

        outputs = []
        for choice in res["choices"]:
            try:
                if choice["finish_reason"] == "stop":
                    content = choice["message"]["content"] if self.apply_chat_template else choice["text"]
                    if self.response_checker(content):
                        outputs.append(content)
                    else:
                        outputs.append(None)
                else:
                    outputs.append(None)

            except Exception:
                outputs.append(None)

        if self.n == 1:
            return outputs[0]
        else:
            return outputs

    async def send_requests(self, prompts):
        timeout = httpx.Timeout(self.timeout)

        results = [None] * len(prompts)  # keep the prompt order the same as that of responses

        async with httpx.AsyncClient(timeout=timeout) as client:

            async def worker(prompt, idx):
                raw_res = await self.send_one_request(client, prompt)
                processed_res = self.process_response(raw_res)
                return idx, processed_res

            tasks = set()

            for i in range(min(self.concurrency, len(prompts))):
                task = asyncio.create_task(worker(prompts[i], i))
                tasks.add(task)

            next_idx = self.concurrency

            while tasks:
                done, pending = await asyncio.wait(
                    tasks,
                    return_when=asyncio.FIRST_COMPLETED
                )

                for d in done:
                    idx, res = await d
                    results[idx] = res

                    if next_idx < len(prompts):
                        new_task = asyncio.create_task(
                            worker(prompts[next_idx], next_idx)
                        )
                        pending.add(new_task)
                        next_idx += 1

                tasks = pending

        return results

