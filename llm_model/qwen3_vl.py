import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import pkg
from pkg.common import chunk_list
from pathlib import Path
from llm_model import LocalModelConfig, ModelInfo
from llm_model.interface import LocalModelInterface
from pkg.result import Result
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from typing import Any


class QwenTransformersModel(LocalModelInterface):
    def __init__(self, config: LocalModelConfig):
        model_path = config.model_path
        if not os.path.exists(model_path):
            model_path = os.path.join(pkg.ModelDir, "qwen", "Qwen3-VL-4B-Instruct")

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype="auto",
            device_map="auto"
        )
        self.model_path = model_path
        self.max_output_tokens = config.max_output_tokens
        self.temperature = config.temperature
        self.executor = ThreadPoolExecutor(max_workers=config.workers)
        self.infer_semaphore = asyncio.Semaphore(config.workers)

    def preprocess_image(self, prompt: str | None = None, images: list[str] | None = None) -> list[dict[str, Any]]:
        content = []

        if images is not None and len(images) > 0:
            for image in images:
                content.append(
                    {
                        "type": "image",
                        "image": image,
                    }
                )

        if prompt is not None and len(prompt) > 0:
            content.append({"type": "text", "text": prompt})

        return self.generate_message(content=content)

    @staticmethod
    def generate_message(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": content
            }
        ]

    def inference(self, messages: list[list[dict[str, Any]]]) -> list[Result]:
        model = self.model
        processor = AutoProcessor.from_pretrained(
            self.model_path
        )

        inputs = processor.apply_chat_template(messages,
                                               tokenize=True,
                                               add_generation_prompt=True,
                                               return_dict=True,
                                               return_tensors="pt",
                                               padding=True,
                                               truncation=True
                                               )

        inputs = inputs.to(model.device)

        do_sample = True
        temperature = self.temperature
        if temperature <= 0:
            do_sample = False
            temperature = 0.1

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=self.max_output_tokens,
            temperature=temperature,
            do_sample=do_sample
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        input_token_counts = inputs.attention_mask.sum(dim=1).tolist()
        output_token_counts = [len(out_ids) for out_ids in generated_ids_trimmed]
        model_version = Path(self.model_path).name

        results: list[Result] = []
        for input_token, output_token, content in zip(input_token_counts, output_token_counts, output_text):
            model_info = ModelInfo(
                input_tokens=input_token,
                output_tokens=output_token,
                model_version=model_version,
                content=content
            )
            result = Result(
                success=True,
                result=model_info
            )
            results.append(result)

        return results

    async def handle_item(self, messages: list[list[dict[str, Any]]]) -> list[Result]:
        loop = asyncio.get_running_loop()

        async with self.infer_semaphore:
            results = await loop.run_in_executor(
                self.executor,
                self.inference,
                messages,
            )

        return results

    async def request_vllm(self, messages: list[list[dict[str, Any]]]) -> list[Result]:
        if not messages:
            return []

        tasks = []

        for batch in chunk_list(messages):
            task = asyncio.create_task(self.handle_item(batch))
            tasks.append(task)

        list_data = await asyncio.gather(*tasks)
        return [x for r in list_data for x in r]
