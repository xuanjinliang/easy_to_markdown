import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from easy_to_markdown import pkg
from pathlib import Path
from easy_to_markdown.llm_model import LocalModelConfig, ModelInfo
from easy_to_markdown.llm_model.interface import LocalModelInterface
from easy_to_markdown.pkg.result import Result
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from typing import Any

import logging
from logging import NullHandler

logger = logging.getLogger(__name__)
logger.addHandler(NullHandler())


class QwenTransformersWorker:
    def __init__(self, config: LocalModelConfig):
        model_path = config.model_path
        if not os.path.exists(model_path):
            model_path = os.path.join(pkg.ModelDir, "qwen", "Qwen3-VL-4B-Instruct")

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype="auto",
            device_map="auto"
        )

        self.processor = AutoProcessor.from_pretrained(
            model_path
        )

        self.model_path = model_path
        self.max_output_tokens = config.max_output_tokens
        self.temperature = config.temperature

    def inference(self, messages: list[list[dict[str, Any]]]) -> list[Result]:
        model = self.model
        inputs = self.processor.apply_chat_template(messages,
                                                    tokenize=True,
                                                    add_generation_prompt=True,
                                                    return_dict=True,
                                                    return_tensors="pt",
                                                    padding=True,
                                                    truncation=True
                                                    )

        inputs = inputs.to(model.device)

        generate_kwargs = {
            "max_new_tokens": self.max_output_tokens
        }

        if self.temperature > 0:
            generate_kwargs.update(
                {
                    "temperature": self.temperature,
                    "do_sample": True
                }
            )

        else:
            generate_kwargs.update(
                {
                    "do_sample": False
                }
            )

        generated_ids = model.generate(
            **inputs,
            **generate_kwargs
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
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


class QwenTransformersModel(LocalModelInterface):
    def __init__(self, config: LocalModelConfig):
        self.workers = []
        for i in range(config.workers):
            worker = QwenTransformersWorker(
                config=config,
            )

            self.workers.append(worker)

        self.executor = ThreadPoolExecutor(
            max_workers=1  # transformer doesn't support worker pools
        )

        self.worker_queue = asyncio.Queue()

        for idx in range(len(self.workers)):
            self.worker_queue.put_nowait(idx)

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
    def generate_message(content: list[dict[str, Any]] | str, role: str = "user") -> list[dict[str, Any]]:
        return [
            {
                "role": role,
                "content": content
            }
        ]

    def _run_worker(self, worker_id: int, messages: list[dict[str, Any]]) -> list[Result]:
        worker = self.workers[worker_id]
        return worker.inference(
            [messages]
        )

    async def handle_item(self, messages: list[dict[str, Any]]) -> list[Result]:
        worker_id = await self.worker_queue.get()

        try:
            loop = asyncio.get_running_loop()

            result = await loop.run_in_executor(
                self.executor,
                self._run_worker,
                worker_id,
                messages
            )

            return result
        except Exception as e:
            logger.error(e)
            return [Result(success=False, result=e)] * len(messages)
        finally:
            self.worker_queue.put_nowait(
                worker_id
            )

    async def request_vllm(self, messages: list[list[dict[str, Any]]]) -> list[Result]:
        if not messages:
            return []

        tasks = [
            asyncio.create_task(
                self.handle_item(msg)
            )
            for msg in messages
        ]

        list_data = await asyncio.gather(*tasks)
        return [x for r in list_data for x in r]


class QwenMlxWorker:
    def __init__(self, config: LocalModelConfig):
        model_path = config.model_path
        if not os.path.exists(config.model_path):
            model_path = os.path.join(pkg.ModelDir, "qwen_mlx", "Qwen3-VL-4B-Instruct-8bit")

        self.model_path = model_path
        self.model, self.processor = load(model_path)
        self.model_config = self.model.config

        self.max_output_tokens = config.max_output_tokens
        self.temperature = config.temperature

    def inference(self, messages: list[dict[str, Any]]) -> Result:
        image_list: list[str] = []
        for message in messages:
            content = message.get("content", None)
            if not isinstance(content, list):
                continue

            for item in content:
                if item.get("type", "") == "image" and len(item.get("image", "")) > 0:
                    image = item.get("image", None)
                    if isinstance(image, str):
                        image_list.append(image)
                        continue

                    elif isinstance(image, list):
                        image_list += image

        formatted_prompt = apply_chat_template(
            self.processor, self.model_config, messages, num_images=len(image_list)
        )

        output = generate(
            model=self.model,
            processor=self.processor,
            prompt=formatted_prompt,
            image=image_list,
            verbose=False,
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
        )

        model_version = Path(self.model_path).name

        model_info = ModelInfo(
            input_tokens=output.prompt_tokens,
            output_tokens=output.generation_tokens,
            model_version=model_version,
            content=output.text
        )
        result = Result(
            success=True,
            result=model_info
        )

        return result


class QwenMlxModel(LocalModelInterface):
    def __init__(self, config: LocalModelConfig):
        self.workers = []
        for i in range(config.workers):
            worker = QwenMlxWorker(
                config=config,
            )

            self.workers.append(worker)

        self.executor = ThreadPoolExecutor(
            max_workers=config.workers
        )

        self.worker_queue = asyncio.Queue()

        for idx in range(len(self.workers)):
            self.worker_queue.put_nowait(idx)

    def preprocess_image(self, prompt: str | None = None, images: list[str] | None = None) -> list[dict[str, Any]]:
        content = []

        if images is not None and len(images) > 0:
            content.append(
                {
                    "type": "image",
                    "image": images,
                }
            )

        if prompt is not None and len(prompt) > 0:
            content.append({"type": "text", "text": prompt})

        return self.generate_message(content=content)

    @staticmethod
    def generate_message(content: list[dict[str, Any]] | str, role: str = "user") -> list[dict[str, Any]]:
        return [
            {
                "role": role,
                "content": content
            }
        ]

    def _run_worker(self, worker_id: int, messages: list[dict[str, Any]]) -> Result:
        worker = self.workers[worker_id]
        return worker.inference(
            messages
        )

    async def handle_item(self, messages: list[dict[str, Any]]) -> Result:
        worker_id = await self.worker_queue.get()

        try:
            loop = asyncio.get_running_loop()

            result = await loop.run_in_executor(
                self.executor,
                self._run_worker,
                worker_id,
                messages
            )

            return result
        except Exception as e:
            logger.error(e)
            return Result(success=False, result=e)
        finally:
            self.worker_queue.put_nowait(
                worker_id
            )

    async def request_vllm(self, messages: list[list[dict[str, Any]]]) -> list[Result]:
        if not messages:
            return []

        tasks = [
            asyncio.create_task(
                self.handle_item(msg)
            )
            for msg in messages
        ]

        return await asyncio.gather(*tasks)
