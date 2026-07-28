import easyocr
from typing import Literal, Any
from pydantic import Field
import os
import pkg
import asyncio
from concurrent.futures import ThreadPoolExecutor


class EasyOcr:
    def __init__(self,
                 lang_list: list[str] | None = None,
                 device: Literal["cpu", "cuda:0"] = "cpu",
                 max_workers: int = 4):
        if lang_list is None:
            lang_list = ['ch_sim', 'en']

        open_gpu = False if device == "cpu" else True
        model_path = os.path.join(pkg.ModelDir, "easyocr")
        self.reader = easyocr.Reader(
            lang_list=lang_list,
            gpu=open_gpu,
            model_storage_directory=model_path
        )
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.infer_semaphore = asyncio.Semaphore(max_workers)

    def inference(self, item: str, detail=1, paragraph=False) -> list[Any]:
        result = self.reader.readtext(
            item,
            detail=detail,
            paragraph=paragraph,
            batch_size=2
        )
        return result

    async def handle_item(self, item: str, detail=1, paragraph=False) -> list[Any]:

        loop = asyncio.get_running_loop()

        async with self.infer_semaphore:
            result = await loop.run_in_executor(
                self.executor,
                self.inference,
                item,
                detail,
                paragraph
            )

        return result

    async def advanced_recognition(self, images: list[str], detail=1, paragraph=False) -> list[Any]:
        if not images:
            return []

        tasks = []

        for img in images:
            task = asyncio.create_task(self.handle_item(item=img, detail=detail, paragraph=paragraph))
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return results

    async def simple_recognition(self, images: list[str]) -> list[Any]:
        return await self.advanced_recognition(images, detail=0)

    async def paragraph_recognition(self, images: list[str]) -> list[Any]:
        return await self.advanced_recognition(images, detail=0, paragraph=True)
