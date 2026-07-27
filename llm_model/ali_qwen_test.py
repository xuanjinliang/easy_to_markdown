import unittest
import os
import pkg
import uuid
import re
import json
from pathlib import Path
from pkg.image_handle import image_to_seed
from pkg.common import remove_fenced_code_block
from llm_model import logger, ClientConfig
from llm_model.ali_qwen import QwenClient, QwenOcrDashscope
from pydantic import BaseModel
from generate import AdvancedOCRVL
import logging

logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.DEBUG)


def extract_markdown(text: str) -> str | None:
    pattern = r"```markdown\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return text
    return match.group(1).strip()


def extract_json(text: str) -> str | None:
    pattern = r"```json\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return text
    return match.group(1).strip()


class NeedToRetryMD(BaseModel):
    position: str
    score: float
    error_txt: str


class TestQwenPI(unittest.IsolatedAsyncioTestCase):
    async def test_analysis_pdf_text(self):
        # image_path = Path(os.path.join(pkg.PdfTempDir, 'test', 'crops_img', 'paddleocr_vl_demo', "3_image.webp"))

        image_path = Path(
            os.path.join(pkg.PdfTempDir,
                         "aws_2024_cdn_24083b34-766a-48ad-9cdc-851744b1085c/crops_img/page_1/5_table/table_crop/row_0_1.webp"
                         ))

        # image_path = Path(
        #     os.path.join(pkg.PdfTempDir,
        #                  'aws_2024_cdn_24083b34-766a-48ad-9cdc-851744b1085c',
        #                  'crops_img',
        #                  'page_3',
        #                  "3_table",
        #                  "table_crop",
        #                  "row_7_1.webp"
        #                  ))

        config = ClientConfig(model="qwen3.5-ocr")
        client = QwenOcrDashscope(config)

        png_files = [image_path]

        for file_path in png_files:
            messages = client.generate_base64(enable_rotate=False, images=[str(file_path)])
            result = await client.request_llm(
                messages=messages,
                task="advanced_recognition"
            )

            self.assertIsNotNone(result)

            text = result.result.get("text", "")
            # is_json = isinstance(text, dict) or isinstance(text, list)
            #
            # data = []
            # if is_json:
            #     data = text
            # else:
            #     text = remove_fenced_code_block(text)
            #     data = json.loads(text)
            #
            # results = [AdvancedOCRVL.model_validate(item) for item in data]

            print("result_text --> ", text)
            print("result_input_tokens --> ", result.result.get("input_tokens", 0))
            print("result_output_tokens --> ", result.result.get("output_tokens", 0))

    async def test_analysis_table(self):
        # image_path = Path(
        #     os.path.join(pkg.PdfTempDir,
        #                  'aws_2023_ba386fee-02ce-4f61-85d4-5e85926ce159',
        #                  'crops_img',
        #                  'page_2',
        #                  "4_table.webp"))
        image_path = Path(
            os.path.join(pkg.PdfTempDir,
                         'aws_2023_ba386fee-02ce-4f61-85d4-5e85926ce159',
                         'crops_img',
                         'page_2',
                         "4_table",
                         "table_crop",
                         "row_8_1.webp"
                         ))
        config = ClientConfig(model="qwen3.5-ocr")
        client = QwenOcrDashscope(config)

        png_files = [image_path]

        for file_path in png_files:
            messages = client.generate_base64(enable_rotate=False, images=[str(file_path)])
            result = await client.request_llm(
                messages=messages,
                task="advanced_recognition"
            )

            self.assertIsNotNone(result)

            if result.success:
                text = result.result.get("text", "")
                print("result_text --> ", text)
                print("result_input_tokens --> ", result.result.get("input_tokens", 0))
                print("result_output_tokens --> ", result.result.get("output_tokens", 0))

            logger.error(result.error)

    async def test_QwenClient(self):
        image_path = Path(
            os.path.join(pkg.PdfTempDir,
                         'aws_2023_ba386fee-02ce-4f61-85d4-5e85926ce159',
                         'crops_img',
                         'page_2',
                         "4_table",
                         "table_crop",
                         "row_8_1.webp"
                         ))

        path_obj = os.path.join(pkg.PromptDir, "table_layout.md")
        system_info = Path(path_obj).read_text(encoding="utf-8")

        config = ClientConfig(model="qwen3.6-flash",
                              max_output_tokens=8192,
                              thinking_budget=0,
                              temperature=0,
                              model_client_stream=True,
                              )
        qwen_client = QwenClient(config)

        png_files = [image_path]

        for file_path in png_files:
            messages = qwen_client.generate_image(images=[str(file_path)])
            result = await qwen_client.request_llm(
                system=system_info,
                messages=messages,
                seed=image_to_seed(str(file_path)),
            )

            self.assertIsNotNone(result)

            if result.success:
                text = result.result.get("text", "")
                print("result_text --> ", text)
                print("result_input_tokens --> ", result.result.get("input_tokens", 0))
                print("result_output_tokens --> ", result.result.get("output_tokens", 0))

            logger.error(result.error)

    def test_generate_batch_json(self):
        output_dir = os.path.join(pkg.BatchJsonDir)
        os.makedirs(output_dir, exist_ok=True)

        image_path = Path(os.path.join(pkg.PdfImageDir, '2f101d2e-a08d-4870-bb1c-01b5c6d0e6b0'))
        txt_path = Path(os.path.join(pkg.PdfOcrDir, 'a61320a2-e58e-4e64-a1c3-7337d602387e'))

        txt_files = sorted(txt_path.glob("page_*.txt"), key=lambda p: int(re.search(r'\d+', p.stem).group()))
        path_obj = os.path.join(pkg.PromptDir, "system", "reconstruction_assistant_1.md")
        system_info = Path(path_obj).read_text(encoding="utf-8")

        config = ClientConfig(model="qwen3-max",
                              max_output_tokens=8192 * 4,
                              thinking_budget=10000,
                              temperature=0,
                              model_client_stream=True
                              )
        qwen_client = QwenClient(config)

        # json_files = json_files[:1]

        for txt_file in txt_files:
            ocr_txt = Path(txt_file).read_text(encoding="utf-8")

            image_file = os.path.join(image_path, txt_file.stem + ".png")

            seed = image_to_seed(image_file)

            messages = qwen_client.generate_base64(
                prompt=f"[Ocr Content]\n{ocr_txt}\n",
                images=[image_file]
            )

            batch_json = qwen_client.generate_batch_json(
                key=image_file,
                system=system_info,
                messages=messages,
                seed=seed,
            )
            qwen_client.write_json_to_file(json_data=batch_json)

        batch_json_path_list = qwen_client.get_batch_file_path()
        print(f"batch_json_path_list-->{batch_json_path_list}")
        batch_id_list = []
        for path_str in batch_json_path_list:
            batch_id = qwen_client.upload_batch_json(Path(path_str))
            batch_id_list.append(batch_id)

        print(f"batch_id_list -> {batch_id_list}")

    def test_check_batch_task(self):
        config = ClientConfig(model="qwen3-max")
        client = QwenClient(config)

        batch_task_data = client.check_batch_task(batch_id="batch_28b9f81f-bc6e-41a0-879a-a774178f2e1f")
        print("batch_task_data-->", batch_task_data)
        if batch_task_data.get("status", "") == "completed":
            output_dir = os.path.join(pkg.PdfMarkdownDir, str(uuid.uuid4()))
            os.makedirs(output_dir, exist_ok=True)

            file_id = batch_task_data.get("output_file_id", "")
            content = client.get_batch_result(file_id)
            json_list = [json.loads(line) for line in content.splitlines()]
            print("json_list-->", json_list)
            for item in json_list:
                custom_id = item.get("custom_id", "")

                filename = Path(custom_id)

                error = item.get("error", None)
                if error is not None:
                    print("error --> ", error)
                    continue
                print(f"custom_id:{custom_id}")

                response = item.get("response", None)
                if response is None:
                    continue

                body = response.get("body", None)
                if body is None:
                    continue

                usage = body.get("usage", None)
                if usage is not None:
                    print(
                        {
                            "input_tokens": usage.get("prompt_tokens", 0),
                            "output_tokens": usage.get("completion_tokens", 0),
                        }
                    )

                choices = body.get("choices", None)
                if choices is not None:
                    messages = choices[0].get("message", None)
                    if messages is None:
                        continue
                    content = messages.get("content", None)
                    if content is None:
                        continue

                    with open(os.path.join(output_dir, filename.stem + ".md"), "w",
                              encoding="utf-8") as f:
                        f.write(content)

    async def test_partition_group(self):
        output_dir = os.path.join(pkg.PdfOcrDir, str(uuid.uuid4()))
        os.makedirs(output_dir, exist_ok=True)

        path_obj = os.path.join(pkg.PromptDir, "partition_group.md")
        system_info = Path(path_obj).read_text(encoding="utf-8")

        image_path = Path(os.path.join(pkg.PdfImageDir, 'aws_2023'))
        webp_files = sorted(image_path.glob("page_*.webp"), key=lambda p: int(re.search(r'\d+', p.stem).group()))
        config = ClientConfig(
            # model="qwen3-vl-plus",
            # max_output_tokens=8192 * 4,
            model="qwen-vl-ocr",
            thinking_budget=0,
            # text_format=PartitionGroupList,
            # extra_args={"task": "advanced_recognition"},
            # extra_args={"vl_high_resolution_images": True},
        )
        client = QwenClient(config)

        webp_files = [webp_files[1]]

        for file_path in webp_files:
            messages = client.generate_image(images=[str(file_path)])
            result = await client.request_llm(
                system=system_info,
                messages=messages,
                seed=image_to_seed(str(file_path)),
            )

            self.assertIsNotNone(result)

            if result.success:
                text = result.result.get("text", "")
                print("result_text --> ", text)
                print("result_input_tokens --> ", result.result.get("input_tokens", 0))
                print("result_output_tokens --> ", result.result.get("output_tokens", 0))
                with open(os.path.join(output_dir, file_path.stem + ".json"), "w", encoding="utf-8") as f:
                    json.dump(json.loads(text), f, ensure_ascii=False)
            else:
                print(result.error)

    async def test_partition_group2(self):
        output_dir = os.path.join(pkg.PdfOcrDir, str(uuid.uuid4()))
        os.makedirs(output_dir, exist_ok=True)

        path_obj = os.path.join(pkg.PromptDir, "partition_group.md")
        system_info = Path(path_obj).read_text(encoding="utf-8")

        image_path = Path(os.path.join(pkg.PdfImageDir, '84c5beaa-42db-48d7-9d63-681c68c71b96'))
        webp_files = sorted(image_path.glob("page_*.webp"), key=lambda p: int(re.search(r'\d+', p.stem).group()))
        config = ClientConfig(
            model="qwen3-vl-plus",
            max_output_tokens=8192 * 4,
            thinking_budget=5000,
        )
        client = QwenClient(config)

        webp_files = [webp_files[1]]

        for file_path in webp_files:
            messages = client.generate_image(prompt=" ", images=[str(file_path)])
            result = await client.request_llm(
                system=system_info,
                messages=messages,
                seed=image_to_seed(str(file_path)),
            )

            if result.success:
                text = result.result.get("text", "")
                print("result_text --> ", text)
                print("result_input_tokens --> ", result.result.get("input_tokens", 0))
                print("result_output_tokens --> ", result.result.get("output_tokens", 0))
                with open(os.path.join(output_dir, file_path.stem + ".json"), "w", encoding="utf-8") as f:
                    json.dump(json.loads(text), f, ensure_ascii=False)
            else:
                print(result.error)

    async def test_qwen3_vl(self):
        output_dir = os.path.join(pkg.PdfOcrDir, str(uuid.uuid4()))
        os.makedirs(output_dir, exist_ok=True)

        path_obj = os.path.join(pkg.PromptDir, "ocr_vl_assistant.md")
        system_info = Path(path_obj).read_text(encoding="utf-8")

        image_path = Path(os.path.join(pkg.PdfImageDir, '84c5beaa-42db-48d7-9d63-681c68c71b96'))
        webp_files = sorted(image_path.glob("page_*.webp"), key=lambda p: int(re.search(r'\d+', p.stem).group()))
        config = ClientConfig(
            model="qwen3-vl-plus",
            max_output_tokens=8192 * 4,
            thinking_budget=0,
            # text_format=ParsingResultList,
            extra_args={"vl_high_resolution_images": True},
        )
        client = QwenClient(config)

        webp_files = [webp_files[1]]

        for file_path in webp_files:
            messages = client.generate_image(images=[str(file_path)])
            result = await client.request_llm(
                system=system_info,
                messages=messages,
                seed=image_to_seed(str(file_path)),
            )

            self.assertIsNotNone(result)

            if result.success:
                text = result.result.get("text", "")
                print("result_text --> ", text)
                print("result_input_tokens --> ", result.result.get("input_tokens", 0))
                print("result_output_tokens --> ", result.result.get("output_tokens", 0))
                with open(os.path.join(output_dir, file_path.stem + ".json"), "w", encoding="utf-8") as f:
                    json.dump(json.loads(text), f, ensure_ascii=False)
            else:
                print(result.error)


class TestQwenAssistant(unittest.IsolatedAsyncioTestCase):
    def test_get_image_seed(self):
        image_path = Path(os.path.join(pkg.PdfImageDir, '2f101d2e-a08d-4870-bb1c-01b5c6d0e6b0'))
        png_files = sorted(image_path.glob("page_*.png"), key=lambda p: int(re.search(r'\d+', p.stem).group()))
        file_path = png_files[0]

        seed = image_to_seed(str(file_path))
        print("seed -->", seed)
