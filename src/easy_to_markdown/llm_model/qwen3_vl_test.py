from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from pydantic import BaseModel, Field
from easy_to_markdown.llm_model import LocalModelConfig
from easy_to_markdown.llm_model.qwen3_vl import QwenTransformersModel, QwenMlxModel
import os
from easy_to_markdown import pkg
import unittest


class Description(BaseModel):
    text: str


class TestLayoutModel(unittest.IsolatedAsyncioTestCase):
    def test_qwen3_vl(self):
        messages = [
            [
                {
                    "role": "system",
                    "content": """You are a professional **Document Understanding Assistant**.

# Task

1. Restore the natural reading order of the document according to human reading habits.
2. Correct obvious OCR recognition errors.
3. Preserve the original document structure.

---

# Input

You will receive:

1. **An image**
2. Text blocks detected by OCR, including:
   - Text content
   - Bounding box coordinates

---

# Processing Steps

Strictly follow the steps below:

1. Only recognize the text inside the bounding boxes located **below `__content__`** in the image.
   - Example:
     - ```
       __content__
       [aaaaaa]
       ```

2. Check whether the text extends beyond its bounding box.
   - **Yes**: Correct the OCR-recognized text accordingly.
   - **No**: Use the OCR text as-is.

# Output

Output **only the corrected text in the natural reading order**, without any additional explanation."""
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": os.path.join(pkg.PdfTempDir,
                                                  "aws_2023_ba386fee-02ce-4f61-85d4-5e85926ce159/llm_crops_img/page_1/1_header.webp"),
                        },
                        {"type": "text", "text": """[Ocr Content]
['DocuSign Envelope ID: EEC495D6-70A9-41AA-B7FE-BB8FCACCBCE2']

[Ocr bbox]
[[0.0, 0.0, 748.0, 54.0]]
"""}]
                },
            ],
        ]

        model_path = os.path.join(pkg.ModelDir, "qwen", "Qwen3-VL-4B-Instruct")

        processor = AutoProcessor.from_pretrained(
            model_path
        )
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype="auto",
            device_map="auto"
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

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=8192,
            do_sample=False
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        input_token_counts = inputs.attention_mask.sum(dim=1).tolist()
        output_token_counts = [len(out_ids) for out_ids in generated_ids_trimmed]
        print(f"output_token --> {output_token_counts}")
        print(f"input_token --> {input_token_counts}")
        # print(f"output_token --> {generated_ids.shape[1] - inputs.input_ids.shape[1]}")
        # print(f"input_token --> {inputs.input_ids.shape[1]}")
        print(f"result --> {output_text}")

    def test_qwen3_vl_mlx_vlm(self):
        model_path = os.path.join(pkg.ModelDir, "qwen_mlx", "Qwen3-VL-4B-Instruct-8bit")
        model, processor = load(model_path)
        config = model.config

        images = [
            os.path.join(pkg.PdfTempDir,
                         "aws_2024_cdn_24083b34-766a-48ad-9cdc-851744b1085c/crops_img/page_1/2_doc_title.webp")
        ]

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": os.path.join(pkg.PdfTempDir,
                                              "aws_2024_cdn_24083b34-766a-48ad-9cdc-851744b1085c/crops_img/page_1/2_doc_title.webp")
                    },
                    {
                        "type": "text",
                        "text": "Describe this image."
                    }
                ]
            }
        ]

        formatted_prompt = apply_chat_template(
            processor, config, messages, num_images=len(images)
        )

        output = generate(
            model=model,
            processor=processor,
            prompt=formatted_prompt,
            image=images,
            verbose=False,
            temperature=0,
            max_tokens=8192
        )
        print(f"output_token --> {output.generation_tokens}")
        print(f"input_token --> {output.prompt_tokens}")
        print(f"result --> {output.text}")

    async def test_qwen_transformers_model(self):
        model_path = os.path.join(pkg.ModelDir, "qwen", "Qwen3-VL-4B-Instruct")
        qwen_transformers_model = QwenTransformersModel(config=LocalModelConfig(
            model_path=model_path,
            temperature=0
        ))

        # schema = Description.model_json_schema()
        image_list = [
            os.path.join(pkg.PdfTempDir,
                         "aws_2024_cdn_24083b34-766a-48ad-9cdc-851744b1085c/crops_img/page_1/1_header.webp"),
            os.path.join(pkg.PdfTempDir,
                         "aws_2024_cdn_24083b34-766a-48ad-9cdc-851744b1085c/crops_img/page_1/2_doc_title.webp")
        ]

        input_list = []
        for image in image_list:
            input_prompt = qwen_transformers_model.preprocess_image(prompt="Describe this image.", images=[image])
            input_list.append(input_prompt)

        results = await qwen_transformers_model.request_vllm(messages=input_list)
        print(results)

    async def test_qwen_mlx_model(self):
        model_path = os.path.join(pkg.ModelDir, "qwen", "Qwen3-VL-4B-Instruct-8bit")
        qwen_mlx_model = QwenMlxModel(config=LocalModelConfig(
            model_path=model_path,
            temperature=0
        ))

        image_list = [
            os.path.join(pkg.PdfTempDir,
                         "aws_2024_cdn_24083b34-766a-48ad-9cdc-851744b1085c/crops_img/page_1/1_header.webp"),
            os.path.join(pkg.PdfTempDir,
                         "aws_2024_cdn_24083b34-766a-48ad-9cdc-851744b1085c/crops_img/page_1/2_doc_title.webp")
        ]

        input_list = []
        for image in image_list:
            input_prompt = qwen_mlx_model.preprocess_image(prompt="Describe this image.",
                                                           images=[image])
            input_list.append(input_prompt)

        results = await qwen_mlx_model.request_vllm(messages=input_list)
        print(results)

    async def test_qwen_mlx_model_schema(self):
        model_path = os.path.join(pkg.ModelDir, "qwen", "Qwen3-VL-4B-Instruct-8bit")
        qwen_mlx_model = QwenMlxModel(config=LocalModelConfig(
            model_path=model_path,
            temperature=0
        ))

        schema = Description.model_json_schema()
        image_list = [
            os.path.join(pkg.PdfTempDir,
                         "aws_2024_cdn_24083b34-766a-48ad-9cdc-851744b1085c/crops_img/page_1/1_header.webp"),
            os.path.join(pkg.PdfTempDir,
                         "aws_2024_cdn_24083b34-766a-48ad-9cdc-851744b1085c/crops_img/page_1/2_doc_title.webp")
        ]

        input_list = []
        for image in image_list:
            input_prompt = qwen_mlx_model.preprocess_image(prompt="Describe this image.",
                                                           images=[image])
            input_list.append(input_prompt)

        results = await qwen_mlx_model.request_vllm(messages=input_list, schema=schema)
        print(results)
