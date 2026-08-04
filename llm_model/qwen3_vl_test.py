from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import os
import pkg
import unittest


class TestLayoutModel(unittest.TestCase):
    def test_qwen3_vl(self):
        messages = [
            [{"role": "user", "content": [
                {
                    "type": "image",
                    "image": "/Users/xuanjinliang/PycharmProjects/easy_to_markdown/pdf_temp/aws_2024_cdn_24083b34-766a-48ad-9cdc-851744b1085c/crops_img/page_1/2_doc_title.webp",
                    "resized_height": 51,
                    "resized_width": 361
                },
                {"type": "text", "text": "Describe this image."}]
              }]
        ]

        model_path = os.path.join(pkg.ModelDir, "qwen_vl", "qwen3_vl")

        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3-VL-4B-Instruct",
            cache_dir=model_path
        )
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-4B-Instruct",
            dtype="auto",
            device_map="auto"
        )

        inputs = processor.apply_chat_template(messages,
                                               tokenize=True,
                                               add_generation_prompt=True,
                                               return_dict=True,
                                               return_tensors="pt")
        inputs = inputs.to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=8192)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text)
