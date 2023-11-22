---
Task: TextGeneration
Tags:
  - TextGeneration
  - zephyr-7b
---

# Model-zephyr-7b-dvc

🔥🔥🔥 Deploy [zephyr-7b](https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha) model on [VDP](https://github.com/instill-ai/vdp).

This repository contains the Zephyr-7b Text Completion Generation Model in PyTorch format, managed using [DVC](https://dvc.org/).

```
{
    "task_inputs": [
        {
            "text_generation": {
                "prompt": "[{\"role\": \"system\", \"content\": \"You are a helpful assistant.\"},{\"role\": \"user\", \"content\": \"Who won the world series in 2020?\"},{\"role\": \"assistant\", \"content\": \"The Los Angeles Dodgers won the World Series in 2020.\"},{\"role\": \"user\", \"content\": \"Where was it played?\"}]",
                "max_new_tokens": "100",
                "temperature": "0.8",
                "top_k": "50",
                "random_seed": "0",
                "extra_params": "{\"top_p\": 0.8}"
            }
        }
    ]
}
```
