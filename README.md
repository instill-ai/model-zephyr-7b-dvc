---
Task: TextGenerationChat
Tags:
  - TextGenerationChat
  - zephyr-7b
---

# Model-zephyr-7b-dvc

🔥🔥🔥 Deploy [zephyr-7b](https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha) model on [VDP](https://github.com/instill-ai/vdp).

This repository contains the Zephyr-7b Text Generation Chat Model in PyTorch format, managed using [DVC v2.34.2](https://dvc.org/).

Notes:

- Disk Space Requirements: 14G
- Memory Requirements: 44G (for fp32 in cpu mode)

```json
{
    "task_inputs": [
        {
            "text_generation_chat": {
                "prompt": "What is your name?",
                "chat_history": [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": "Your name is Tony. A helpful assistant."
                            }
                        ]
                    }
                    ,{
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "I like to call you Toro."
                            }
                        ]
                    },
                    {
                        "role": "ASSISTANT",
                        "content": [
                            {
                                "type": "text",
                                "text": "Ok, My name is Toro now. What can I help you?"
                            }
                        ]
                    }
                ],
                // "system_message": "You are not a human.", // You can use either chat_history or system_message
                "max_new_tokens": "100",
                "temperature": "0.8",
                "top_k": "10",
                "seed": "42"
                // ,"extra_params": {
                //     "test_param_string": "test_param_string_value",
                //     "test_param_int": 123,
                //     "test_param_float": 0.2,
                //     "test_param_arr": [1, 2, 3],
                //     "test_param_onject": { "some_key": "some_value" }
                // }
            }
        }
    ]
}
```
