# pylint: skip-file
import os
import random
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3" # TODO: Put in CPU mode

import time
import json
from pathlib import Path

import traceback

import numpy as np
import triton_python_backend_utils as pb_utils

import transformers
import torch

from conversation import (
    Conversation,
    conv_templates,
    SeparatorStyle
)

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    # Reference: https://docs.nvidia.com/launchpad/data-science/sentiment/latest/sentiment-triton-overview.html
    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
Implementing `initialize` function is optional. This function allows
the model to initialize any state associated with this model.
Parameters
----------
args : dict
Both keys and values are strings. The dictionary keys and values are:
* model_config: A JSON string containing the model configuration
* model_instance_kind: A string containing model instance kind
* model_instance_device_id: A string containing model instance device ID
* model_repository: Model repository path
* model_version: Model version
* model_name: Model name
"""
        self.logger = pb_utils.Logger
        self.model_config = json.loads(args["model_config"])

        # Load the model
        model_path = str(Path(__file__).parent.absolute().joinpath('zephyr-7b-alpha/'))

        self.pipe = transformers.pipeline(
            "text-generation", 
            model=model_path, 
            torch_dtype=torch.float32,  # if gpu mode turn to float16
            device_map="cpu" 
        )

        # Get output configurations
        output0_config = pb_utils.get_output_config_by_name(self.model_config, "text")
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):
        responses = []

        for request in requests:
            try:
                prompt = str(pb_utils.get_input_tensor_by_name(request, "prompt").as_numpy()[0].decode("utf-8"))
                print(f'[DEBUG] input `prompt` type({type(prompt)}): {prompt}')

                prompt_in_conversation = False
                try:
                    parsed_conversation = json.loads(prompt)
                    # turn in to converstation?

                    # using fixed roles
                    roles = ['USER', 'ASSISTANT']
                    # roles = ["<|im_start|>user\n", "<|im_start|>assistant\n"]
                    roles_lookup = {x: i for i, x in enumerate(roles)}

                    conv = None
                    for i, x in enumerate(parsed_conversation):
                        role = str(x['role']).upper()
                        print(f'[DEBUG] Message {i}: {role}: {x["content"]}')
                        if i == 0:
                            if role == 'SYSTEM':
                                conv = Conversation(
                                    system=f"<|im_start|>system\n{str(x['content'])}",
                                    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
                                    version="mpt",
                                    messages=[],
                                    offset=0,
                                    sep_style=SeparatorStyle.MPT,
                                    sep="<|im_end|>"
                                )
                            else:
                                conv = conv_templates["mpt"].copy()
                                conv.roles = tuple(roles)
                                conv.append_message(conv.roles[roles_lookup[role]], x['content'])
                        else:
                            conv.append_message(conv.roles[roles_lookup[role]], x['content'])
                    prompt_in_conversation = True
                except json.decoder.JSONDecodeError:
                    pass

                if not prompt_in_conversation:
                    conv = conv_templates["mpt"].copy()
                    conv.append_message(conv.roles[0], prompt)
                conv.append_message(conv.roles[1], None)
                extra_params_str = ""
                if pb_utils.get_input_tensor_by_name(request, "extra_params") is not None:
                    extra_params_str = str(pb_utils.get_input_tensor_by_name(request, "extra_params").as_numpy()[0].decode("utf-8"))
                print(f'[DEBUG] input `extra_params` type({type(extra_params_str)}): {extra_params_str}')

                extra_params = {}
                try:
                    extra_params = json.loads(extra_params_str)
                except json.decoder.JSONDecodeError:
                    pass
            
                max_new_tokens = 256
                if pb_utils.get_input_tensor_by_name(request, "max_new_tokens") is not None:
                    max_new_tokens = int(pb_utils.get_input_tensor_by_name(request, "max_new_tokens").as_numpy()[0])
                print(f'[DEBUG] input `max_new_tokens` type({type(max_new_tokens)}): {max_new_tokens}')

                top_k = 50
                if pb_utils.get_input_tensor_by_name(request, "top_k") is not None:
                    top_k = int(pb_utils.get_input_tensor_by_name(request, "top_k").as_numpy()[0])
                print(f'[DEBUG] input `top_k` type({type(top_k)}): {top_k}')

                temperature = 0.8
                if pb_utils.get_input_tensor_by_name(request, "temperature") is not None:
                    temperature = float(pb_utils.get_input_tensor_by_name(request, "temperature").as_numpy()[0])
                temperature = round(temperature, 2)
                print(f'[DEBUG] input `temperature` type({type(temperature)}): {temperature}')

                random_seed = 0
                if pb_utils.get_input_tensor_by_name(request, "random_seed") is not None:
                    random_seed = int(pb_utils.get_input_tensor_by_name(request, "random_seed").as_numpy()[0])
                print(f'[DEBUG] input `random_seed` type({type(random_seed)}): {random_seed}')

                if random_seed > 0:
                   random.seed(random_seed)
                   np.random.seed(random_seed)
                   torch.manual_seed(random_seed)
                   if torch.cuda.is_available():
                       torch.cuda.manual_seed_all(random_seed)

                stop_words = ""
                if pb_utils.get_input_tensor_by_name(request, "stop_words") is not None:
                    stop_words = pb_utils.get_input_tensor_by_name(request, "stop_words").as_numpy()
                print(f'[DEBUG] input `stop_words` type({type(stop_words)}): {stop_words}')
                if len(stop_words) == 0:
                    stop_words = None
                elif stop_words.shape[0] > 1:
                    # TODO: Check wether shoule we decode this words
                    stop_words = list(stop_words)
                else:
                    stop_words = [str(stop_words[0])]

                # if stop_words is not None:
                #     extra_params['stop'] = stop_words
                print(f'[DEBUG] parsed input `stop_words` type({type(stop_words)}): {stop_words}')

                if "top_p" not in extra_params:
                    extra_params['top_p'] = 0.95
                # Inference
                # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
                # https://huggingface.co/docs/transformers/v4.30.1/en/main_classes/text_generation#transformers.GenerationConfig
                t0 = time.time()
                # sequences = self.pipeline(
                #     prompt,
                #     do_sample=True,
                #     top_k=top_k,
                #     temperature=temperature,
                #     num_return_sequences=1,
                #     eos_token_id=self.tokenizer.eos_token_id,
                #     max_new_tokens=max_new_tokens,
                #     **extra_params
                    
                # )
                print(f'[DEBUG] Conversation Prompt: \n{conv.get_prompt()}')

                # if you want to manually add prompt
                # prompt = self.pipe.tokenizer.apply_chat_template(
                #     messages, 
                #     tokenize=False, 
                #     add_generation_prompt=True
                # )
                
                sequences = self.pipe(
                        conv.get_prompt(), 
                        max_new_tokens=max_new_tokens, 
                        do_sample=True, 
                        temperature=temperature, 
                        top_k=top_k, 
                        # top_p=0.95
                        **extra_params
                )


                self.logger.log_info(f'Inference time cost {time.time()-t0}s with input lenth {len(prompt)}')

                text_outputs = [
                    seq['generated_text'][len(conv.get_prompt()):].encode('utf-8')
                    for seq in sequences
                ]
                print('-'*100)
                print(text_outputs)
                print('-'*100)
                
                triton_output_tensor = pb_utils.Tensor(
                    "text", np.asarray(text_outputs, dtype=self.output0_dtype)
                )
                responses.append(pb_utils.InferenceResponse(output_tensors=[triton_output_tensor]))
            except Exception as e:
                print("DEBUG\n", traceback.format_exc())
                self.logger.log_info(f"Error generating stream: {e}")
                error = pb_utils.TritonError(f"Error generating stream: {e}")
                triton_output_tensor = pb_utils.Tensor(
                    "text", np.asarray(["N/A"], dtype=self.output0_dtype)
                )
                response = pb_utils.InferenceResponse(
                    output_tensors=[triton_output_tensor], error=error
                )
                responses.append(response)
                self.logger.log_info("The model did not receive the expected inputs")
                raise e
            return responses

    def finalize(self):
        self.logger.log_info("Cleaning up ...")
