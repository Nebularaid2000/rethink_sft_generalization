# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
SFT dataset
- We assume user pass a single parquet file.
- We load all the data into the memory.
Each parquet file contains
"""

import pandas as pd
import torch
from omegaconf.listconfig import ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
import json
from datasets import load_dataset, concatenate_datasets
import numpy as np


class SFTDataset(Dataset):
    """
    This is an in-memory SFTDataset

    Arguments:
        config (OmegaConf): the data config
    """

    def __init__(self, parquet_files: str | ListConfig, tokenizer, config):
        prompt_key = config.get("prompt_key", "prompt")
        prompt_dict_keys = config.get("prompt_dict_keys", None)
        response_key = config.get("response_key", "response")
        response_dict_keys = config.get("response_dict_keys", None)
        max_length = config.get("max_length", 1024)
        truncation = config.get("truncation", "error")
        use_shm = config.get("use_shm", False)

        assert truncation in ["error", "left", "right"]
        self.truncation = truncation
        self.use_shm = use_shm

        if not isinstance(parquet_files, ListConfig):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.prompt_key = prompt_key if isinstance(prompt_key, tuple | list) else [prompt_key]
        self.response_key = response_key if isinstance(response_key, tuple | list) else [response_key]
        self.prompt_dict_keys = prompt_dict_keys if prompt_dict_keys else []
        self.response_dict_keys = response_dict_keys if response_dict_keys else []

        self.max_length = max_length

        self._download()
        self._read_files_and_tokenize()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, pandas.core.series.Series | numpy.ndarray) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)
        self.prompts = self.dataframe[self.prompt_key]
        for key in self.prompt_dict_keys:
            # type(x): pandas.core.series.Series
            # type(x[0]): numpy.ndarray
            # type(x[0][0]): dict
            try:
                self.prompts = self.prompts.apply(lambda x: series_to_item(x)[key], axis=1)  # noqa: B023
            except Exception:
                print(f"self.prompts={self.prompts}")
                raise
        if isinstance(self.prompts, pd.DataFrame):
            self.prompts = self.prompts.squeeze()
        self.prompts = self.prompts.tolist()
        self.responses = self.dataframe[self.response_key]
        for key in self.response_dict_keys:
            try:
                self.responses = self.responses.apply(lambda x: series_to_item(x)[key], axis=1)  # noqa: B023
            except Exception:
                print(f"self.responses={self.responses}")
                raise
        if isinstance(self.responses, pd.DataFrame):
            self.responses = self.responses.squeeze()
        self.responses = self.responses.tolist()

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item):
        tokenizer = self.tokenizer

        prompt = self.prompts[item]
        response = self.responses[item]

        # apply chat template
        if isinstance(prompt, str):
            prompt_chat = [{"role": "user", "content": prompt}]
        else:
            prompt_chat = prompt

        # Original apply_chat_template approach: concatenate into a string first, then tokenize.
        # prompt_chat_str = tokenizer.apply_chat_template(prompt_chat, add_generation_prompt=True, tokenize=False)
        # prompt_ids_output = tokenizer(prompt_chat_str, return_tensors="pt", add_special_tokens=False)
        # prompt_ids = prompt_ids_output["input_ids"][0]
        # prompt_attention_mask = prompt_ids_output["attention_mask"][0]

        # Updated tokenization approach: tokenize directly.
        prompt_ids_output = tokenizer.apply_chat_template(
            prompt_chat, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        )
        if isinstance(prompt_ids_output, BatchEncoding) and "input_ids" in prompt_ids_output: # Different tokenizers may behave differently; some return a dict, others only input IDs.
            prompt_ids = prompt_ids_output["input_ids"][0]
            prompt_attention_mask = prompt_ids_output["attention_mask"][0]
        else:
            prompt_ids = prompt_ids_output[0]
            prompt_attention_mask = torch.ones_like(prompt_ids)

        # Original eos_token append approach: append eos token string first, then tokenize.
        # response_chat_str = response + tokenizer.eos_token
        # response_ids_output = tokenizer(response_chat_str, return_tensors="pt", add_special_tokens=False)
        # response_ids = response_ids_output["input_ids"][0]
        # response_attention_mask = response_ids_output["attention_mask"][0]

        # Updated eos_token append approach.
        response_ids_output = tokenizer(response, return_tensors="pt", add_special_tokens=False)
        response_ids = response_ids_output["input_ids"][0]
        response_attention_mask = response_ids_output["attention_mask"][0]
        # append eos_token_id directly
        eos_token_id = torch.tensor([tokenizer.eos_token_id], dtype=response_ids.dtype)
        eos_attention_mask = torch.ones(1, dtype=response_attention_mask.dtype)
        response_ids = torch.cat([response_ids, eos_token_id])
        response_attention_mask = torch.cat([response_attention_mask, eos_attention_mask])
        
        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)

        # padding to max length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            padded_input_ids = (
                torch.ones(size=(self.max_length - sequence_length,), dtype=input_ids.dtype)
                * self.tokenizer.pad_token_id
            )
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                # actually, left truncation may not be reasonable
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
            elif self.truncation == "error":
                raise NotImplementedError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise NotImplementedError(f"Unknown truncation method {self.truncation}")

        position_ids = compute_position_id_with_mask(attention_mask)

        loss_mask = attention_mask.clone().detach()
        if prompt_length > 1:
            # mask out prompt for SFT.
            loss_mask[: min(prompt_length, loss_mask.size(0)) - 1] = 0
        # mask out the last token in response
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }



class OurSFTDataset(Dataset):
    """
    This is an in-memory SFTDataset

    Arguments:
        config (OmegaConf): the data config
    """

    def __init__(self, parquet_files: str | ListConfig, tokenizer, config):
        prompt_key = config.get("prompt_key", "prompt")
        prompt_dict_keys = config.get("prompt_dict_keys", None)
        response_key = config.get("response_key", "response")
        response_dict_keys = config.get("response_dict_keys", None)
        max_length = config.get("max_length", 1024)
        truncation = config.get("truncation", "error")
        use_shm = config.get("use_shm", False)
        self.use_simple_template = config.get("use_simple_template", False)

        assert truncation in ["error", "left", "right"]
        self.truncation = truncation
        self.use_shm = use_shm

        if not isinstance(parquet_files, ListConfig):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.prompt_key = prompt_key if isinstance(prompt_key, tuple | list) else [prompt_key]
        self.response_key = response_key if isinstance(response_key, tuple | list) else [response_key]
        self.prompt_dict_keys = prompt_dict_keys if prompt_dict_keys else []
        self.response_dict_keys = response_dict_keys if response_dict_keys else []

        self.logprob_key = config.get("logprob_key", "logprob")
        self.advantage_key = config.get("advantage_key", "advantage")
        self.return_logprob = config.ref_log_prob.enable
        self.only_return_mean_logprob = config.ref_log_prob.only_return_mean_logprob
        # TODO: may need entropy?

        self.max_length = max_length

        self._download()
        self._read_files_and_tokenize()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, pandas.core.series.Series | numpy.ndarray) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)
        self.prompts = self.dataframe[self.prompt_key]
        for key in self.prompt_dict_keys:
            # type(x): pandas.core.series.Series
            # type(x[0]): numpy.ndarray
            # type(x[0][0]): dict
            try:
                self.prompts = self.prompts.apply(lambda x: series_to_item(x)[key], axis=1)  # noqa: B023
            except Exception:
                print(f"self.prompts={self.prompts}")
                raise
        if isinstance(self.prompts, pd.DataFrame):
            self.prompts = self.prompts.squeeze()
        self.prompts = self.prompts.tolist()

        self.responses = self.dataframe[self.response_key]
        for key in self.response_dict_keys:
            try:
                self.responses = self.responses.apply(lambda x: series_to_item(x)[key], axis=1)  # noqa: B023
            except Exception:
                print(f"self.responses={self.responses}")
                raise
        if isinstance(self.responses, pd.DataFrame):
            self.responses = self.responses.squeeze()
        self.responses = self.responses.tolist()

        if self.return_logprob:
            self.log_probs = self.dataframe[self.logprob_key].tolist()
        else:
            self.log_probs = None

        self.advantages = self.dataframe[self.advantage_key].tolist() # if self.advantage_key in self.dataframe else None


    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item):
        tokenizer = self.tokenizer

        prompt = self.prompts[item]
        response = self.responses[item]
        assert type(response) is str

        if not self.use_simple_template:
            # apply chat template
            if isinstance(prompt, str):
                prompt_chat = [{"role": "user", "content": prompt}]
            else:
                prompt_chat = prompt

            # Original apply_chat_template approach: concatenate into a string first, then tokenize.
            # prompt_chat_str = tokenizer.apply_chat_template(prompt_chat, add_generation_prompt=True, tokenize=False)
            # prompt_ids_output = tokenizer(prompt_chat_str, return_tensors="pt", add_special_tokens=False)
            # prompt_ids = prompt_ids_output["input_ids"][0]
            # prompt_attention_mask = prompt_ids_output["attention_mask"][0]

            # Updated tokenization approach: tokenize directly.
            prompt_ids_output = tokenizer.apply_chat_template(
                prompt_chat, add_generation_prompt=True, tokenize=True, return_tensors="pt"
            )
        else:
            if isinstance(prompt, str):
                prompt_chat = prompt
            else: # message list, we extract the prompt from the first item of the list
                assert isinstance(prompt, list) or isinstance(prompt, np.ndarray)
                assert len(prompt) > 0 
                assert isinstance(prompt[0], dict)
                assert "content" in prompt[0]
                prompt_chat = prompt[0]["content"]
            prompt_chat = f"Question: {prompt_chat}\nAnswer:"
            # Tokenize the prompt directly.
            prompt_ids_output = tokenizer(prompt_chat, return_tensors="pt")

        if isinstance(prompt_ids_output, BatchEncoding) and "input_ids" in prompt_ids_output: # Different tokenizers may behave differently; some return a dict, others only input IDs.
            prompt_ids = prompt_ids_output["input_ids"][0]
            prompt_attention_mask = prompt_ids_output["attention_mask"][0]
        else:
            prompt_ids = prompt_ids_output[0]
            prompt_attention_mask = torch.ones_like(prompt_ids)

        # Original eos_token append approach: append eos token string first, then tokenize.
        # response_chat_str = response + tokenizer.eos_token
        # response_ids_output = tokenizer(response_chat_str, return_tensors="pt", add_special_tokens=False)
        # response_ids = response_ids_output["input_ids"][0]
        # response_attention_mask = response_ids_output["attention_mask"][0]

        # Updated eos_token append approach.
        response_ids_output = tokenizer(response, return_tensors="pt", add_special_tokens=False)
        response_ids = response_ids_output["input_ids"][0]
        response_attention_mask = response_ids_output["attention_mask"][0]
        # append eos_token_id directly
        eos_token_id = torch.tensor([tokenizer.eos_token_id], dtype=response_ids.dtype)
        eos_attention_mask = torch.ones(1, dtype=response_attention_mask.dtype)
        response_ids = torch.cat([response_ids, eos_token_id])
        response_attention_mask = torch.cat([response_attention_mask, eos_attention_mask])
        
        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)        

        # padding to max length
        # Default is right padding, same as in openrlhf.
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            padded_input_ids = (
                torch.ones(size=(self.max_length - sequence_length,), dtype=input_ids.dtype)
                * self.tokenizer.pad_token_id
            )
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                # actually, left truncation may not be reasonable
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
            elif self.truncation == "error":
                raise NotImplementedError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise NotImplementedError(f"Unknown truncation method {self.truncation}")

        position_ids = compute_position_id_with_mask(attention_mask)

        loss_mask = attention_mask.clone().detach()
        if prompt_length > 1:
            # Mask out prompt tokens for SFT. Note the `-1` here, which effectively shifts by one position.
            loss_mask[: min(prompt_length, loss_mask.size(0)) - 1] = 0
        # Mask out the last token in the response (likely eos token). This is fine because logits are all shifted by one position.
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0

        # log prob from teacher model
        if self.return_logprob:
            assert self.log_probs is not None
            log_probs = self.log_probs[item]
            if isinstance(log_probs, np.ndarray):
                # print("concerting log_probs from np.ndarray to list")
                log_probs = log_probs.tolist()
            log_probs = torch.tensor(log_probs, dtype=torch.float32)
            mean_log_probs = torch.mean(log_probs, dim=-1, keepdim=True)
            assert len(mean_log_probs.shape) == 1
            if not self.only_return_mean_logprob:
                if response_length != log_probs.shape[0]:
                    raise ValueError(f"response_length={response_length}, log_probs.shape={log_probs.shape}")
                log_probs_padded = torch.zeros_like(attention_mask, dtype=torch.float32)
                log_probs_padded[prompt_length - 1 : prompt_length - 1 + response_length] = log_probs

        # advantage (e.g., +1 or -1)
        advantages = torch.tensor([self.advantages[item]], dtype=torch.float32) # tensor of (1,)
        assert advantages.shape == (1,)

        data_item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
            "advantages": advantages,
        }

        if self.return_logprob:
            if self.only_return_mean_logprob:
                data_item["mean_ref_log_prob"] = mean_log_probs
            else:
                data_item["ref_log_prob"] = log_probs_padded

        return data_item
