# # ===== PATCH for Step3VL Processor compatibility =====
# try:
#     from transformers import ProcessorMixin
#     def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
#         if not image_sizes:
#             # 不能返回空列表，vLLM 会调用 [0] 取第一个元素
#             return {"num_image_tokens": [1024]}  # ✅ 默认值用于 profiling
#         return {"num_image_tokens": [1024] * len(image_sizes)}  # ✅ 正确 key
#     if not hasattr(ProcessorMixin, '_get_num_multimodal_tokens'):
#         ProcessorMixin._get_num_multimodal_tokens = _get_num_multimodal_tokens
# except Exception:
#     pass
# # ===== END PATCH =====

import copy
import logging
from importlib.metadata import version
from importlib.util import find_spec
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union

from more_itertools import distribute
from packaging.version import parse as parse_version
from tqdm import tqdm

from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import (
    Collator,
    configure_pad_token,
    handle_stop_sequences,
    undistribute,
)
from lm_eval.utils import (
    get_rolling_token_windows,
    make_disjoint_window,
)

try:
    import ray
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ModuleNotFoundError:
    pass

if TYPE_CHECKING:
    pass

eval_logger = logging.getLogger(__name__)


@register_model("vllm")
class VLLM(TemplateLM):
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        pretrained: str,
        dtype: Literal["float16", "bfloat16", "float32", "auto"] = "auto",
        revision: Optional[str] = None,
        trust_remote_code: Optional[bool] = True,
        tokenizer: Optional[str] = None,
        tokenizer_mode: Literal["auto", "slow"] = "auto",
        tokenizer_revision: Optional[str] = None,
        add_bos_token: Optional[bool] = False,
        prefix_token_id: Optional[int] = None,
        tensor_parallel_size: int = 1,
        quantization: Optional[str] = None,
        max_gen_toks: int = 256,
        swap_space: int = 4,
        batch_size: Union[str, int] = 1,
        max_batch_size=None,
        max_length: int = None,
        max_model_len: int = None,
        seed: int = 1234,
        gpu_memory_utilization: float = 0.9,
        device: str = "cuda",
        data_parallel_size: int = 1,
        lora_local_path: str = None,
        **kwargs,
    ):
        super().__init__()

        if not find_spec("vllm"):
            raise ModuleNotFoundError(
                "attempted to use 'vllm' LM type, but package `vllm` is not installed. "
                "Please install vllm via `pip install lm-eval[vllm]` or `pip install -e .[vllm]`"
            )

        assert max_length is None or max_model_len is None, (
            "Either max_length or max_model_len may be provided, but not both"
        )

        self._max_length = max_model_len if max_model_len is not None else max_length
        self.tensor_parallel_size = int(tensor_parallel_size)
        self.data_parallel_size = int(data_parallel_size)
        self.model_args = {
            "model": pretrained,
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "revision": revision,
            "dtype": dtype,
            "tokenizer": tokenizer,
            "tokenizer_mode": tokenizer_mode,
            "tokenizer_revision": tokenizer_revision,
            "trust_remote_code": trust_remote_code,
            "tensor_parallel_size": int(tensor_parallel_size),
            "max_model_len": int(self._max_length) if self._max_length else None,
            "swap_space": int(swap_space),
            "quantization": quantization,
            "seed": int(seed),
        }
        self.model_args.update(kwargs)
        self.batch_size = (
            "auto"
            if isinstance(batch_size, str) and "auto" in batch_size
            else int(batch_size)
        )
        if self.data_parallel_size <= 1:
            # self.model = LLM(**self.model_args)

            # ---- 构造 vLLM 的参数（去掉不兼容的） ----
            vllm_args = dict(self.model_args)

            # vLLM 对 MistralCommonBackend 不支持这两个参数 → 必须删除
            vllm_args.pop("tokenizer_mode", None)
            vllm_args.pop("trust_remote_code", None)

            # ---- 根据 partial_pretrain 判断是否为 Ministral-3 ----
            if "Ministral-3" in pretrained or "mistral3" in pretrained.lower():
                print("Using Ministral-3 model with vLLM!")
                self.model = LLM(**vllm_args)
            else:
                print("Using standard model with vLLM!")
                self.model = LLM(**self.model_args)

        else:
            eval_logger.warning(
                "You might experience occasional issues with model weight downloading when data_parallel is in use. To ensure stable performance, run with data_parallel_size=1 until the weights are downloaded and cached."
            )
            self.model_args["distributed_executor_backend"] = "ray"
            self.batch_size = "auto"
            eval_logger.info("Manual batching is not compatible with data parallelism.")

        from transformers import AutoConfig

        # import transformers
        # import os
        # print("Transformers version:", transformers.__version__)
        # print("Transformers package path:", os.path.dirname(transformers.__file__))
        # from transformers import MistralCommonBackend

        if "Ministral-3" in pretrained or "mistral3" in pretrained.lower(): # TODO: 注意这里可能有多种情况
            print("***** Using MistralCommonBackend tokenizer for Ministral-3 model!!!")
            from transformers import MistralCommonBackend
            self.tokenizer = MistralCommonBackend.from_pretrained(
                    tokenizer if tokenizer else pretrained,
                    # tokenizer_mode=tokenizer_mode,
                    # trust_remote_code=trust_remote_code,
                    revision=tokenizer_revision,
                    # add_bos_token=add_bos_token,
                )
        else:
            self.tokenizer = get_tokenizer(
                tokenizer if tokenizer else pretrained,
                tokenizer_mode=tokenizer_mode,
                trust_remote_code=trust_remote_code,
                revision=tokenizer_revision,
                # add_bos_token=add_bos_token, #FFALSEEEEEEEEEEEEEEEEEEEEEEE
            ) 

        if "Ministral-3" in pretrained or "mistral3" in pretrained.lower():
            from transformers import Mistral3Config, Mistral3ForConditionalGeneration
            self._config = Mistral3Config.from_pretrained(
                pretrained, trust_remote_code=trust_remote_code, revision=revision
            )
        else:
            self._config = AutoConfig.from_pretrained(
                pretrained, trust_remote_code=trust_remote_code, revision=revision
            )
        # if "Ministral-3" in self.config.model.partial_pretrain:
        #         print("using Ministral-3 model!!")
        #         self.model: PreTrainedModel = Mistral3ForConditionalGeneration.from_pretrained(
        #             local_model_path,
        #             config=config,
        #             torch_dtype=torch_dtype,
        #             attn_implementation="flash_attention_2",
        #             trust_remote_code=trust_remote_code,
        #         )            
        #     else:
        #         self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        #             local_model_path,
        #             config=config,
        #             torch_dtype=torch_dtype,
        #             attn_implementation="flash_attention_2",
        #             trust_remote_code=trust_remote_code,
        #         )


        
        self.tokenizer = configure_pad_token(self.tokenizer, model_config=self._config)
        self.add_bos_token = add_bos_token
        if "gemma" in pretrained.lower():
            self.add_bos_token = True
            eval_logger.info(
                "Found 'gemma' in model name, a BOS token will be used as Gemma series models underperform without it."
            )

        self.custom_prefix_token_id = prefix_token_id
        if prefix_token_id is not None:
            eval_logger.info(
                f"Loglikelihood prefix token id used in evaluation: {self.prefix_token_id}"
            )

        self._max_gen_toks = max_gen_toks

        if lora_local_path is not None:
            assert parse_version(version("vllm")) > parse_version("0.3.0"), (
                "lora adapters only compatible with vllm > v0.3.0."
            )
            self.lora_request = LoRARequest("finetuned", 1, lora_local_path)
        else:
            self.lora_request = None

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self):
        # it is used as prefix for loglikelihood
        if self.custom_prefix_token_id is not None:
            return self.custom_prefix_token_id
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        if self.data_parallel_size <= 1:
            return self.model.llm_engine.model_config.max_model_len
        else:
            seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
            for attr in seqlen_config_attrs:
                if hasattr(self._config, attr):
                    return getattr(self._config, attr)
            if hasattr(self.tokenizer, "model_max_length"):
                if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                    return self._DEFAULT_MAX_LENGTH
                return self.tokenizer.model_max_length
            return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    def apply_chat_template(
        self, chat_history: List[Dict[str, str]], add_generation_prompt: bool = True, **kwargs
    ) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        chat_templated = self.tokenizer.apply_chat_template(
            chat_history,
            #
            tokenize=True,
            #
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
            **kwargs,
        )

        return chat_templated

        # ## new template
        # simple_text = ""                                                            
        # for msg in chat_history:                                                    
        #     if msg["role"] == "system":                                             
        #         continue  # 跳过 system 消息                                        
        #     elif msg["role"] == "user":                                             
        #         simple_text += msg["content"]                                       
        #     elif msg["role"] == "assistant":                                        
        #         simple_text += msg["content"]  
        # print(f"fuck you simple_text:{simple_text}")                                     
        # return simple_text    
        # # ## return chat_history
        # ##

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    # def tok_encode(
    #     self,
    #     string: Union[str, List[str]],
    #     left_truncate_len: int = None,
    #     add_special_tokens: bool = False,
    #     truncation: bool = False,
    # ) -> Union[List[int], List[List[int]]]:
    #     if not add_special_tokens:
    #         add_special_tokens = False or self.add_bos_token
    #     encoding: Union[List[List[int]], List[int]] = self.tokenizer(
    #         string,
    #         add_special_tokens=add_special_tokens,
    #         truncation=truncation,
    #         return_attention_mask=False,
    #     ).input_ids

    #     # left-truncate the encoded context to be at most `left_truncate_len` tokens long
    #     if left_truncate_len:
    #         if not isinstance(string, str):
    #             encoding = [enc[-left_truncate_len:] for enc in encoding]
    #         else:
    #             encoding = encoding[-left_truncate_len:]

    #     return encoding

    ## OLD

    def tok_encode(
        self,
        string: Union[str, List[str]],
        left_truncate_len: int = None,
        add_special_tokens: bool = False,
        truncation: bool = False,
    ) -> Union[List[int], List[List[int]]]:
        if not add_special_tokens:
            add_special_tokens = False or self.add_bos_token
        # if isinstance(string, list):
        #     # 检查是否是 list[list[int]] (batch of token ids)
        #     if len(string) > 0 and isinstance(string[0], list):
        #         if all(isinstance(x, int) for x in string[0]):
        #             return string  # 已经是 token IDs，直接返回
        #     # 检查是否是 list[int] (single sequence of token ids)
        #     elif all(isinstance(x, int) for x in string):
        #         return string  # 已经是 token IDs，直接返回

        # 尝试传 return_attention_mask，如果不支持就退回不传
        try:
            encoding: Union[List[List[int]], List[int]] = self.tokenizer(
                string,
                add_special_tokens=add_special_tokens,
                truncation=truncation,
                return_attention_mask=False,
            ).input_ids
        except TypeError:
            # 比如 MistralTokenizer.__call__ 没有 return_attention_mask 参数
            encoding: Union[List[List[int]], List[int]] = self.tokenizer(
                string,
                add_special_tokens=add_special_tokens,
                truncation=truncation,
            ).input_ids

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            if not isinstance(string, str):
                encoding = [enc[-left_truncate_len:] for enc in encoding]
            else:
                encoding = encoding[-left_truncate_len:]

        return encoding


    def _model_generate(
        self,
        requests: List[List[int]] = None,
        generate: bool = False,
        max_tokens: int = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ):
        if generate:
            kwargs = self.modify_gen_kwargs(kwargs)
            sampling_params = SamplingParams(max_tokens=max_tokens, stop=stop, **kwargs)
        else:
            sampling_params = SamplingParams(
                temperature=0, prompt_logprobs=1, max_tokens=1, detokenize=False
            )
        if self.data_parallel_size > 1:
            # vLLM hangs if resources are set in ray.remote
            # also seems to only work with decorator and not with ray.remote() fn
            # see https://github.com/vllm-project/vllm/issues/973
            @ray.remote
            def run_inference_one_model(
                model_args: dict,
                sampling_params: SamplingParams,
                requests: List[List[int]],
                lora_request: LoRARequest,
            ):
                llm = LLM(**model_args)
                return llm.generate(
                    prompt_token_ids=requests,
                    sampling_params=sampling_params,
                    lora_request=lora_request,
                )

            # dispatch requests to all self.data_parallel_size workers, in interleaved fashion
            # interleaved important to balance context lengths across workers
            requests = [list(x) for x in distribute(self.data_parallel_size, requests)]
            inputs = (
                (self.model_args, sampling_params, req, self.lora_request)
                for req in requests
            )
            object_refs = [run_inference_one_model.remote(*x) for x in inputs]
            results = ray.get(object_refs)
            # Invoke ray.shutdown() to prevent hang-ups if subsequent calls required.
            ray.shutdown()
            # flatten results
            return undistribute(results)

        # print(f"params from vllm_causallms: {sampling_params}\n")




        # TRUEEEEEEEEEEEE
        # outputs = self.model.generate(
        #     prompt_token_ids=requests,
        #     # prompts = prompts,
        #     sampling_params=sampling_params,
        #     use_tqdm=True if self.batch_size == "auto" else False,
        #     lora_request=self.lora_request,
        # )









        # print(f"outputs from vllm_causallms: {outputs}\n")

        # prefix_token_id = self.tokenizer.encode("Answer:<think>")
        # requests.append(prefix_token_id)
        # outputs = self.model.generate(
        #     prompt_token_ids=requests,
        #     sampling_params=sampling_params,
        #     use_tqdm=True if self.batch_size == "auto" else False,
        #     lora_request=self.lora_request,
        # )

        # from vllm.inputs import TokensPrompt
        # print(len(requests))
        # prompts = [TokensPrompt(prompt_token_ids=req) for req in requests]
        # outputs = self.model.generate(
        #     # prompt_token_ids=requests,
        #     prompts = prompts,
        #     sampling_params=sampling_params,
        #     use_tqdm=True if self.batch_size == "auto" else False,
        #     lora_request=self.lora_request,
        # )

        from vllm.inputs import TokensPrompt
        # print(len(requests))
        ## for no applied chat template
        # tokenized_requests = [self.tokenizer.encode(req) for req in requests]
        # prompts = [TokensPrompt(prompt_token_ids=req) for req in tokenized_requests]

        ##
        prompts = [TokensPrompt(prompt_token_ids=req) for req in requests]
        # prefix_token_id = self.tokenizer.encode("<think>")
        ##
        
        outputs = self.model.generate(
            # prompt_token_ids=requests,
            prompts = prompts,
            sampling_params=sampling_params,
            use_tqdm=True if self.batch_size == "auto" else False,
            lora_request=self.lora_request,
            # prefix_token_ids=prefix_token_id,
        )
        return outputs

    def loglikelihood_rolling(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[float]:
        adaptive_batch_size = None
        if self.batch_size == "auto":
            adaptive_batch_size = len(requests)

        # First, collect all windows from all requests
        all_windows = []  # List of (request_idx, window) tuples
        request_window_counts = []  # Track number of windows per request

        for req_idx, (string,) in enumerate(
            tqdm(
                [req.args for req in requests],
                disable=(disable_tqdm or (self.rank != 0)),
            )
        ):
            rolling_token_windows: List[Tuple[List[int], List[int]]] = list(
                map(
                    make_disjoint_window,
                    get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.prefix_token_id,
                        # max_seq_len - (1 for context)
                        max_seq_len=self.max_length - 1,
                        context_len=1,
                    ),
                )
            )

            # TODO: Right now, we pass single EOT token to the Encoder and the full context to the decoder, in seq2seq case
            windows = [(None,) + x for x in rolling_token_windows]

            # Store windows with their request index
            all_windows.extend((req_idx, window) for window in windows)
            request_window_counts.append(len(windows))

        all_nlls = []
        batch_size = adaptive_batch_size or int(self.batch_size)
        for i in range(0, len(all_windows), batch_size):
            batch = all_windows[i : i + batch_size]
            # Extract just the windows for processing, keeping track of request indices
            batch_indices, batch_windows = zip(*batch)

            batch_nlls = self._loglikelihood_tokens(
                requests=batch_windows,
                disable_tqdm=False,
            )
            # Store results with their request indices
            all_nlls.extend(zip(batch_indices, batch_nlls))

        # Reconstruct per-request loglikelihoods
        loglikelihoods = []
        current_idx = 0
        for window_count in request_window_counts:
            # Get all nlls for this request
            request_nlls = all_nlls[current_idx : current_idx + window_count]
            # Sum up the nlls for this request (discarding is_greedy)
            request_total = sum(nll[0] for _, nll in request_nlls)
            loglikelihoods.append(request_total)
            current_idx += window_count

            string = requests[len(loglikelihoods) - 1].args[0]
            self.cache_hook.add_partial(
                "loglikelihood_rolling", (string,), request_total
            )

        return loglikelihoods

    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        from transformers.tokenization_utils_base import BatchEncoding

        res = []

        # batch tokenize contexts
        context, all_gen_kwargs = zip(*(req.args for req in requests))
        # context_encoding: List[List[int]] = self.tok_encode(
        #     context, add_special_tokens=self.add_bos_token
        # )
        # print(f"[DEBUG] Context type: {type(context)}")
        # print(f"[DEBUG] Context length: {len(context)}")
        
        # if len(context) > 0:
        #     print(f"[DEBUG] First element type: {type(context[0])}")
        #     print(f"[DEBUG] First element: {context[0]}")
            
        #     if isinstance(context[0], list):
        #         print(f"[DEBUG] First element length: {len(context[0])}")
        #         print(f"[DEBUG] First element sample: {context[0][:10]}")
        #         if len(context[0]) > 0:
        #             print(f"[DEBUG] First element's first item type: {type(context[0][0])}")
        #     elif isinstance(context[0], str):
        #         print(f"[DEBUG] First element (str) preview: {context[0][:100]}")
        
        # # 如果是嵌套列表，查看形状
        # if isinstance(context, list):
        #     try:
        #         import numpy as np
        #         arr = np.array(context, dtype=object)
        #         print(f"[DEBUG] Context shape (as numpy array): {arr.shape}")
        #     except:
        #         pass
        # if len(context) > 0 and isinstance(context[0], BatchEncoding):
        #     # 提取 input_ids 作为 context_encoding
        #     context_encoding = [item['input_ids'] for item in context]
        #     # context 也使用相同的 input_ids（避免排序问题）
        #     context = context_encoding
        # else:
        #     # 原有的 tokenization 逻辑
        #     context_encoding = self.tok_encode(
        #         context, add_special_tokens=self.add_bos_token
        #     )

        context_encoding = []
        for item in context:
            try:
                context_encoding.append(item['input_ids'])
            except (KeyError, TypeError):
                context_encoding.append(item)


        context = context_encoding

        requests = [
            ((a, b), c) for a, b, c in zip(context, context_encoding, all_gen_kwargs)
        ]

        def _collate_gen(_requests):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            return -len(_requests[0][1]), _requests[0][0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = Collator(requests, _collate_gen, group_by="gen_kwargs")
        chunks = re_ords.get_batched(
            n=int(self.batch_size) if self.batch_size != "auto" else 0, batch_fn=None
        )

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )
        # for each different set of kwargs, we execute all requests, by batch.
        eos = self.tokenizer.decode(self.eot_token_id)
        for chunk in chunks:
            context_and_encoding, all_gen_kwargs = zip(*chunk)
            context, context_encoding = zip(*context_and_encoding)
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            # unpack our keyword arguments.
            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                # add EOS token to stop sequences
                until = handle_stop_sequences(kwargs.pop("until", None), eos=eos)
            else:
                raise ValueError(
                    f"Expected `kwargs` to be of type `dict` but got {type(gen_kwargs)}"
                )
            if "max_gen_toks" in kwargs.keys():
                max_gen_toks = kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self.max_gen_toks

            # set the max length in tokens of inputs ("context_enc")
            # max len for inputs = max length, minus room to generate the max new tokens
            max_ctx_len = self.max_length - max_gen_toks
            context_encoding = [x[-max_ctx_len:] for x in context_encoding]

            # perform batched generation
            cont = self._model_generate(
                requests=context_encoding,
                generate=True,
                max_tokens=max_gen_toks,
                stop=until,
                **kwargs,
            )
            # print(f"outputs from generate until_1:{cont}\n")
            # cache generations
            # original code
            # for output, context in zip(cont, context):
            #     generated_text = output.outputs[0].text
            #     res.append(generated_text)
            #     # res.append(output.outputs[1].text)
            #     # res.append(output.outputs[2].text)
            #     self.cache_hook.add_partial(
            #         "generate_until", (context, gen_kwargs), generated_text
            #     )
            #     pbar.update(1)
            #     print("count+1")
            #
            if len(cont[0].outputs) > 1:
                res = []
                for output, context in zip(cont, context):
                    outputs_for_this_example = [o.text for o in output.outputs]  # n个回复
                    res.append(outputs_for_this_example)
                    # 仍然缓存第一个生成的（比如主要答案）
                    self.cache_hook.add_partial(
                        "generate_until", (context, gen_kwargs), outputs_for_this_example[0]
                    )
                    pbar.update(1)
            else:
                for output, context in zip(cont, context):
                    generated_text = output.outputs[0].text
                    res.append(generated_text)
                    # res.append(output.outputs[1].text)
                    # res.append(output.outputs[2].text)
                    self.cache_hook.add_partial(
                        "generate_until", (context, gen_kwargs), generated_text
                    )
                    pbar.update(1)
            # print(f"res from generate until_2:{res}\n")

        pbar.close()
        # reorder all group of results back to original unsorted form
        # print(res,"fuck you:",re_ords.get_original(res))
        # print(f"reordered res is {re_ords.get_original(res)}")
        # print(f"len is {len(re_ords.get_original(res))}")
        # print(f"res[0] is {re_ords.get_original(res)[0]}")
        return re_ords.get_original(res)

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
    ) -> List[Tuple[float, bool]]:
        res = []

        def _collate(x):
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        # Reorder requests by length and batch
        re_ord = Collator(requests, sort_fn=_collate)
        chunks = re_ord.get_batched(
            n=int(self.batch_size) if self.batch_size != "auto" else 0, batch_fn=None
        )

        pbar = tqdm(
            total=len(requests),
            disable=disable_tqdm,
            desc="Running loglikelihood requests",
        )
        for chunk in chunks:
            inputs = []
            ctxlens = []
            for cache_key, context_enc, continuation_enc in chunk:
                inp = (context_enc + continuation_enc)[-(self.max_length) :]
                ctxlen = len(context_enc) - max(
                    0, len(context_enc) + len(continuation_enc) - (self.max_length)
                )

                inputs.append(inp)
                ctxlens.append(ctxlen)

            outputs = self._model_generate(requests=inputs, generate=False)

            for output, ctxlen, (cache_key, _, _), inp in zip(
                outputs, ctxlens, chunk, inputs
            ):
                answer = self._parse_logprobs(
                    tokens=inp,
                    outputs=output,
                    ctxlen=ctxlen,
                )

                res.append(answer)

                if cache_key is not None:
                    # special case: loglikelihood_rolling produces a number of loglikelihood requests
                    # all with cache key None. instead do add_partial on the per-example level
                    # in the loglikelihood_rolling() function for those.
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)
                pbar.update(1)
        pbar.close()
        return re_ord.get_original(res)

    @staticmethod
    def _parse_logprobs(tokens: List, outputs, ctxlen: int) -> Tuple[float, bool]:
        """Process logprobs and tokens.

        :param tokens: list
            Input tokens (potentially left-truncated)
        :param outputs: RequestOutput
            Contains prompt_logprobs
        :param ctxlen: int
            Length of context (so we can slice them away and only keep the predictions)
        :return:
            continuation_logprobs: float
                Log probabilities of continuation tokens
            is_greedy: bool
                Whether argmax matches given continuation exactly
        """

        # The first entry of prompt_logprobs is None because the model has no previous tokens to condition on.
        continuation_logprobs_dicts = outputs.prompt_logprobs

        def coerce_logprob_to_num(logprob):
            # vLLM changed the return type of logprobs from float
            # to a Logprob object storing the float value + extra data
            # (https://github.com/vllm-project/vllm/pull/3065).
            # If we are dealing with vllm's Logprob object, return
            # the logprob value stored as an attribute. Otherwise,
            # return the object itself (which should be a float
            # for older versions of vLLM).
            return getattr(logprob, "logprob", logprob)

        continuation_logprobs_dicts = [
            {
                token: coerce_logprob_to_num(logprob)
                for token, logprob in logprob_dict.items()
            }
            if logprob_dict is not None
            else None
            for logprob_dict in continuation_logprobs_dicts
        ]

        # Calculate continuation_logprobs
        # assume ctxlen always >= 1
        continuation_logprobs = sum(
            logprob_dict.get(token)
            for token, logprob_dict in zip(
                tokens[ctxlen:], continuation_logprobs_dicts[ctxlen:]
            )
        )

        # Determine if is_greedy
        is_greedy = True
        for token, logprob_dict in zip(
            tokens[ctxlen:], continuation_logprobs_dicts[ctxlen:]
        ):
            # Get the token with the maximum log probability from the logprob_dict
            if logprob_dict:  # Ensure the logprob_dict is not None
                top_token = max(logprob_dict, key=logprob_dict.get)
                if top_token != token:
                    is_greedy = False
                    break

        return continuation_logprobs, is_greedy

    @staticmethod
    def modify_gen_kwargs(kwargs: dict) -> dict:
        # sampling_params
        kwargs["temperature"] = kwargs.get("temperature", 0.0)
        do_sample = kwargs.pop("do_sample", None)
        if do_sample is False and "temperature" not in kwargs:
            eval_logger.debug(
                "Got `do_sample=False` and no temperature value, setting VLLM temperature to 0.0 ..."
            )
            kwargs["temperature"] = 0.0
        # hf defaults
        kwargs["skip_special_tokens"] = kwargs.get("skip_special_tokens", False)
        kwargs["spaces_between_special_tokens"] = kwargs.get(
            "spaces_between_special_tokens", False
        )
        return kwargs
