
from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch
from minisgl.message import TokenizeMsg

if TYPE_CHECKING:
    from transformers import LlamaTokenizer


class TokenizeManager:
    """
    Tokenizer 管理器
    
    负责将文本转换为 token IDs。
    """
    def __init__(self, tokenizer: LlamaTokenizer) -> None:
        self.tokenizer = tokenizer

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[torch.Tensor]:
        """
        批量 Tokenize
        
        处理用户输入的文本，将其编码为 input_ids。
        支持纯文本和聊天模版 (list of messages)。
        """
        results: List[torch.Tensor] = []
        # TODO: batch tokenization (目前是并在循环中处理，有优化空间)
        for msg in msgs:
            if isinstance(msg.text, list):
                # 使用聊天模版 (Chat Template)
                prompt = self.tokenizer.apply_chat_template(
                    msg.text,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                assert isinstance(prompt, str)
            else:
                # 纯文本 prompt
                prompt = msg.text
            
            # 使用 HuggingFace tokenizer 进行编码
            input_ids: torch.Tensor = (  # type: ignore
                self.tokenizer.encode(prompt, return_tensors="pt")
            )
            results.append(input_ids.view(-1).to(torch.int32))
        return results
