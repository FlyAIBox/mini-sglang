
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from transformers import LlamaTokenizer

from minisgl.message import DetokenizeMsg

# Borrowed from sglang


def _is_chinese_char(cp: int):
    """
    检查字符码点是否为中日韩 (CJK) 统一表意文字。
    
    Checks whether CP is the codepoint of a CJK character.
    这一定义并未包含所有的日韩字符 (如韩语谚文、日语假名)，
    因为它们通常像字母一样组成单词，不需要特殊断句处理。
    """
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)  #
        or (cp >= 0x20000 and cp <= 0x2A6DF)  #
        or (cp >= 0x2A700 and cp <= 0x2B73F)  #
        or (cp >= 0x2B740 and cp <= 0x2B81F)  #
        or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
    ):  #
        return True

    return False


def find_printable_text(text: str):
    """
    寻找可以安全打印的文本子串，避免输出不完整的单词。
    
    Returns the longest printable substring of text that contains only entire words.
    """
    # Borrowed from https://github.com/huggingface/transformers/blob/061580c82c2db1de9139528243e105953793f7a2/src/transformers/generation/streamers.py#L99

    # 如果以换行符结尾，通常意味整句结束，可以安全打印
    if text.endswith("\n"):
        return text
    # 如果最后一个字符是 CJK 字符，通常不需要等待单词结束
    elif len(text) > 0 and _is_chinese_char(ord(text[-1])):
        return text
    # 如果倒数第二个字符是 CJK 字符，除了最后一个字符外都可以打印
    elif len(text) > 1 and _is_chinese_char(ord(text[-2])):
        return text[:-1]
    # 否则，打印到最后一个空格为止 (简单的启发式规则，避免打印不完整的英文单词)
    else:
        return text[: text.rfind(" ") + 1]


@dataclass
class DecodeStatus:
    """
    单个请求的解码状态
    
    用于在流式生成过程中跟踪解码进度，处理 UTF-8 截断和增量输出。
    """
    decoded_ids: List[int]  # 已生成的所有 token IDs
    decoded_str: str        # 已解码并确认输出的完整字符串
    read_offset: int        # 上次成功解码到的 token 位置
    surr_offset: int        # 乱码/截断字符的起始位置 (Surrogate offset)
    sent_offset: int        # 已发送给用户的字符串长度


class DetokenizeManager:
    """
    Detokenizer 管理器
    
    负责将生成的 token IDs 转换回文本，支持流式增量解码。
    能够处理 UTF-8 字符跨 token 分割的情况 (即乱码/ 字符的处理)。
    """
    def __init__(self, tokenizer: LlamaTokenizer) -> None:
        # uid -> DecodeStatus 映射
        self.decode_map: Dict[int, DecodeStatus] = {}
        self.tokenizer = tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id

    def detokenize(self, msgs: List[DetokenizeMsg]) -> List[str]:
        """
        批量执行 Detokenize
        
        参数:
            msgs: 一批待解码的消息，包含新生成的 token ID
            
        返回:
            List[str]: 对应的增量文本输出 (incremental output)
        """
        read_ids: List[List[int]] = []
        surr_ids: List[List[int]] = []
        
        # 1. 更新每个请求的 token ID 列表
        for msg in msgs:
            if msg.uid not in self.decode_map:
                self.decode_map[msg.uid] = DecodeStatus(
                    decoded_ids=[],
                    decoded_str="",
                    read_offset=0,
                    surr_offset=0,
                    sent_offset=0,
                )
            s = self.decode_map[msg.uid]
            
            # 只有非 EOS token 才加入列表 (EOS 通常不显示)
            if not (msg.finished and msg.next_token == self.eos_token_id):
                s.decoded_ids.append(msg.next_token)
            
            # 准备解码用的 ID 片段
            # read_ids: 包含可能乱码的部分，尝试重新解码
            read_ids.append(s.decoded_ids[s.surr_offset :])
            # surr_ids: 之前的乱码部分，用于辅助判断
            surr_ids.append(s.decoded_ids[s.surr_offset : s.read_offset])

        # 2. 批量解码 (这是很耗时的 CPU 操作，所以要批量做)
        read_texts = self.tokenizer.batch_decode(read_ids)
        surr_texts = self.tokenizer.batch_decode(surr_ids)

        incremental_strs: List[str] = []
        for msg, read_str, surr_str in zip(msgs, read_texts, surr_texts, strict=True):
            s = self.decode_map[msg.uid]
            # 获取新解码出的文本 (去掉之前已确认是乱码的部分)
            new_text = read_str[len(surr_str) :]
            
            # Streaming chunk: update the decode status
            # 如果解码成功且没有乱码字符 ("" 通常表示 decoder replacement character)
            if len(new_text) > 0 and not new_text.endswith(""):
                output_str = s.decoded_str + new_text
                s.decoded_str = output_str
                s.surr_offset = s.read_offset
                s.read_offset = len(s.decoded_ids)
            else:
                # 如果有潜在乱码或单词未结束，尝试只取可打印部分
                new_text = find_printable_text(new_text)
                output_str = s.decoded_str + new_text

            # 计算增量输出 (只返回本次新增的字符)
            incremental_output = output_str[s.sent_offset :]
            s.sent_offset = len(output_str)
            incremental_strs.append(incremental_output)
            
            # 请求结束，清理状态
            if msg.finished:
                del self.decode_map[msg.uid]

        return incremental_strs
