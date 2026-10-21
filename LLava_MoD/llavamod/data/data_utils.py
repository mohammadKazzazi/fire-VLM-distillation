import copy
from typing import Dict, Sequence

import torch

from llavamod.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN, DEFAULT_VIDEO_TOKEN, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN, MAX_IMAGE_LENGTH, \
    MAX_VIDEO_LENGTH

from llavamod import conversation as conversation_lib
from llavamod.model import *
from llavamod.mm_utils import tokenizer_image_token
from PIL import Image

local_rank = None


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
        sources: Sequence[str],
        data_args
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        # try:
        for sentence in source:

            # ======================================================================================================
            if sentence['value'].startswith(DEFAULT_IMAGE_TOKEN) or sentence['value'].startswith(
                    DEFAULT_VIDEO_TOKEN):  # run with multi-im, multi-vid, multi-im & multi-vid
                # <video><video><image><image>\nxxxxxxxxxxxxx  # must <video> first
                # <image>\nxxxxxxxxxxxxx -> <image>\nxxxxxxxxxxxxx
                # <video>\nxxxxxxxxxxxxx -> <video>\nxxxxxxxxxxxxx

                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN,
                                                                  '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')

                IMAGE_TOKEN_NUM = sentence['value'].count(DEFAULT_IMAGE_TOKEN)
                if IMAGE_TOKEN_NUM > MAX_IMAGE_LENGTH:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN * IMAGE_TOKEN_NUM,
                                                                  DEFAULT_IMAGE_TOKEN * MAX_IMAGE_LENGTH).strip()
                VIDEO_TOKEN_NUM = sentence['value'].count(DEFAULT_VIDEO_TOKEN)
                if VIDEO_TOKEN_NUM > MAX_VIDEO_LENGTH:
                    raise ValueError(f"{sentence['value']}")
                    sentence['value'] = sentence['value'].replace(DEFAULT_VIDEO_TOKEN * VIDEO_TOKEN_NUM,
                                                                  DEFAULT_VIDEO_TOKEN * MAX_VIDEO_LENGTH).strip()

            # a <video> is treated as `num_frames * <image>`
            replace_token, vid_replace_token = DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_TOKEN * data_args.num_frames
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                vid_replace_token = DEFAULT_VID_START_TOKEN + vid_replace_token + DEFAULT_VID_END_TOKEN

            # <video><video><image><image>\nxxxxxxxxxxxxx -> `num_frames*<image>``num_frames*<image>`<image><image>\nxxxxxxxxxxxxx
            # <video>\nxxxxxxxxxxxxx -> `num_frames*<image>`\nxxxxxxxxxxxxx
            # print('before replace_token:', [sentence['value']])
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
            sentence['value'] = sentence['value'].replace(DEFAULT_VIDEO_TOKEN, vid_replace_token)
            # print('after replace_token:', [sentence['value']])
            # ======================================================================================================
        # except Exception as e:
        #     print(e)
        #     print(source)
    return sources


def preprocess_llama_2(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def _llama_3_build_prompt(conv, messages, add_generation_prompt=False):
    header = lambda role: f"<|start_header_id|>{role}<|end_header_id|>\n\n"
    ret = conv.sep
    if conv.system:
        ret += header("system") + conv.system + conv.sep2
    for i, (role, message) in enumerate(messages):
        ret += header(role) + message
        if add_generation_prompt and i == len(messages) - 1 and role == conv.roles[0]:
            ret += conv.sep2 + header(conv.roles[1])
            break
        ret += conv.sep2
    return ret


def preprocess_llama_3(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_3

    def tokenize_prompt(prompt):
        if has_image:
            return tokenizer_image_token(prompt, tokenizer, return_tensors='pt')
        return torch.tensor(tokenizer(prompt).input_ids, dtype=torch.long)

    input_ids_list = []
    targets_list = []

    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            messages.append((role, sentence["value"]))

        full = tokenize_prompt(_llama_3_build_prompt(conv, messages))
        target = torch.full_like(full, IGNORE_INDEX)

        for j, (role, _) in enumerate(messages):
            if role != conv.roles[1]:
                continue
            prompt_len = len(tokenize_prompt(_llama_3_build_prompt(conv, messages[:j], add_generation_prompt=True)))
            through_len = len(tokenize_prompt(_llama_3_build_prompt(conv, messages[:j + 1])))
            target[prompt_len:through_len] = full[prompt_len:through_len]

        input_ids_list.append(full)
        targets_list.append(target)

    if len(input_ids_list) == 1:
        input_ids = input_ids_list[0].unsqueeze(0)
        targets = targets_list[0].unsqueeze(0)
    else:
        max_len = min(max(len(x) for x in input_ids_list), tokenizer.model_max_length)
        input_ids = torch.full((len(input_ids_list), max_len), tokenizer.pad_token_id, dtype=torch.long)
        targets = torch.full((len(input_ids_list), max_len), IGNORE_INDEX, dtype=torch.long)
        for row, (ids, lbl) in enumerate(zip(input_ids_list, targets_list)):
            seq_len = min(len(ids), max_len)
            input_ids[row, :seq_len] = ids[:seq_len]
            targets[row, :seq_len] = lbl[:seq_len]

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def _qwen_build_prompt(conv, messages, add_generation_prompt=False):
    ret = ""
    if conv.system:
        ret += f"{conv.sep}system\n{conv.system}{conv.sep2}\n"
    for i, (role, message) in enumerate(messages):
        ret += f"{conv.sep}{role}\n{message}"
        if add_generation_prompt and i == len(messages) - 1 and role == conv.roles[0]:
            ret += f"{conv.sep2}\n{conv.sep}{conv.roles[1]}\n"
            break
        ret += f"{conv.sep2}\n"
    return ret


def preprocess_qwen(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    assert conv.sep_style == conversation_lib.SeparatorStyle.QWEN

    def tokenize_prompt(prompt):
        if has_image:
            return tokenizer_image_token(prompt, tokenizer, return_tensors='pt')
        return torch.tensor(tokenizer(prompt).input_ids, dtype=torch.long)

    input_ids_list = []
    targets_list = []

    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            messages.append((role, sentence["value"]))

        full = tokenize_prompt(_qwen_build_prompt(conv, messages))
        target = torch.full_like(full, IGNORE_INDEX)

        for j, (role, _) in enumerate(messages):
            if role != conv.roles[1]:
                continue
            prompt_len = len(tokenize_prompt(_qwen_build_prompt(conv, messages[:j], add_generation_prompt=True)))
            through_len = len(tokenize_prompt(_qwen_build_prompt(conv, messages[:j + 1])))
            target[prompt_len:through_len] = full[prompt_len:through_len]

        input_ids_list.append(full)
        targets_list.append(target)

    if len(input_ids_list) == 1:
        input_ids = input_ids_list[0].unsqueeze(0)
        targets = targets_list[0].unsqueeze(0)
    else:
        max_len = min(max(len(x) for x in input_ids_list), tokenizer.model_max_length)
        input_ids = torch.full((len(input_ids_list), max_len), tokenizer.pad_token_id, dtype=torch.long)
        targets = torch.full((len(input_ids_list), max_len), IGNORE_INDEX, dtype=torch.long)
        for row, (ids, lbl) in enumerate(zip(input_ids_list, targets_list)):
            seq_len = min(len(ids), max_len)
            input_ids[row, :seq_len] = ids[:seq_len]
            targets[row, :seq_len] = lbl[:seq_len]

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # print('00000000000', sources)
    # Apply prompt templates
    conversations = []

    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    # Tokenize conversations
    # print('before tokenizer_image_token', conversations)
    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_phi(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    sep = conv.sep + conv.roles[1] + ": "

    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) + 1  # for eos_token
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids) + 1  # for eos_token
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX  # instruction_len is before the answer

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:

            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_openchat(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []

    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            # print('i rou, parts\n', i, rou, parts)
            if len(parts) != 2:
                break
            parts[0] += sep
            # print('after add sep, parts\n', parts)

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX  # instruction_len is before the answer

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations],
                            dim=0)
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx + 2]))  # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer_image_token(rou, tokenizer)) + len(tokenizer_image_token(conv.sep, tokenizer))
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_gemma_2(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.GEMMA_2

    # Mask targets
    sep = "<start_of_turn>" + conv.sep + conv.roles[1] + "\n"
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                print(f"WARNING: parts!=: {parts}")
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1  # exclude <bos>
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1  # exclude <bos>

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    # print('sources', sources)
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # print('conversations', conversations)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    # print('after tokenizer_image_token', input_ids)
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    # print('target:', target)
    return dict(input_ids=input_ids, labels=targets)


def preprocess(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_3:
        return preprocess_llama_3(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.QWEN:
        return preprocess_qwen(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("phi"):
        return preprocess_phi(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("stablelm"):  # stablelm same as phi
        return preprocess_phi(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("openchat") or \
            conversation_lib.default_conversation.version.startswith("mistral"):  # for openchat
        return preprocess_openchat(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("minicpm"):  # minicpm same as openchat
        return preprocess_openchat(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)
    elif conversation_lib.default_conversation.version.startswith("gemma_2"):
        return preprocess_gemma_2(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
