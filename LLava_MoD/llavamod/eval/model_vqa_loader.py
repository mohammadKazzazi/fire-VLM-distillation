import argparse
import re
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llavamod.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llavamod.conversation import conv_templates, SeparatorStyle
from llavamod.model.builder import load_pretrained_model
from llavamod.utils import disable_torch_init
from llavamod.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, KeywordsStoppingCriteria
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math



def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if getattr(self.model_config, 'mm_use_im_start_end', False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor

    def __len__(self):
        return len(self.questions)


def get_conv_stop_str(conv) -> str:
    """Stop string for generation; must match each chat template's end-of-turn token."""
    if conv.sep_style in (SeparatorStyle.TWO, SeparatorStyle.GEMMA_2, SeparatorStyle.LLAMA_3):
        return conv.sep2
    if conv.sep_style == SeparatorStyle.QWEN:
        return conv.sep2 or ""
    return conv.sep


def trim_repetitive_answer(text: str) -> str:
    """Trim runaway repetition when the model omits the chat end-of-turn token."""
    text = (text or "").strip()
    if not text:
        return text

    non_fire_tail = (
        "No visible flames or clear smoke indicating an active fire are present."
    )
    if text.startswith("No.") and non_fire_tail in text:
        end = text.find(non_fire_tail) + len(non_fire_tail)
        return text[:end].strip()

    if text.startswith("Yes."):
        sev = "Fire Severity Level:"
        pos = text.find(sev)
        if pos != -1:
            second = text.find(sev, pos + len(sev))
            if second != -1:
                text = text[:second].rstrip(" .")
            else:
                dot = text.find(".", pos)
                if dot != -1:
                    text = text[:dot + 1].strip()

    anchors = (
        "This is a non-fire scene.",
        "No fire or clear smoke indicating an active fire is visible in this image.",
        "Fire Severity Level:",
        "Fire Severity Level Explanation:",
    )
    cut = len(text)
    for anchor in anchors:
        first = text.find(anchor)
        if first == -1:
            continue
        second = text.find(anchor, first + len(anchor))
        if second != -1:
            cut = min(cut, second)

    if cut < len(text):
        text = text[:cut].rstrip(" .")

    seen = set()
    parts = []
    for sent in re.split(r'(?<=[.!?])\s+', text):
        sent = sent.strip()
        if not sent:
            continue
        key = sent.lower()
        if key in seen:
            break
        seen.add(key)
        parts.append(sent)
    return " ".join(parts) if parts else text.strip()


def get_generate_token_kwargs(conv, tokenizer) -> dict:
    """Generation kwargs for chat templates that need explicit pad/eos ids."""
    kwargs = {}
    if conv.sep_style == SeparatorStyle.LLAMA_3:
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        eot_id = tokenizer.convert_tokens_to_ids(conv.sep2)
        if eot_id is not None and eot_id != tokenizer.unk_token_id:
            kwargs["eos_token_id"] = eot_id
        if tokenizer.pad_token_id is not None:
            kwargs["pad_token_id"] = tokenizer.pad_token_id
    elif conv.sep_style == SeparatorStyle.QWEN:
        im_end = conv.sep2 or ""
        im_end_id = tokenizer.convert_tokens_to_ids(im_end)
        if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
            kwargs["eos_token_id"] = im_end_id
        if tokenizer.pad_token_id is not None:
            kwargs["pad_token_id"] = tokenizer.pad_token_id
        elif tokenizer.eos_token_id is not None:
            kwargs["pad_token_id"] = tokenizer.eos_token_id
    return kwargs


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    if args.return_gating_logit is not None:
        from llavamod.utils import get_gating_logit_by_hook
        print(model)
        fea_hooks = get_gating_logit_by_hook(model)
        all_gating_logits = {}
    image_processor = processor['image']
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(
            f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    cnt = -1
    for (input_ids, image_tensor), line in tqdm(zip(data_loader, questions), total=len(questions)):
        cnt += 1
        # if cnt == 30:
        #     break
        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)

        conv = conv_templates[args.conv_mode].copy()
        stop_str = get_conv_stop_str(conv)
        keywords = [stop_str]
        stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)]
        gen_token_kwargs = get_generate_token_kwargs(conv, tokenizer)
        # transformers 4.57+ DynamicCache + multimodal decode: attention_mask extension
        # in llava_arch hits negative dims (past_len vs mask length). Slower but stable.
        # Qwen3/Llama3 already needed this; Qwen2.5/Gemma2 hit it once GenerationMixin was restored.
        use_cache = False

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.bfloat16, device='cuda', non_blocking=True),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=use_cache,
                stopping_criteria=stopping_criteria,
                **gen_token_kwargs,
            )
        if args.return_gating_logit is not None:
            # import ipdb
            # ipdb.set_trace()
            all_gating_logits[cnt] = dict(gating_logit=[i.fea for i in fea_hooks],
                                          images=image_tensor if image_tensor is None else image_tensor.detach().cpu(),
                                          input_ids=input_ids.detach().cpu(),
                                          output_ids=output_ids.detach().cpu())
            print(input_ids.shape, output_ids.shape, fea_hooks[0].fea.shape,
                  image_tensor.shape if image_tensor is not None else [])
            # assert fea_hooks[0].fea.shape[0] + 1 == output_ids.shape[1] + 575
            print('The number of hooks is:', len(fea_hooks))

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = trim_repetitive_answer(outputs.strip())
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)].strip()
        print(outputs)

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()

    if args.return_gating_logit is not None:
        torch.save(all_gating_logits, f'{args.return_gating_logit}.pt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--return_gating_logit", type=str, default=None)
    args = parser.parse_args()

    eval_model(args)
