import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

import format_prompt


def json_safe(obj):
    """Convert numpy/pandas scalar objects so json.dumps does not fail."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(json_safe(v) for v in obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if pd.isna(obj) if not isinstance(obj, (list, dict, tuple, np.ndarray)) else False:
        return None
    return obj


def resolve_path(path_or_name, hf_root=None):
    """
    Resolve either:
      - an absolute/relative filesystem path, or
      - a Hugging Face-style model name stored under $DSDIR/HuggingFace_Models.
    """
    p = Path(path_or_name).expanduser()
    if p.exists():
        return str(p)

    if hf_root is None:
        dsd = os.environ.get("DSDIR")
        if dsd:
            hf_root = Path(dsd) / "HuggingFace_Models"

    if hf_root is not None:
        candidate = Path(hf_root).expanduser() / path_or_name
        if candidate.exists():
            return str(candidate)

    # Let transformers/peft handle it. This can still work if online or cached.
    return path_or_name


def apply_chat_template_batch(tokenizer, batch_messages):
    """Gemma/MedGemma does not need enable_thinking; fallback keeps compatibility."""
    try:
        return tokenizer.apply_chat_template(
            batch_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            batch_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

def get_stop_token_ids(tokenizer):
    """
    Stop on both the tokenizer EOS token and Gemma/MedGemma <end_of_turn>.
    SFT training template ended assistant answers with <end_of_turn>,
    so inference must stop on it.
    """
    stop_token_ids = []

    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)

    end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if (
        end_of_turn_id is not None
        and end_of_turn_id != tokenizer.unk_token_id
        and end_of_turn_id not in stop_token_ids
    ):
        stop_token_ids.append(end_of_turn_id)

    return stop_token_ids





def load_model_and_tokenizer(
    base_model_name_or_path,
    lora_adapter_path,
    cache_dir="./cache",
    merge_lora=False,
):
    base_model_path = resolve_path(base_model_name_or_path)
    adapter_path = resolve_path(lora_adapter_path)

    print(f"Base model: {base_model_path}")
    print(f"LoRA adapter: {adapter_path}")

    # tokenizer = AutoTokenizer.from_pretrained(
    #     base_model_path,
    #     cache_dir=cache_dir,
    #     trust_remote_code=True,
    # )
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=False,
    )

    if merge_lora:
        print("Merging LoRA adapter into base model...")
        model = model.merge_and_unload()

    model.eval()
    return tokenizer, model


@torch.inference_mode()
def inference_model(
    data,
    base_model_name_or_path,
    lora_adapter_path,
    batch_size=1,
    greedy=True,
    system_prompt=False,
    cache_dir="./cache",
    max_new_tokens=1200,
    min_new_tokens=200,
    num_beams=4,
    repetition_penalty=1.1,
    temperature=0.2,
    top_p=0.9,
    top_k=0,
    merge_lora=False,
):
    tokenizer, model = load_model_and_tokenizer(
        base_model_name_or_path=base_model_name_or_path,
        lora_adapter_path=lora_adapter_path,
        cache_dir=cache_dir,
        merge_lora=merge_lora,
    )

    stop_token_ids = get_stop_token_ids(tokenizer)
    
    print("EOS token:", tokenizer.eos_token, tokenizer.eos_token_id)
    print("<end_of_turn> token id:", tokenizer.convert_tokens_to_ids("<end_of_turn>"))
    print("Stop token ids:", stop_token_ids)
    
    data_to_save = []
    pbar = tqdm(total=len(data), desc="Generating completions")

    for start in range(0, len(data), batch_size):
        batch = data.iloc[start:start + batch_size]

        batch_messages = []
        entry_datas = []
        indexes = []
        donnees_cliniques_batch = []

        for idx, row in batch.iterrows():
            entry_data = row.to_dict()
            indexes.append(idx)
            entry_datas.append(entry_data)

            donnees_cliniques = format_prompt.create_donnees_cliniques(entry_data)
            message = format_prompt.generate_report_prompt(
                donnees_cliniques,
                system_prompt=system_prompt,
            )

            donnees_cliniques_batch.append(donnees_cliniques)
            batch_messages.append(message)

        text_batch = apply_chat_template_batch(tokenizer, batch_messages)

        model_inputs_batch = tokenizer(
            text_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192,
        ).to(model.device)

        if greedy:
            generated_ids_batch = model.generate(
                **model_inputs_batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                length_penalty=1.0,
                early_stopping=True if num_beams > 1 else False,
                pad_token_id=tokenizer.pad_token_id,
                # eos_token_id=tokenizer.eos_token_id,
                eos_token_id=stop_token_ids,
            )
        else:
            generated_ids_batch = model.generate(
                **model_inputs_batch,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=1.0,
                pad_token_id=tokenizer.pad_token_id,
                # eos_token_id=tokenizer.eos_token_id,
                eos_token_id=stop_token_ids,
            )

        generated_ids_batch = generated_ids_batch[:, model_inputs_batch.input_ids.shape[1]:]
        generated_text_batch = tokenizer.batch_decode(
            generated_ids_batch,
            skip_special_tokens=True,
        )

        for idx, entry_data, donnees_cliniques, generated_text in zip(
            indexes,
            entry_datas,
            donnees_cliniques_batch,
            generated_text_batch,
        ):
            data_to_save.append({
                "index": int(idx) if isinstance(idx, (np.integer, int)) else idx,
                "entry_data": json_safe(entry_data),
                "donnees_cliniques": donnees_cliniques,
                "generated_text": generated_text.strip(),
            })

        pbar.update(len(batch))
        pbar.set_postfix({"done": f"{min(start + batch_size, len(data))}/{len(data)}"})

    pbar.close()
    return data_to_save


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_file", type=str, required=True)
    parser.add_argument("-o", "--output_file_folder", type=str, default="./output_finetuned")

    parser.add_argument(
        "-bm",
        "--base_model_name_or_path",
        type=str,
        default="google/medgemma-27b-text-it",
        help="Base model path or name. If not found, $DSDIR/HuggingFace_Models/<value> is tried.",
    )
    parser.add_argument(
        "-ad",
        "--lora_adapter_path",
        type=str,
        default="../models/medgemma-27b-report-generation-lora",
        help="Fine-tuned LoRA adapter path. If not found, $DSDIR/HuggingFace_Models/<value> is tried.",
    )

    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("-gree", "--greedy", type=int, default=1)
    parser.add_argument("-s", "--system_prompt", type=int, default=0)
    parser.add_argument("--cache_dir", type=str, default="./cache")

    parser.add_argument("--max_new_tokens", type=int, default=1200)
    parser.add_argument("--min_new_tokens", type=int, default=200)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--merge_lora", type=int, default=0)

    args = parser.parse_args()

    greedy = args.greedy == 1
    system_prompt = args.system_prompt == 1
    merge_lora = args.merge_lora == 1

    os.makedirs(args.output_file_folder, exist_ok=True)

    greedy_text = "greedy" if greedy else "no_greedy"
    adapter_name = Path(args.lora_adapter_path.rstrip("/")).name
    input_stem = Path(args.input_file).stem
    save_specific_path = (
        Path(args.output_file_folder)
        / f"{input_stem}_{greedy_text}_{adapter_name}_bs{args.batch_size}_beams{args.num_beams}.json"
    )

    print(f"Input: {args.input_file}")
    print(f"Saving to: {save_specific_path}")

    df_data_patient = pd.read_csv(args.input_file)
    print(f"Rows: {len(df_data_patient)}")

    data_to_save = inference_model(
        data=df_data_patient,
        base_model_name_or_path=args.base_model_name_or_path,
        lora_adapter_path=args.lora_adapter_path,
        batch_size=args.batch_size,
        greedy=greedy,
        system_prompt=system_prompt,
        cache_dir=args.cache_dir,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        merge_lora=merge_lora,
    )

    with open(save_specific_path, "w", encoding="utf-8") as outfile:
        json.dump(data_to_save, outfile, ensure_ascii=False, indent=2)

    print(f"Saved to {save_specific_path}")


if __name__ == "__main__":
    main()
