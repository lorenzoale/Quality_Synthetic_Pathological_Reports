import pandas as pd
import numpy as np
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
from sample_dataFrame_values import sample_dataframe

import argparse
from tqdm import tqdm
# import torch
import json
import os

import format_prompt

# torch.cuda.empty_cache()


def inference_model(data,model_name,batch_size,greedy,system_prompt):
    #  load the encoder for RAG
    hf_path = os.path.join(os.environ["DSDIR"], 'HuggingFace_Models/')
    model_nameFull = os.path.join(hf_path, model_name)

    
    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_nameFull, cache_dir = "./cache")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_nameFull,
        torch_dtype="auto",
        device_map="auto",
        cache_dir = "./cache"
    )

    # prepare the model input
    
    data_to_save = []
    data_index = 1

    # sampled_data=data.sample(n=100)
    # replace all range values by sampled data 
    # data = sample_dataframe(ranges_data, seed=42)
    
    # Create a total progress bar
    # pbar = tqdm(total=len(sampled_data), desc="Generating completions")
    pbar = tqdm(total=len(data), desc="Generating completions")
    
    # Inference with batch size batch_size (4)
    
    # for i in range(0, len(sampled_data), batch_size):
    #     batch = sampled_data.iloc[i:i + batch_size]   # slice rows safely
    for i in range(0, len(data), batch_size):
        batch = data.iloc[i:i + batch_size]   # slice rows safely
        batch_messages = []
        entry_datas=[]
        indexes=[]
        for idx, row in batch.iterrows():
            entry_data = row.to_dict()
            indexes.append(idx)
            entry_datas.append(entry_data)
            donnes_cliniques=format_prompt.create_donnees_cliniques(entry_data)
            message= format_prompt.generate_report_prompt(donnes_cliniques,system_prompt)
            batch_messages.append(message)        

        text_batch = tokenizer.apply_chat_template(
            batch_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
        )

        model_inputs_batch = tokenizer(text_batch, return_tensors="pt", padding=True).to(model.device)

        # conduct text completion
        if greedy:
            generated_ids_batch = model.generate(
                **model_inputs_batch,
                max_new_tokens=1200,
                do_sample=False,
                num_beams=1, 
                repetition_penalty=1.1,
                length_penalty=1.0,
                early_stopping=True,
            )
        else:
            # not greedy: 
            generated_ids_batch = model.generate(
                **model_inputs_batch,
                max_new_tokens=1300,
                min_new_tokens=200,
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
                top_k=0,
                repetition_penalty=1.1,
                length_penalty=1.0,
            )
        generated_ids_batch = generated_ids_batch[:, model_inputs_batch.input_ids.shape[1]:]

        # the result will begin with thinking content in <think></think> tags, followed by the actual response
        generated_text_batch = tokenizer.batch_decode(generated_ids_batch, skip_special_tokens=True)

        # Filter out the thinking content
        # generated_text = generated_text.split("</think>")[1].strip()

        # Save the result
        for idx,entry_data, generated_text in zip(indexes,entry_datas, generated_text_batch):
            data_to_save.append({
                "index":idx,
                "entry_data": entry_data,
                "generated_text": generated_text,
            })

        # Update the progress bar after each task
        pbar.update(batch_size)
        pbar.set_postfix({
            'data_index': f"{data_index}/{len(data)/batch_size:.2f}",
        })

    data_index += batch_size
    
    pbar.close()
    return data_to_save





def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", type=str, default="$WORK/NewReportsGeneration/data/train.csv")
    parser.add_argument("-o", "--output_file_folder", type=str, default="./output")
    parser.add_argument("-m", "--model_name", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1")
    parser.add_argument("-b", "--batch_size", type=int, default=4)
    parser.add_argument("-gree", "--greedy", type=int, default=1)
    parser.add_argument("-s", "--system_prompt", type=int, default=0)
    args = parser.parse_args()
    greedy=(args.greedy==1)
    system_prompt= (args.system_prompt==1)

    input_file = args.input_file
    model_name = args.model_name
    output_file_folder = args.output_file_folder
    batch_size = args.batch_size
    greedy_text="no_greedy"
    if greedy: 
        greedy_text="greedy"
    save_specific_path = f"{output_file_folder}/1beam_output{greedy_text}_{model_name.replace('/', '_')}.json"
    print(f"Saving to {save_specific_path}")
    # Make sure the folder exists
    os.makedirs(output_file_folder, exist_ok=True)

    # Load jsonl file
    
    df_data_patient=pd.read_csv(input_file)

    # Inference
    data_to_save = inference_model(df_data_patient,model_name,batch_size,greedy,system_prompt)

    # Save the results
    input_file_name = input_file.split("/")[-1].split(".")[0]
    
    # with open(save_specific_path, "w") as f:
    #     for item in data_to_save:
    #         json.dump(item, f, ensure_ascii=False)
    #         f.write("\n")

    df_res_json = json.dumps(data_to_save,ensure_ascii=False)  #default=str,indent=4)
    with open(save_specific_path, "w") as outfile:
        outfile.write(df_res_json)            

    print(f"Saved to {save_specific_path}")

if __name__ == "__main__":
    main()
