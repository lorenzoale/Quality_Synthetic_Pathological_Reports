import pandas as pd
import numpy as np
import random
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

import argparse
from tqdm import tqdm
# import torch
import json
import os

import format_prompt

# torch.cuda.empty_cache()
# delimiter between abrege and text
DELIMITER="\n---\n"


def load_RAG_indexes(entries_df, encoder):
    facts = []

    for _, row in entries_df.iterrows():
        abrege = row["Abrégé"]
        organ = row["organ"]
        examples = row["examples"]

        if organ == "ESTOMAC":
            continue
        elif organ == "SEIN (ÉGALEMENT UTILISÉ CHEZ L'HOMME)":
            organ = "SEIN"

        if not isinstance(examples, list):
            continue

        for ex in examples:
            if not isinstance(ex, dict):
                continue

            text = ex.get("report_text", "")
            if isinstance(text, str) and text.startswith("Titre"):
                text = text[len("Titre"):].lstrip()

            text_data = f"{organ if pd.notna(organ) else ''} - {abrege if pd.notna(abrege) else ''}{DELIMITER}{text}"
            facts.append(text_data)

    encoded_facts = encoder.encode(facts, show_progress_bar=True).astype("float32")

    index = faiss.IndexFlatL2(encoded_facts.shape[1])
    index.add(encoded_facts)

    return index, facts    



def to_json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    else:
        return obj

def getRAGexamples (donnees_rag,facts,vectorstore_RAG,encoder,k_value):
    question_encoding_mm = encoder.encode([donnees_rag])
    distances_mm, positions_mm = vectorstore_RAG.search(question_encoding_mm, k=k_value)
    docs=[]
    relevant_facts_mm = []
    ind_ex=0
    for p in positions_mm[0]:
        abrege, text = facts[p].split(DELIMITER, 2)
        if (not text.strip()):
            continue
        ind_ex=ind_ex+1
        relevant_facts_mm.append(f"Example {ind_ex}:\n {text}")
        docs.append(facts[p])
    rag_ex="\n\n".join(relevant_facts_mm)
    
    res={
        "rag_ex":rag_ex,
        "distances_mm":to_json_safe(distances_mm),
        "positions_mm":to_json_safe(positions_mm),
        "docs":docs
    }
        
    return res


# # Set cache folder for Hugging Face
# os.environ["HF_HOME"] = "./cache"           # base cache dir
# os.environ["TRANSFORMERS_OFFLINE"] = "1"    # force offline mode
# os.environ["HF_DATASETS_OFFLINE"] = "1"


def inference_model(data,
                    df_templates_gs, 
                    model_name,
                    encoder_name,
                    k_value,
                    batch_size,
                    greedy,
                    system_prompt):
    #  load the encoder for RAG
    hf_path = os.path.join(os.environ["DSDIR"], 'HuggingFace_Models/')
    model_nameFull = os.path.join(hf_path, model_name)
    local_encoder = os.path.join(hf_path, encoder_name)

    encoder = SentenceTransformer(
        local_encoder,#encoder_name,
        model_kwargs={
            # "attn_implementation": "flash_attention_2",
            "torch_dtype": "bfloat16",   
        },
        tokenizer_kwargs={"padding_side": "left"},
    )
    vectorstore_RAG,facts=load_RAG_indexes(df_templates_gs,encoder)

    
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
    
    # Create a total progress bar
    # pbar = tqdm(total=len(sampled_data), desc="Generating completions")
    pbar = tqdm(total=len(data), desc="Generating completions")
    
    
    # Inference with batch size batch_size (4)
    
    # for i in range(0, len(sampled_data), batch_size):
    #     batch = sampled_data.iloc[i:i + batch_size]   # slice rows safely
    for i in range(0, len(data), batch_size):
        batch = data.iloc[i:i + batch_size]   # slice rows safely
        batch_messages = []
        rag_res_batch = []
        entry_datas=[]
        indexes=[]
        for idx, row in batch.iterrows():
            entry_data = row.to_dict()
            indexes.append(idx)
            entry_datas.append(entry_data)
            
            donnees_cliniques,donnees_Rag=format_prompt.create_donnees_cliniques_rag(entry_data)

            res=getRAGexamples(donnees_Rag,facts,vectorstore_RAG,encoder,k_value)
            RAG_ex=res["rag_ex"]
            rag_res_batch.append(res)
            message= format_prompt.generate_report_prompt(donnees_cliniques, RAG_ex,system_prompt)
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
                max_new_tokens=1300,
                do_sample=False,
                num_beams=4,
                # temperature=0.3,
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
        for idx,entry_data, generated_text,rag_res in zip(indexes,entry_datas, generated_text_batch,rag_res_batch):
            data_to_save.append({
                "index":idx,
                "entry_data": entry_data,
                "rag_res":rag_res,
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
    parser.add_argument("-f", "--template_file", type=str, default="./data/templates/df_final_templates.json")
    parser.add_argument("-o", "--output_file_folder", type=str, default="./output")
    parser.add_argument("-m", "--model_name", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1")
    parser.add_argument("-e", "--encoder_name", type=str, default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("-k", "--k_value", type=str, default=2)
    # parser.add_argument("-m", "--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    # parser.add_argument("-e", "--encoder_name", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("-b", "--batch_size", type=int, default=4)
    parser.add_argument("-gree", "--greedy", type=int, default=1)
    parser.add_argument("-s", "--system_prompt", type=int, default=0)
    args = parser.parse_args()
    greedy=(args.greedy==1)
    system_prompt= (args.system_prompt==1)
    k_value=args.k_value
    
    input_file = args.input_file
    template_file=args.template_file
    model_name = args.model_name
    encoder_name=args.encoder_name
    output_file_folder = args.output_file_folder
    batch_size = args.batch_size
    greedy_text="no_greedy"
    if greedy: 
        greedy_text="greedy"
    save_specific_path = f"{output_file_folder}/output{greedy_text}_{model_name.replace('/', '_')}_RAG_k{k_value}.json"
    print(f"Saving to {save_specific_path}")

    # Load jsonl file
    df_templates_gs=pd.read_json(template_file)
    df_data_patient=pd.read_csv(input_file)

    # Inference
    data_to_save = inference_model(df_data_patient,df_templates_gs, model_name,encoder_name,k_value,batch_size,greedy,system_prompt)

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
