import argparse
import os
from datasets import Dataset
from transformers import LlamaTokenizer, AutoConfig
from modeling_llama import LlamaForCausalLM
import torch
import torch.nn as nn
import numpy as np
import jsonlines
import random
from model import Effective_Draft_Decoder
import time
from tqdm import tqdm


os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
seed_val = 888 
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
random.seed(seed_val)



parser = argparse.ArgumentParser(description="推理超参数设置")
parser.add_argument("--num_layers", type=int, default=1, help="模型层数")
parser.add_argument("--model_checkpoint", type=str, default="./llama-2-7b-chat-hf", help="模型检查点路径")
parser.add_argument("--data_dir", type=str, default="./data/humaneval.jsonl", help="Path to the data file")
parser.add_argument("--draft_model_checkpoint", type=str, default="./draft_model.pt", help="Path to the draft model checkpoint")
parser.add_argument("--hidden_layer", type=int, default=-4, help="隐藏层索引")
args = parser.parse_args()

# 超参数
num_layers = args.num_layers
model_checkpoint = args.model_checkpoint
data_dir = args.data_dir
draft_model_model_checkpoint = args.draft_model_checkpoint
hidden_layer = args.hidden_layer


# 加载数据
def load_data():
    def preprocess_function(examples):
        tokenizer.padding_side = "left"
        prompt = tokenizer(
            examples["prompt"],
        )
        tokenizer.padding_side = "right"
        pred = tokenizer(
            examples["pred"],
            add_special_tokens=False,
        )
        return {"input_ids": prompt["input_ids"], "labels": pred["input_ids"], 
                "input_attention_mask": prompt["attention_mask"], "labels_attention_mask": pred["attention_mask"]}

    test_data = []
    with jsonlines.open(data_dir) as f:
        for line in f:
            test_data.append({"prompt": line['prompt'],  "pred": line['pred']})

    dataset_test = Dataset.from_list(test_data)
    print(dataset_test)
    test_tokenized_datasets = dataset_test.map(preprocess_function, batched=True, num_proc=16, remove_columns=['prompt', 'pred'])
    test_tokenized_datasets.set_format("torch")
    return test_tokenized_datasets




def test(batch, llm_model, small_model, eos_id):
    iter_num, sum_acc_num, iter_num_samll = 0, 0, 0
    for i in tqdm(range(len(batch)), desc="Testing Progress"):  # 使用 tqdm 包裹循环
        input_ids, attention_mask_input, label_num = batch[i]["input_ids"].unsqueeze(0).cuda(), batch[i]["input_attention_mask"].unsqueeze(0).cuda(), batch[i]["labels_attention_mask"].sum().item()
        cur_num, past_key_values, pred_ids = 0, None, None
        while cur_num < label_num:
            iter_num += 1
            if past_key_values is None: # 使用LLMs进行首次编码，并生成一个新的token
                outputs = llm_model(input_ids, attention_mask=attention_mask_input, output_hidden_states=True)
                past_key_values, output_hidden_states = outputs.past_key_values, outputs.hidden_states[hidden_layer]
                new_token_1 = outputs.logits[:, -1:, :].argmax(dim=-1)
                cur_num += 1
            last_past_key_values_len = past_key_values[0][0].shape[2]
            # 生成draft
            small_pred, verify_mask, root, verify_position_ids = small_model.generate(encoder_out=output_hidden_states, decoder_inp_token=new_token_1, threshold=0.036)
            # 使用tree mask进行并行验证
            outputs = llm_model(torch.cat([new_token_1, small_pred], dim=1),
                                attention_mask=torch.cat([attention_mask_input, torch.ones(1, 1+small_pred.shape[1]).cuda()], dim=1),
                                past_key_values=past_key_values, output_hidden_states=True,
                                verify_mask=verify_mask, verify_position_ids=verify_position_ids,
                                )

            llm_pred = outputs.logits[:, -small_pred.shape[1]-1:, :].argmax(dim=-1)
            new_token_2, llm_verify_token = llm_pred[:, 0], llm_pred[:, 1:] 

            # 判断是否接收
            max_draft_len = verify_position_ids.max().item() 
            llm_verify_token, cur_llm_p = llm_verify_token.cpu(), new_token_2.item()
            acc_token, acc_ids, acc_num = [cur_llm_p], [], 1
            cur_node = root
            for _ in range(max_draft_len):
                cur_node = cur_node.get_child(cur_llm_p)
                if cur_node is not None: # 存在未终止的候选
                    acc_num += 1
                    cur_llm_p = llm_verify_token[0, cur_node.idx]
                    acc_token.append(cur_llm_p)
                    acc_ids.append(cur_node.idx+1)
                else:
                    break
            cur_num += acc_num
            sum_acc_num += acc_num
            iter_num_samll += max_draft_len
            pred_ids = torch.cat([new_token_1, torch.tensor(acc_token).cuda().unsqueeze(0)], dim=1) if pred_ids is None else \
                        torch.cat([pred_ids, torch.tensor(acc_token).cuda().unsqueeze(0)], dim=1)
            
            # 如果生成终止符，则停止生成
            if (pred_ids == eos_id).any():
                break

            # 更细更新past_key_values和soft prompt
            past_key_values = list(outputs.past_key_values)
            for x in range(len(past_key_values)):
                past_key_values[x] = list(past_key_values[x])
                for y in range(len(past_key_values[x])):
                    past_key_values[x][y] = torch.cat([past_key_values[x][y][:, :, :last_past_key_values_len+1, :], 
                                                    past_key_values[x][y][:, :, [x + last_past_key_values_len for x in acc_ids], :]], dim=2)
            output_hidden_states = torch.cat([output_hidden_states, outputs.hidden_states[hidden_layer][:, 0, :].unsqueeze(1), outputs.hidden_states[hidden_layer][:, acc_ids, :]], dim=1)
            new_token_1, attention_mask_input = pred_ids[:, -1:], torch.cat([attention_mask_input, torch.ones(1, acc_num).cuda()], dim=1)
            
    print("平均接收率:", sum_acc_num / iter_num_samll, "平均生成长度:", sum_acc_num / iter_num)



if __name__ == "__main__":
    # 加载LLM
    tokenizer_kwargs = {
            "use_fast": True,
            "padding_side":'left'
        }
    tokenizer = LlamaTokenizer.from_pretrained(model_checkpoint, **tokenizer_kwargs)
    tokenizer.save_pretrained('./llama_tok')
    tokenizer = LlamaTokenizer.from_pretrained('./llama_tok/')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    torch_dtype = getattr(torch, "bfloat16")
    model = LlamaForCausalLM.from_pretrained(model_checkpoint , torch_dtype=torch.bfloat16, device_map="auto",)
    # 加载草稿模型
    config = AutoConfig.from_pretrained(model_checkpoint)
    Draft_Decoder = Effective_Draft_Decoder(config.hidden_size, config.hidden_size * 2, config.num_attention_heads, num_layers, config)
    Draft_Decoder.lm_head.load_state_dict(model.lm_head.state_dict()) 
    Draft_Decoder.embedding_layer.load_state_dict(model.model.embed_tokens.state_dict())
    Draft_Decoder = Draft_Decoder.bfloat16()
    Draft_Decoder.load_state_dict(torch.load(draft_model_model_checkpoint))
    model = model.cuda().eval()
    Draft_Decoder = Draft_Decoder.cuda().bfloat16().eval()
    # 加载数据
    test_tokenized_datasets = load_data()

    print("start testing...")
    start_time = time.time()
    with torch.no_grad():
        test(test_tokenized_datasets, model, Draft_Decoder, tokenizer.eos_token_id)
    end_time = time.time()
    print(f"{end_time - start_time:.4f} seconds")
