import os
import argparse
from datasets import Dataset
from transformers import LlamaTokenizer, AutoConfig
from modeling_llama import LlamaForCausalLM
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch import nn
import numpy as np
import jsonlines
import random
from model import Effective_Draft_Decoder

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
seed_val = 888 
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
random.seed(seed_val)


# 解析命令行参数
parser = argparse.ArgumentParser(description="训练模型的超参数设置")
parser.add_argument("--min_length", type=int, default=128, help="最小序列长度")
parser.add_argument("--max_length", type=int, default=2048, help="最大序列长度")
parser.add_argument("--model_checkpoint", type=str, default="./llama-2-7b-chat-hf", help="模型检查点路径")
parser.add_argument("--epoch", type=int, default=1, help="训练轮数")
parser.add_argument("--dir", type=str, default="./data/sharegpt_llama.jsonl", help="数据文件路径")
parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="梯度累积步数")
parser.add_argument("--warm_up_iter", type=int, default=1000, help="预热迭代次数")
parser.add_argument("--save_step", type=int, default=1000, help="保存模型的步数")
parser.add_argument("--hidden_layer", type=int, default=-4, help="隐藏层索引")
parser.add_argument("--lr_max", type=float, default=1e-4, help="学习率")
parser.add_argument("--num_layers", type=int, default=1, help="模型层数")
parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
args = parser.parse_args()


# 超参数
min_length, max_length = args.min_length, args.max_length
model_checkpoint = args.model_checkpoint
epoch = args.epoch
dir = args.dir
gradient_accumulation_steps = args.gradient_accumulation_steps
warm_up_iter = args.warm_up_iter
save_step = args.save_step
hidden_layer = args.hidden_layer
lr_max = args.lr_max
num_layers = args.num_layers
batch_size = args.batch_size

# 加载LLM
tokenizer_kwargs = {"use_fast": True, "padding_side":'left'}
tokenizer = LlamaTokenizer.from_pretrained(model_checkpoint, **tokenizer_kwargs)
tokenizer.save_pretrained('./llama_tok')
tokenizer = LlamaTokenizer.from_pretrained('./llama_tok/')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
torch_dtype = getattr(torch, "bfloat16")
model = LlamaForCausalLM.from_pretrained(model_checkpoint, torch_dtype=torch.bfloat16)


# 初始化草稿模型
config = AutoConfig.from_pretrained(model_checkpoint)
Draft_Decoder = Effective_Draft_Decoder(config.hidden_size, config.hidden_size * 2, config.num_attention_heads, num_layers, config)
Draft_Decoder.lm_head.load_state_dict(model.lm_head.state_dict()) 
Draft_Decoder.embedding_layer.load_state_dict(model.model.embed_tokens.state_dict())
p_num = sum(p.numel() for p in Draft_Decoder.parameters() if p.requires_grad)
# print("参数量：", p_num)
Draft_Decoder = Draft_Decoder.bfloat16()


def preprocess_function(examples):
    '''
        数据处理函数，数据格式：
        {
            "prompt": "prompt text",
            "pred": "predicted text"
        }
    '''
    prompt = tokenizer(
        examples["prompt"],
    )
    pred = tokenizer(
        examples["pred"],
        add_special_tokens=False,
    )
    return {"input_ids": prompt["input_ids"], "labels": pred["input_ids"]}


# 加载训练数据
train_data = []
with jsonlines.open(dir) as f:
    num = 0
    for line in f:
        num += 1
        train_data.append({"prompt": line['prompt'] + line['pred'], "pred": line['pred']})
dataset_train = Dataset.from_list(train_data)
train_tokenized_datasets = dataset_train.map(preprocess_function, batched=True, num_proc=16, remove_columns=['prompt', 'pred'],
                                            load_from_cache_file=True,
                                            cache_file_name="/mnt/local/zhr/data/ShareGPT/tokenized.arrow",
                                        )
train_tokenized_datasets = train_tokenized_datasets.filter(lambda x: len(x["labels"]) > 10)
train_tokenized_datasets = train_tokenized_datasets.filter(lambda x: len(x["input_ids"]) > min_length)
train_tokenized_datasets = train_tokenized_datasets.filter(lambda x: len(x["input_ids"]) < max_length)
print(train_tokenized_datasets)
print(tokenizer.decode(train_tokenized_datasets[0]["input_ids"]))
print(tokenizer.decode(train_tokenized_datasets[0]["labels"]))
train_tokenized_datasets.set_format("torch")
train_dataloader = DataLoader(
    train_tokenized_datasets, batch_size=batch_size, shuffle=True,
)


# 设置学习率调度器
T_max = len(train_dataloader) * epoch	# 周期
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, Draft_Decoder.parameters()), lr=lr_max)
# 余弦退火学习率调度器
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max - warm_up_iter)
# 自定义热身学习率调度器
class WarmUpLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, base_lr, max_lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.max_lr = max_lr
        super(WarmUpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [self.base_lr + (self.max_lr - self.base_lr) * self.last_epoch / self.warmup_epochs for _ in self.optimizer.param_groups]
        return [self.max_lr for _ in self.optimizer.param_groups]
# 初始化热身调度器
warmup_scheduler = WarmUpLR(optimizer, warm_up_iter, base_lr=0, max_lr=lr_max)


# 使用Accelerator进行分布式训练
accelerator = Accelerator()
model, train_dataloader, Draft_Decoder, optimizer, cosine_scheduler, warmup_scheduler = accelerator.prepare(model, train_dataloader, Draft_Decoder, optimizer, cosine_scheduler, warmup_scheduler)
# 训练
progress_bar = tqdm(range(len(train_dataloader) * epoch))
num = 0
min_loss = 9999
for i in range(epoch):
    for step, batch in enumerate(train_dataloader):
        batch["input_ids"] = batch["input_ids"].to('cuda:0')
        labels = batch["input_ids"].clone()
        labels[:, :-batch["labels"].shape[1]] = -100
        with torch.no_grad():
            model.eval()
            outputs = model(batch["input_ids"], output_hidden_states=True, labels=labels)
        loss, kl_loss  = Draft_Decoder(outputs.hidden_states[hidden_layer], 
                                        batch["input_ids"], 
                                        llm_logits=outputs.logits[:, -batch["labels"].shape[1]:, :], 
                                    )
        progress_bar.update(1)
        loss = kl_loss
        loss = loss / gradient_accumulation_steps
        accelerator.backward(loss)
        if step % gradient_accumulation_steps == 0:
            num += 1
            optimizer.step()
            if num < warm_up_iter:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()
            optimizer.zero_grad()
        if num % save_step == 0:
            print(f"loss: {loss}, kl_loss: {kl_loss}, llm_loss: {outputs.loss}, p: {(num * gradient_accumulation_steps / len(train_dataloader) * epoch)}")
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(Draft_Decoder)
            torch.save(unwrapped_model.state_dict(), f"./draft_model_{num}.pt")
        

