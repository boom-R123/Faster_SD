from modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
import torch
import torch.nn.functional as F
from torch import nn
import random
from torch.nn import CrossEntropyLoss
from typing import List, Dict, Any, Optional



class TreeNode:
    """树节点类"""
    def __init__(self, value: Any):
        self.value = value
        self.children = []
        self.parent = None
        self.idx = -1 # 展平后的idx
    
    def add_child(self, child_node):
        """添加子节点"""
        child_node.parent = self
        self.children.append(child_node)
    
    def __repr__(self):
        return f"TreeNode({self.value})"
    
    def has_child(self, value: Any) -> bool:
        """检查是否有指定值的子节点"""
        return any(child.value == value for child in self.children)
    
    def get_child(self, value: Any) -> Optional['TreeNode']:
        """获取指定值的子节点"""
        for child in self.children:
            if child.value == value:
                return child
        return None
    
    def flatten(self, seq: List[int],  position_id: List[int] = None):
        """通过层次遍历将树展平为一维序列"""
        queue = [self]
        layer = 0
        num = 0
        while queue:
            level = []
            next_queue = []
            for node in queue:
                level.append(node)
                next_queue.extend(node.children)
            for node in level:
                if node.value == -1:
                    break
                node.idx = num
                num += 1
                seq += [node.value]
                position_id += [layer]
            layer += 1
            queue = next_queue

    def get_mask(self, mask, parent_ids):
        """获取当前节点的mask"""
        if self.value != -1:
            mask[self.idx, self.idx] = False
            mask[self.idx, parent_ids] = False
            new_parent_ids = parent_ids + [self.idx]
        for child in self.children:
            child.get_mask(mask, new_parent_ids if self.value != -1 else parent_ids) 

    

def Flatten_tree(lst):
    """
    将树展平为一维序列。
    """
    device = lst[0].device
    root = TreeNode(-1) 
    for i in range(len(lst)):
        lst[i] = lst[i].cpu()
    for ts in lst:
        for r in range(ts.shape[0]):
            parent = root
            for c in range(ts.shape[1]):
                if parent.has_child(ts[r, c]):
                # 如果当前节点的值已经存在于子节点中，则无需添加
                    parent = parent.get_child(ts[r, c])
                    continue
                # 如果当前节点的值不存在于子节点中，则添加新节点
                else:
                    new_node = TreeNode(ts[r, c].item())
                    parent.add_child(new_node)
                    parent = new_node
    seq, position_ids = [], []
    root.flatten(seq, position_ids)
    attention_mask = torch.ones((len(seq), len(seq)), dtype=torch.bool)
    root.get_mask(attention_mask, [])
    seq = torch.tensor(seq, dtype=torch.long, device=device)
    position_ids = torch.tensor(position_ids, dtype=torch.long, device=device)
    attention_mask = attention_mask.to(device)
    return seq, attention_mask, root, position_ids




# 以LLM编码的结果为输入，构造NAR模型
class Effective_Draft_Decoder(nn.Module):
    def __init__(self, hidden_size, dim_feedforward, head_num, num_layers, config):
        super(Effective_Draft_Decoder, self).__init__()
        self.embedding_layer = nn.Embedding(32000, hidden_size)
        self.decoder = LlamaDecoderLayer(config)
        self.lm_head = nn.Linear(hidden_size, 32000, bias=False)
        self.norm = LlamaRMSNorm(hidden_size)

        
    def forward(self, encoder_out, labels, llm_logits):
        '''
            encoder_out: LLM编码的结果，shape为(batch_size, seq_len, hidden_size)
            labels: 目标token的ID，shape为(batch_size, seq_len)
            llm_logits: LLM的输出，shape为(batch_size, seq_len, vocab_size)
        '''
        pred_len = random.randint(5, 10) # 随机划分block的长度
        inp_len = encoder_out.shape[1] - llm_logits.shape[1]
        label_len = llm_logits.shape[1]

        # 获得输入的embedding
        input_ids = labels[:, inp_len:]
        input_embeds = self.embedding_layer(input_ids)

        # 将LLM输出的向量作为soft-prompt输入，用于后续token的生成
        hidden_states = torch.cat([encoder_out, input_embeds], dim=1)
        # 重新构造position_ids
        position_ids = torch.arange(encoder_out.shape[1], dtype=torch.long, device=hidden_states.device)
        position_ids = torch.cat([position_ids, torch.arange(inp_len, inp_len + label_len, dtype=torch.long, device=hidden_states.device)], dim=0)[None, :]

        # block attention mask
        causal_mask = torch.triu(torch.ones(hidden_states.shape[1], hidden_states.shape[1]), diagonal=1).bool()
        attention_mask = torch.zeros_like(causal_mask).float()
        attention_mask[causal_mask==1] = float('-inf')
        for i in range(encoder_out.shape[1], hidden_states.shape[1], pred_len):
            attention_mask[i:i+pred_len, inp_len+i-encoder_out.shape[1]:i] = float('-inf')
        attention_mask = attention_mask.bfloat16().to(hidden_states.device)

        # 进行解码
        hidden_states = self.decoder(hidden_states, attention_mask=attention_mask[None, None, :, :], position_ids=position_ids)
        hidden_states = hidden_states[0]
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        # 计算loss
        labels = labels[:, -label_len+1:].view(-1).contiguous()
        logits = logits[:, -label_len:-1, :].float().view(-1, logits.size(-1)).contiguous()
        llm_logits = llm_logits[:, :-1, :].float().view(-1, logits.size(-1)).contiguous()
        kl_loss = F.kl_div(F.log_softmax(logits, dim=-1), F.softmax(llm_logits, dim=-1).detach(), reduction='batchmean')
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits, labels)

        return loss, kl_loss
    
     
    # 用于Speculative Decoding
    def generate(self, encoder_out, decoder_inp_token, max_length=10, top_k=5, threshold=0.1):
        '''
            encoder_out: LLM编码的结果，shape为(batch_size, seq_len, hidden_size)
            decoder_inp_token: 当前解码器输入的token ID，shape为(batch_size, cur_length)
            max_length: 最大生成长度
            threshold: 用于剪枝的阈值
        '''
        cur_p = torch.tensor([1.0], device=encoder_out.device).unsqueeze(-1)  # 初始概率
        all_candidates_new = []
        past_key_values = None
        for _ in range(max_length):
            # 构造hidden_states
            if past_key_values is None:
                dec_inp = torch.cat([encoder_out, self.embedding_layer(decoder_inp_token)], dim=1)
                hidden_states = self.decoder(dec_inp, use_cache=True)
                past_key_values = tuple(kv[:, :, :encoder_out.shape[1], :] for kv in hidden_states[1])
            else:
                cur_past_key_values = [kv.repeat(decoder_inp_token.shape[0], 1, 1, 1) for kv in past_key_values] # 复制soft prompt
                position_ids = torch.arange(encoder_out.shape[1], encoder_out.shape[1]+decoder_inp_token.shape[1], dtype=torch.long, device=encoder_out.device)[None, :]
                dec_inp = self.embedding_layer(decoder_inp_token)
                hidden_states = self.decoder(dec_inp, past_key_value=cur_past_key_values, position_ids=position_ids)

            # 计算logits并筛选top_k
            logits = self.lm_head(self.norm(hidden_states[0]))
            top_scores, top_indices = logits[:, -1, :].softmax(-1).topk(top_k, dim=-1)

            # 更新概率并筛选有效候选
            cur_p = cur_p * top_scores
            mask = cur_p > threshold
            if mask.sum().item() == 0: # 全部低于阈值，终止生成，并将此次生成概率最大的token拼接到后面
                decoder_inp_token = torch.cat([decoder_inp_token, top_indices[:, 0].unsqueeze(-1)], dim=1)
                all_candidates_new.append(decoder_inp_token[:, 1:])
                break
  
            decoder_inp_token = torch.cat([decoder_inp_token.unsqueeze(1).repeat(1, top_k, 1), top_indices.unsqueeze(-1)], dim=2)
            # 将概率低于阈值的序列添加到候选列表中   
            all_candidates_new.append(decoder_inp_token[~mask][:, 1:])
            # 仅保留概率高于阈值的序列
            decoder_inp_token, cur_p = decoder_inp_token[mask], cur_p[mask].unsqueeze(-1)

        seq, attention_mask, division, position_ids = Flatten_tree(all_candidates_new)
        return seq.unsqueeze(0), attention_mask, division, position_ids
