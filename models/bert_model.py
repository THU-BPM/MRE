import torch
import os
from torch import nn
import torch.nn.functional as F
from torchcrf import CRF
from .modeling_bert import BertModel
from transformers.modeling_outputs import TokenClassifierOutput
from torchvision.models import resnet50
class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        self.resnet = resnet50(pretrained=True)
    
    def forward(self, x, aux_imgs=None):
        # full image prompt
        prompt_guids = self.get_resnet_prompt(x)    # 4x[bsz, 256, 7, 7]
        
        # aux_imgs: bsz x 3(nums) x 3 x 224 x 224
        if aux_imgs is not None:
            aux_prompt_guids = []   # goal: 3 x (4 x [bsz, 256, 7, 7])
            aux_imgs = aux_imgs.permute([1, 0, 2, 3, 4])  # 3(nums) x bsz x 3 x 224 x 224
            for i in range(len(aux_imgs)):
                aux_prompt_guid = self.get_resnet_prompt(aux_imgs[i]) # 4 x [bsz, 256, 7, 7]
                aux_prompt_guids.append(aux_prompt_guid)   
            return prompt_guids, aux_prompt_guids
        return prompt_guids, None

    def get_resnet_prompt(self, x):
        """generate image prompt

        Args:
            x ([torch.tenspr]): bsz x 3 x 224 x 224

        Returns:
            prompt_guids ([List[torch.tensor]]): 4 x List[bsz x 256 x 7 x 7]
        """
        # image: bsz x 3 x 224 x 224
        prompt_guids = []
        for name, layer in self.resnet.named_children():
            if name == 'fc' or name == 'avgpool':  continue
            x = layer(x)    # (bsz, 256, 56, 56)
            if 'layer' in name:
                bsz, channel, ft, _ = x.size()
                kernel = ft // 2
                prompt_kv = nn.AvgPool2d(kernel_size=(kernel, kernel), stride=kernel)(x)    # (bsz, 256, 7, 7)
                prompt_guids.append(prompt_kv)   # conv2: (bsz, 256, 7, 7)
        return prompt_guids


class HMNeTREModel(nn.Module):
    def __init__(self, num_labels, tokenizer, args):
        super(HMNeTREModel, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_name)
        self.bert.resize_token_embeddings(len(tokenizer))
        self.args = args

        self.dropout = nn.Dropout(0.5)
        #self.classifier = nn.Linear(self.bert.config.hidden_size*2, num_labels)
        self.input_dim = 2
        if args.entity or args.entity_sent or self.args.use_entity_cos:
            self.input_dim += 1
        if args.caption or args.caption_sent:
            self.input_dim += 1
        self.classifier = nn.Linear(self.bert.config.hidden_size*self.input_dim, num_labels)
        self.head_start = tokenizer.convert_tokens_to_ids("<s>")
        self.tail_start = tokenizer.convert_tokens_to_ids("<o>")
        self.tokenizer = tokenizer
        self.attention_query_caption = nn.MultiheadAttention(768,12,batch_first=True)
        self.attention_query_entity = nn.MultiheadAttention(768,12,batch_first=True)
        self.cos = nn.CosineSimilarity(dim = 1)
        if self.args.use_prompt:
            self.image_model = ImageModel()

            self.encoder_conv =  nn.Sequential(
                                    nn.Linear(in_features=3840, out_features=800),
                                    nn.Tanh(),
                                    nn.Linear(in_features=800, out_features=4*2*768)
                                )

            self.gates = nn.ModuleList([nn.Linear(4*768*2, 4) for i in range(12)])
    def bert_tokenizer_get_maxlen(self,array_of_arrays):
        lengths_arr = []
        for evidence_set in array_of_arrays:
            for one_caption in evidence_set:
                tokenize_out = self.tokenizer.tokenize(one_caption)
                lengths_arr.append(len(tokenize_out))
        if len(lengths_arr)==0: return 2 
        max_len = max(lengths_arr)+2  #2 special tokens 
        return max_len
    def bert_tokenizer(self,text, max_len,prompt_guids = None):
        #text is an array of sentences 
        input_ids = []
        attention_masks = []
        type_ids = []
        for sent in text:
            tokenize_out = self.tokenizer.encode_plus(sent,add_special_tokens = True, truncation = True, padding = 'max_length', max_length = max_len, return_tensors = 'pt')
            input_ids.append(tokenize_out['input_ids'])
            if self.args.use_prompt:
                attention_masks.append(torch.cat((prompt_guids,tokenize_out['attention_mask'].squeeze().cuda())).unsqueeze(0))
            else:
                attention_masks.append(tokenize_out['attention_mask'])
            type_ids.append(tokenize_out['token_type_ids'])
        input_ids = torch.cat(input_ids, dim=0).cuda()
        type_ids = torch.cat(type_ids, dim=0).cuda()
        attention_masks = torch.cat(attention_masks, dim=0).cuda()
        return input_ids, type_ids, attention_masks 


    def evidence_BertEmbs_batch(self,array_of_arrays,prompt_mask = None,prompt_guids = None):
        #input is an array of arrays. each sub-array contains the evidence for that example.   
        max_tokens_len = self.bert_tokenizer_get_maxlen(array_of_arrays)
        evidence_embs_batch = []
        for i,evidence_set in enumerate(array_of_arrays):
            if prompt_mask is not None:
                input_ids, type_ids, attention_masks = self.bert_tokenizer(evidence_set, max_tokens_len,prompt_mask[i,:])
            else:
                input_ids, type_ids, attention_masks = self.bert_tokenizer(evidence_set, max_tokens_len)

            if self.args.use_prompt:
                guids = prompt_guids.copy()
                for j in range(12):
                    guids[j] = (guids[j][0][i,:,:,:].unsqueeze(0).repeat(11,1,1,1),guids[j][1][i,:,:,:].unsqueeze(0).repeat(11,1,1,1))
            else :
                guids = None
            outputs = self.bert(input_ids=input_ids, token_type_ids=type_ids, attention_mask=attention_masks, output_hidden_states=True,past_key_values=guids, #传入bert
                    output_attentions=True,
                    return_dict=True)
            one_evidence_embs = outputs.last_hidden_state
            evidence_embs_batch.append(one_evidence_embs)
        if len(evidence_embs_batch) == 0:
            print('Empty')
            print(array_of_arrays)
        evidence_embs_batch = torch.stack(evidence_embs_batch, dim=0)
        evidence_embs_batch = torch.sum(evidence_embs_batch,dim = 1)
        evidence_embs_batch = evidence_embs_batch / 11
        return evidence_embs_batch 
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        cap_input_ids = None,
        cap_attention_mask = None,
        cap_token_type_ids = None,
        ent_input_ids = None,
        ent_attention_mask = None,
        ent_token_type_ids = None,
        entities = None,
        labels=None,
        images=None,
        aux_imgs=None,
    ):

        
        bsz = input_ids.size(0)
        entities = [batch.split('|') for batch in entities]
        if self.args.use_prompt:
            prompt_guids = self.get_visual_prompt(images, aux_imgs) #获得视觉部分的K V
            prompt_guids_length = prompt_guids[0][0].shape[2]

            prompt_guids_mask = torch.ones((bsz, prompt_guids_length)).to(self.args.device)
            prompt_attention_mask = torch.cat((prompt_guids_mask, attention_mask), dim=1)

            cap_prompt_attention_mask = torch.cat((prompt_guids_mask, cap_attention_mask),dim=1)
            ent_prompt_attention_mask = torch.cat((prompt_guids_mask, ent_attention_mask),dim=1)
        else:
            prompt_guids = None
            prompt_attention_mask = attention_mask
            cap_prompt_attention_mask = cap_attention_mask
            ent_prompt_attention_mask = ent_attention_mask
            prompt_guids_mask = None
            prompt_guids = None


        output = self.bert(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=prompt_attention_mask, 
                    past_key_values=prompt_guids, #传入bert
                    output_attentions=True,
                    return_dict=True
        )
        
            
        last_hidden_state, pooler_output = output.last_hidden_state, output.pooler_output
        bsz, seq_len, hidden_size = last_hidden_state.shape
        entity_hidden_state = torch.Tensor(bsz, hidden_size*self.input_dim) # batch, 2*hidden#
        cap_embedding = torch.Tensor(bsz,hidden_size)
        if self.args.entity:
            entities_emb = self.evidence_BertEmbs_batch(entities,prompt_guids_mask,prompt_guids)
            if self.args.use_attention:
                entities_emb , _= self.attention_query_entity(last_hidden_state,entities_emb,entities_emb)
        # for i in range(bsz):
        #     for j in cap_input_ids[i]:
        #         print(j)
            # for j in range(cap_input_ids.shape(1)):
            #     output = self.bert(
            #             input_ids=cap_input_ids[i:j:].squeeze(1),
            #             token_type_ids=cap_token_type_ids[i:j:].squeeze(1),
            #             attention_mask=cap_prompt_attention_mask,
            #             past_key_values=prompt_guids,
            #             output_attentions=True,
            #             return_dict=True
            #     )
            #     last_hidden_state, pooler_output = output.last_hidden_state, output.pooler_output
            #     bsz, seq_len, hidden_size = last_hidden_state.shape

        if self.args.caption:
            cap_out = self.bert(
                        input_ids=cap_input_ids,
                        token_type_ids=cap_token_type_ids,
                        attention_mask=cap_prompt_attention_mask,
                        past_key_values=prompt_guids,
                        output_attentions=True,
                        return_dict=True
            )
            cap_last_hidden_state = cap_out.last_hidden_state
            if self.args.use_attention:
                cap_last_hidden_state,_ = self.attention_query_caption(last_hidden_state,cap_last_hidden_state,cap_last_hidden_state)
        if self.args.entity_sent or self.args.use_entity_cos:
            ent_out = self.bert(
                        input_ids=ent_input_ids,
                        token_type_ids=ent_token_type_ids,
                        attention_mask=ent_prompt_attention_mask,
                        past_key_values=prompt_guids,
                        output_attentions=True,
                        return_dict=True
            ) 
            ent_last_hidden_state = ent_out.last_hidden_state
            if self.args.use_attention:
                ent_last_hidden_state,_ = self.attention_query_entity(last_hidden_state,ent_last_hidden_state,ent_last_hidden_state)
        for i in range(bsz):
            head_idx = input_ids[i].eq(self.head_start).nonzero().item()
            tail_idx = input_ids[i].eq(self.tail_start).nonzero().item()
            head_hidden = last_hidden_state[i, head_idx, :].squeeze()
            tail_hidden = last_hidden_state[i, tail_idx, :].squeeze()
            if self.args.caption:
                cap_hidden = cap_last_hidden_state[i,0,:].squeeze()
            if self.args.entity:
                en_hidden = entities_emb[i,0,:].squeeze()
            if self.args.entity_sent or self.args.use_entity_cos:
                ent_hidden = ent_last_hidden_state[i,0,:].squeeze()
            #entity_hidden_state[i] = torch.cat([head_hidden, tail_hidden], dim=-1)
            embedding_list = [head_hidden, tail_hidden]
            if self.args.caption: 
                embedding_list.append(cap_hidden)
            if self.args.entity:
                embedding_list.append(en_hidden)
            if self.args.use_entity_cos:
                cos_distance = self.cos(ent_hidden.unsqueeze(dim = 0),last_hidden_state[i,:,:] )
                cos_distance = cos_distance[attention_mask[i,:] == 1]
                cos_distance = cos_distance[1:]
                idx = torch.argmax(cos_distance,dim = 0) + 1
                print(f'shape is {cos_distance.shape},idx is {idx}')
                embedding_list.append(last_hidden_state[i,idx,:])
            if self.args.entity_sent:
                embedding_list.append(ent_hidden)
            entity_hidden_state[i] = torch.cat(embedding_list, dim=-1)
            #entity_hidden_state[i] = torch.cat([head_hidden, tail_hidden,cap_hidden], dim=-1)
            #entity_hidden_state[i] = 1/2*(head_hidden+tail_hidden)#+cap_hidden)
        entity_hidden_state = entity_hidden_state.to(self.args.device)
        logits = self.classifier(entity_hidden_state)
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            return loss_fn(logits, labels.view(-1)), logits
        return logits

    def get_visual_prompt(self, images, aux_imgs):
        '''
            input : images:图片 (1) aux_imgs:object (3)
        '''
        bsz = images.size(0)
        # full image prompt   得到 resnet输出 (每张图片4个embedding)
        prompt_guids, aux_prompt_guids = self.image_model(images, aux_imgs)  # [bsz, 256, 2, 2], [bsz, 512, 2, 2]....
        #将图片输出转化为(bsz,4,3080),将每张图片的4个embedding融合
        prompt_guids = torch.cat(prompt_guids, dim=1).view(bsz, self.args.prompt_len, -1)   # bsz, 4, 3840
        #同样修改object输出
        # aux image prompts # 3 x (4 x [bsz, 256, 2, 2])
        aux_prompt_guids = [torch.cat(aux_prompt_guid, dim=1).view(bsz, self.args.prompt_len, -1) for aux_prompt_guid in aux_prompt_guids]  # 3 x [bsz, 4, 3840]
        #将图片embedding与object embedding 放入多层感知机计算权重
        '''
            encoder_conv =  nn.Sequential(
                                    nn.Linear(in_features=3840, out_features=800),
                                    nn.Tanh(),
                                    nn.Linear(in_features=800, out_features=4*2*768)
                                )
        '''
        prompt_guids = self.encoder_conv(prompt_guids)  # bsz, 4, 4*2*768
        aux_prompt_guids = [self.encoder_conv(aux_prompt_guid) for aux_prompt_guid in aux_prompt_guids] # 3 x [bsz, 4, 4*2*768]
        #将图片embedding 与 object embedding 重新切分 (上述过程将embedding融合),每张图片4个embedding
        split_prompt_guids = prompt_guids.split(768*2, dim=-1)   # 4 x [bsz, 4, 768*2]
        split_aux_prompt_guids = [aux_prompt_guid.split(768*2, dim=-1) for aux_prompt_guid in aux_prompt_guids]   # 3x [4 x [bsz, 4*768*2]]
        #对每张图片的四个embedding取平均
        sum_prompt_guids = torch.stack(split_prompt_guids).sum(0).view(bsz, -1) / 4     # bsz, 4 * 768*2
        
        result = []
        for idx in range(12):  # 12
            '''
            gates = nn.ModuleList([nn.Linear(4*768*2, 4) for i in range(12)])
            '''
            prompt_gate = F.softmax(F.leaky_relu(self.gates[idx](sum_prompt_guids)), dim=-1) # bsz , 4
            #生成 K V键值对
            key_val = torch.zeros_like(split_prompt_guids[0]).to(self.args.device)  # bsz, 4, 768*2
            #将图片信息加入键值对
            for i in range(4):
                key_val = key_val + torch.einsum('bg,blh->blh', prompt_gate[:, i].view(-1, 1), split_prompt_guids[i])
            # use gate mix aux image prompts
            aux_key_vals = []   # 3 x [bsz, 4, 768*2]
            #将object信息加入键值对
            for split_aux_prompt_guid in split_aux_prompt_guids:
                sum_aux_prompt_guids = torch.stack(split_aux_prompt_guid).sum(0).view(bsz, -1) / 4     # bsz, 4, 768*2
                aux_prompt_gate = F.softmax(F.leaky_relu(self.gates[idx](sum_aux_prompt_guids)), dim=-1)
                aux_key_val = torch.zeros_like(split_aux_prompt_guid[0]).to(self.args.device)  # bsz, 4, 768*2
                for i in range(4):
                    aux_key_val = aux_key_val + torch.einsum('bg,blh->blh', aux_prompt_gate[:, i].view(-1, 1), split_aux_prompt_guid[i])
                aux_key_vals.append(aux_key_val)
            key_val = [key_val] + aux_key_vals
            key_val = torch.cat(key_val, dim=1)
            key_val = key_val.split(768, dim=-1)
            #得到(K,V)键值对，(batch_size, num_heads, sequence_length - 1, embed_size_per_head)(bsz 12 4 64)
            key, value = key_val[0].reshape(bsz, 12, -1, 64).contiguous(), key_val[1].reshape(bsz, 12, -1, 64).contiguous()  # bsz, 12, 4, 64
            temp_dict = (key, value)
            result.append(temp_dict)
        return result


class HMNeTNERModel(nn.Module):
    def __init__(self, label_list, args):
        super(HMNeTNERModel, self).__init__()
        self.args = args
        self.prompt_dim = args.prompt_dim
        self.prompt_len = args.prompt_len
        self.bert = BertModel.from_pretrained(args.bert_name)
        self.bert_config = self.bert.config

        if args.use_prompt:
            self.image_model = ImageModel()  # bsz, 6, 56, 56
            self.encoder_conv =  nn.Sequential(
                            nn.Linear(in_features=3840, out_features=800),
                            nn.Tanh(),
                            nn.Linear(in_features=800, out_features=4*2*768)
                            )
            self.gates = nn.ModuleList([nn.Linear(4*768*2, 4) for i in range(12)])

        self.num_labels  = len(label_list)  # pad
        print(self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)
        self.fc = nn.Linear(self.bert.config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, images=None, aux_imgs=None):
        if self.args.use_prompt:
            prompt_guids = self.get_visual_prompt(images, aux_imgs)
            prompt_guids_length = prompt_guids[0][0].shape[2]
            # attention_mask: bsz, seq_len
            # prompt attention， attention mask0
            bsz = attention_mask.size(0)
            prompt_guids_mask = torch.ones((bsz, prompt_guids_length)).to(self.args.device)
            prompt_attention_mask = torch.cat((prompt_guids_mask, attention_mask), dim=1)
        else:
            prompt_attention_mask = attention_mask
            prompt_guids = None

        bert_output = self.bert(input_ids=input_ids,
                            attention_mask=prompt_attention_mask,
                            token_type_ids=token_type_ids,
                            past_key_values=prompt_guids,
                            return_dict=True)
        sequence_output = bert_output['last_hidden_state']  # bsz, len, hidden
        sequence_output = self.dropout(sequence_output)  # bsz, len, hidden
        emissions = self.fc(sequence_output)    # bsz, len, labels
        
        logits = self.crf.decode(emissions, attention_mask.byte())
        loss = None
        if labels is not None:
            loss = -1 * self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean') 
        return TokenClassifierOutput(
            loss=loss,
            logits=logits
        )

    def get_visual_prompt(self, images, aux_imgs):
        bsz = images.size(0)
        prompt_guids, aux_prompt_guids = self.image_model(images, aux_imgs)  # [bsz, 256, 2, 2], [bsz, 512, 2, 2]....

        prompt_guids = torch.cat(prompt_guids, dim=1).view(bsz, self.args.prompt_len, -1)   # bsz, 4, 3840
        aux_prompt_guids = [torch.cat(aux_prompt_guid, dim=1).view(bsz, self.args.prompt_len, -1) for aux_prompt_guid in aux_prompt_guids]  # 3 x [bsz, 4, 3840]

        prompt_guids = self.encoder_conv(prompt_guids)  # bsz, 4, 4*2*768
        aux_prompt_guids = [self.encoder_conv(aux_prompt_guid) for aux_prompt_guid in aux_prompt_guids] # 3 x [bsz, 4, 4*2*768]
        split_prompt_guids = prompt_guids.split(768*2, dim=-1)   # 4 x [bsz, 4, 768*2]
        split_aux_prompt_guids = [aux_prompt_guid.split(768*2, dim=-1) for aux_prompt_guid in aux_prompt_guids]   # 3x [4 x [bsz, 4, 768*2]]

        result = []
        for idx in range(12):  # 12
            sum_prompt_guids = torch.stack(split_prompt_guids).sum(0).view(bsz, -1) / 4     # bsz, 4, 768*2
            prompt_gate = F.softmax(F.leaky_relu(self.gates[idx](sum_prompt_guids)), dim=-1)

            key_val = torch.zeros_like(split_prompt_guids[0]).to(self.args.device)  # bsz, 4, 768*2
            for i in range(4):
                key_val = key_val + torch.einsum('bg,blh->blh', prompt_gate[:, i].view(-1, 1), split_prompt_guids[i])

            aux_key_vals = []   # 3 x [bsz, 4, 768*2]
            for split_aux_prompt_guid in split_aux_prompt_guids:
                sum_aux_prompt_guids = torch.stack(split_aux_prompt_guid).sum(0).view(bsz, -1) / 4     # bsz, 4, 768*2
                aux_prompt_gate = F.softmax(F.leaky_relu(self.gates[idx](sum_aux_prompt_guids)), dim=-1)
                aux_key_val = torch.zeros_like(split_aux_prompt_guid[0]).to(self.args.device)  # bsz, 4, 768*2
                for i in range(4):
                    aux_key_val = aux_key_val + torch.einsum('bg,blh->blh', aux_prompt_gate[:, i].view(-1, 1), split_aux_prompt_guid[i])
                aux_key_vals.append(aux_key_val)
            key_val = [key_val] + aux_key_vals
            key_val = torch.cat(key_val, dim=1)
            key_val = key_val.split(768, dim=-1)
            key, value = key_val[0].reshape(bsz, 12, -1, 64).contiguous(), key_val[1].reshape(bsz, 12, -1, 64).contiguous()  # bsz, 12, 4, 64
            temp_dict = (key, value)
            result.append(temp_dict)
        return result
