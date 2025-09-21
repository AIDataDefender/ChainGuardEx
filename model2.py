import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
import transformers



class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 10) # 10 classes

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])cl
        #x = x.reshape(-1,x.size(-1)*2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Model(nn.Module):   
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args
        self.opcode_dense = nn.Linear(1,768)
        self.attention = nn.MultiheadAttention(embed_dim=768, num_heads=8)
    
    @staticmethod
    def expand_tensor_with_padding(tensor, new_size):
        batch_size, old_size, _ = tensor.shape
        pad_size = new_size - old_size
        padded_tensor = F.pad(tensor, (0, pad_size, 0, pad_size), "constant", 0)
        return padded_tensor

    @staticmethod
    def expand_tensor_with_positionidx(tensor, new_size):
        batch_size, old_size = tensor.shape
        pad_size = new_size - old_size
        padded_tensor = F.pad(tensor, (0, pad_size), "constant", 0)
        return padded_tensor

    def forward(self, inputs_ids, position_idx, attn_mask, labels=None):# bytecode_embedding, opcode_tensor, labels=None):
        bs, l = inputs_ids.size()
        
        # Embedding
        nodes_mask = position_idx.eq(0)
        token_mask = position_idx.ge(2)
        inputs_embeddings = self.encoder.roberta.embeddings.word_embeddings(inputs_ids)


        nodes_to_token_mask = nodes_mask[:, :, None] & token_mask[:, None, :] & attn_mask
        nodes_to_token_mask = nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:, :, None]
        avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
        inputs_embeddings = inputs_embeddings * (~nodes_mask)[:, :, None] + avg_embeddings * nodes_mask[:, :, None]

        # print('Shape of inputs_embedding', inputs_embeddings.shape)
        # print('Shape of input_ids', inputs_ids.shape)
        # print('Shape of input_ids', position_idx.shape)
        # print('Shape of input_ids', attn_mask.shape)
        # print('Shape of bytecode embedding', bytecode_embedding.shape)
        # print('Shape of opcode', opcode_tensor.shape)
        # print('opcode receivrd from extract', opcode_tensor)
        # print('bytecode recevied from extract', bytecode_embedding)
        # print('input embediign ', inputs_embeddings)
        # print('Position_idx', position_idx)
        # print('attn_mask',attn_mask)

        #inputs_embeddings[:, :opcode_tensor.size(1), :opcode_tensor.size(1)] += opcode_tensor.unsqueeze(2)
        #opcode_tensor =opcode_tensor.float()
        #opcode_transformed = self.opcode_dense(opcode_tensor)

        #opcode_transformed = opcode_tensor.unsqueeze(-1)  # [batch_size, 512, 1]
        #opcode_transformed = self.opcode_dense(opcode_transformed)  # [batch_size, 512, 768]
        #print('Shape of opcode after transform', opcode_transformed.shape,'\n')

        #print('Shape of Inputs_embedidng', inputs_embeddings.shape)
        #Shape of inputs embeeing torch.Size([8, 640, 768])0700
        #inputs_embeddings = torch.cat([inputs_embeddings,bytecode_embedding,opcode_transformed],dim=1)
        #inputs_embeddings = torch.cat([inputs_embeddings,bytecode_embedding],dim=1)
        #print('Shape final', inputs_embeddings.shape)
        #shape is [batch_size, 860,768]

        attn_mask = self.expand_tensor_with_padding(attn_mask,inputs_embeddings.size(dim=1))#1110)
        position_idx = self.expand_tensor_with_positionidx(position_idx,inputs_embeddings.size(dim=1))#1110)
        #print('Shape of attn_mask',attn_mask.shape)
        #print('Shape of position', position_idx.shape)
        # Encode with RoBERTa
        #print('Shape of inputs_embedding after plus', inputs_embeddings.shape)
        #print('Input_embediing', inputs_embeddings)
        

        outputs = self.encoder.roberta(
            inputs_embeds=inputs_embeddings,
            attention_mask=attn_mask,
            position_ids=position_idx,
            token_type_ids=position_idx.eq(-1).long()
            )[0]
        
        #print(f"outputs shape before classifier: {outputs.shape}")  # Add this

        logits = self.classifier(outputs)
        prob = torch.sigmoid(logits)

        prediction = (prob > 0.5).int()

        pred_str = " ".join(map(str, prediction.tolist()[0]))
        
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
            return loss, prediction
        else:
            return prob

      
        

       
