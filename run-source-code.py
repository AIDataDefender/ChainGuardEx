from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import json
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, RobertaModel, BertModel, BertTokenizer, AutoTokenizer, AutoModel)
from tqdm import tqdm, trange
import multiprocessing
from model2 import Model
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

cpu_cont = 16
logger = logging.getLogger(__name__)

from parser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript
from parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
from tree_sitter import Language, Parser
dfg_function={
    'python':DFG_python,
    'java':DFG_java,
    'ruby':DFG_ruby,
    'go':DFG_go,
    'php':DFG_php,
    'javascript':DFG_javascript
}

#load parsers
#Create a dictionary for progamming language suit with data source code chua cac bo phan tich 
parsers={}        
for lang in dfg_function:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE) 
    parser = [parser,dfg_function[lang]]    
    parsers[lang]= parser
    
    
#remove comments, tokenize code and extract dataflow           
# code la function 
# parser la chi so tu dictionary parser line 63
# lang la ngon ngu                             
def extract_dataflow(code, parser,lang):
    #remove comments 
    try:
        tree = parser[0].parse(bytes(code,'utf8'))    
        root_node = tree.root_node  
        tokens_index=tree_to_token_index(root_node)     
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index]  
        index_to_code={}
        for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
            index_to_code[index]=(idx,code)  
        try:
            DFG,_=parser[1](root_node,index_to_code,{}) 
        except:
            DFG=[]
        DFG=sorted(DFG,key=lambda x:x[1])
        indexs=set()
        for d in DFG:
            if len(d[-1])!=0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG=[]
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg=new_DFG
    except:
        dfg=[]
    #logger.info(f'code tokens, {len(code_tokens)} \n')
    #logger.info(f'dfg len, {len(dfg)}\n')
    return code_tokens,dfg


######### Dinh dang cua input cho model #########
class InputFeatures(object):
    """A single training/test features for an example."""
    def __init__(self, input_tokens, input_ids, position_idx, dfg_to_code,
                  dfg_to_dfg, label, url):# , opcode_tensor, label, url):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.position_idx = position_idx
        self.dfg_to_code = dfg_to_code
        self.dfg_to_dfg = dfg_to_dfg
        #self.bytecode_embedding = bytecode_embedding
        #self.opcode_tensor = opcode_tensor
        self.label = label
        self.url = url

def convert_examples_to_features(item):
    url, label, tokenizer, args, cache, url_to_sourcecode = item #,  opcode_tensor = item
    #url index cua sourcode 
    #cache de luu lai cac ma nguon da trich xuat tranh trich xuat trung nhau 
    #url_to_sourcecoe ma nguon tuong ung voi index
    parser = parsers['javascript']
    
    if url not in cache:
        source = url_to_sourcecode[url] #trich xuat soucecode tuong ung voi url
        code_tokens, dfg = extract_dataflow(source, parser, 'javascript') # extract ra duoc danh sach code token va dfg co shape (x,)
        #logger.info(f'Code token cua sourcode {url}: {code_tokens}\n')
        #logger.info(f'dfg cua sourcode {url}: {dfg}\n')
        code_tokens = [tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in enumerate(code_tokens)]
        #logger.info(f'code token sau khi dc tokenize {code_tokens}\n')
        ori2cur_pos = {}
        ori2cur_pos[-1] = (0, 0)
        for i in range(len(code_tokens)):
            ori2cur_pos[i] = (ori2cur_pos[i - 1][1], ori2cur_pos[i - 1][1] + len(code_tokens[i]))
        code_tokens = [y for x in code_tokens for y in x]
        
        code_tokens = code_tokens[:args.code_length + args.data_flow_length - 3][:512 - 3]
        source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
        dfg = dfg[:args.code_length + args.data_flow_length - len(source_tokens)]
        source_tokens += [x[0] for x in dfg]
        position_idx += [0 for x in dfg]
        source_ids += [tokenizer.unk_token_id for x in dfg]
        padding_length = args.code_length + args.data_flow_length - len(source_ids)
        position_idx += [tokenizer.pad_token_id] * padding_length
        source_ids += [tokenizer.pad_token_id] * padding_length
        
        reverse_index = {}
        for idx, x in enumerate(dfg):
            reverse_index[x[1]] = idx
        for idx, x in enumerate(dfg):
            dfg[idx] = x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)
        dfg_to_dfg = [x[-1] for x in dfg]
        dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
        length = len([tokenizer.cls_token])
        dfg_to_code = [(x[0] + length, x[1] + length) for x in dfg_to_code]
        cache[url] = source_tokens, source_ids, position_idx, dfg_to_code, dfg_to_dfg

    source_tokens, source_ids, position_idx, dfg_to_code, dfg_to_dfg = cache[url]
    # logger.info('source_tokens',source_tokens)
    # logger.info('source_ids',source_ids)
    # logger.info('position_idx',position_idx)
    # logger.info('dfg_to_code',dfg_to_code)
    # logger.info('dfg_to_dfg',dfg_to_dfg)
    # logger.info('label',label)
    # logger.info('url',url)
    return InputFeatures(source_tokens, source_ids, position_idx, dfg_to_code, dfg_to_dfg, label,url)#bytecode_embedding, opcode_tensor, label, url)


def preprocess_bytecode(bytecode, max_length = 512, batch_size = 32):
    tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
    modelBert = AutoModel.from_pretrained('microsoft/codebert-base')
    embedding = [] # danh sach embedding
    indices = [] # danh sach index tuong ung
    num_sample = len(bytecode)
    num_batches = (num_sample + batch_size -1) // batch_size
    logger.info(f'Number of batch size {num_batches}')

    for batch_idx in tqdm(range(num_batches)):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_sample)
        batch_bytecodes = bytecode['bytecode'].iloc[start_idx:end_idx].fillna('')
        batch_indices = bytecode['index'].iloc[start_idx:end_idx] 
        #logger.info(f'number of sample in a batch {end_idx-start_idx}')
        tokenized_texts = tokenizer(batch_bytecodes.tolist(), max_length=max_length, truncation= True, padding= 'max_length', return_tensors= 'pt')
        with torch.no_grad():
            outputs = modelBert(**tokenized_texts)
        batch_embeddings = outputs.last_hidden_state.numpy()
        #logger.info(f'embeding shape {batch_embeddings.shape}\n indices {batch_indices}\n')
        embedding.append(batch_embeddings)
        indices.append(batch_indices)

    embedding = np.concatenate(embedding,axis =0)
    indices = np.concatenate(indices,axis = 0)
    return embedding, indices

# def clean_opcode(opcode_str):
#     #opcode_str = re.sub(r'0x[a-fA-F0-9]+', '', opcode_str)
#     opcode_str = opcode_str.replace('\n', ' ')
#     #codes = re.findall(r'[A-Z]+', opcode_str)
#     return opcode_str
# def process_opcode(opcode):
#     Vocab_of_size = opcode['opcode'].nunique()
#     #print(opcode['opcode'])
#     opcode['opcode']= opcode['opcode'].str.split().apply(lambda x: ' '.join(x[:640]) if len(x) > 640 else ' '.join(x))
#     #print(opcode['opcode'])
#     length = opcode['opcode'].str.split().apply(len)
#     avg_length = int(length.mean())
#     maxlength = avg_length
#     logger.info(f'Average of length {maxlength} (must be convert to 640 for now)')
#     embedding_size = 256
#     tokenizer = Tokenizer(num_words = Vocab_of_size)
#     tokenizer.fit_on_texts(opcode['opcode']) #xay dung tu dien 
#     sequences = tokenizer.texts_to_sequences(opcode['opcode']) #chuyen sequence ve mang interger
#     opcode_matrix = pad_sequences(sequences,maxlen=maxlength)#padding ve cung 1 size 
#     opcode_tensor = torch.tensor(opcode_matrix) #chuyen ve tensor 
#     index_list = opcode['index'].tolist()
#     result_dict = {index_list[i]: opcode_tensor[i] for i in range(len(index_list))}
#     return result_dict


class ModelDataset(Dataset):
    def __init__(self, tokenizer, args, file_path='train'):
        self.examples = []
        self.args=args
        index_filename=file_path #chua danh sach index cua file tranning set valid set hoac test set
        #self.bytecode_embedding = bytecode_embeddings
        #load index
        logger.info("Creating features from index file at %s ", index_filename)
        index_to_sourcecode = {}
        #index_to_bytecode = {}
        index_to_opcode = {}
        with open('Data/Mando-test/test_source.jsonl') as f: 
            for line in f:
                line=line.strip()
                js=json.loads(line)
                index_to_sourcecode[js['idx']]=js['source'] # load sourcode theo index
                #index_to_bytecode[js['idx']] = js['byte'] # load bytecode theo index
                #index_to_opcode[js['idx']] = js['codeop'] # load opcode theo index
            
        #chuyen bytecode ve Dataframe   
        #df_bytecode = pd.DataFrame.from_dict(index_to_bytecode, orient='index', columns=['bytecode'])
       # df_bytecode.index.name = 'index'
        #df_bytecode.reset_index(inplace=True)
        
        #chuyen opcode ve Dataframe
        # df_opcode =pd.DataFrame.from_dict(index_to_opcode, orient='index', columns=['opcode'])
        # df_opcode.index.name = 'index'
        # df_opcode.reset_index(inplace=True)
        
        #opcode_matrix = process_opcode(df_opcode)
        #print('Processed opcode: ',opcode_matrix,'\n')
    
        #logger.info('loaded Data')
        #logger.info('df_bytecode :n%s', df_bytecode)

        #bytecode_embedding, bytecode_index = preprocess_bytecode(df_bytecode,max_length=20) #embedding bytecode
        #print('Processed bytecode')
        #logger.info('byte code embedding%s', bytecode_embedding.shape,'\n')
        #logger.info('bytecode index%s',bytecode_index,'\n')
        #embedding_dict = {index : embedding for index, embedding in zip(bytecode_index, bytecode_embedding)}

        #logger.info('Bytecode embeding%s', embedding_dict)
        #load code function according to index
        data_source = []
        cache={}
        f=open(index_filename)
        with open(index_filename) as f:
            for line in f:
                line=line.strip()
                url1,labels =line.split(' ', 1) # la indx cua tung function
                labels = [int(label) for label in labels.split()] # convert labels tahnh mang [1,0,1,0]
                if url1 not in index_to_sourcecode: #or url1 not in index_to_opcode:
                    continue
                #embedding_code = embedding_dict[url1]
                #opcode_tensor = opcode_matrix[url1]
                #if embedding_code is None:
                 #   logger.info('Key invalid')
                data_source.append((url1,labels,tokenizer, args,cache,index_to_sourcecode))#, opcode_tensor))
                
        #only use 10% valid data_source to keep best model        
        if 'valid' in file_path:
            data_source=random.sample(data_source,int(len(data_source)*0.1))

        
        #convert example to input features    
        self.examples=[convert_examples_to_features(x) for x in tqdm(data_source,total=len(data_source))]
        #self.examples la 1 mang Inputfeature cho tung sample
        #[<__main__.InputFeatures object at 0x73b9ea3a5810>, <__main__.InputFeatures object at 0x73b9ee9bd490>,....]

        #logger.info('seld examples %s', self.examples)


        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]): #moi line example chua cac thuoc tinh source_tokens_1,source_ids_1,position_idx_1
                logger.info("*** Example ***")
                # logger.info("idx: {}".format(idx))
                # logger.info("label: {}".format(example.label))
                # logger.info("input_tokens_1: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                # logger.info("input_ids_1: {}".format(' '.join(map(str, example.input_ids))))       
                # logger.info("position_idx_1: {}".format(example.position_idx))
                # logger.info("dfg_to_code_1: {}".format(' '.join(map(str, example.dfg_to_code))))
                # logger.info("dfg_to_dfg_1: {}".format(' '.join(map(str, example.dfg_to_dfg))))
                

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, item):
        #calculate graph-guided masked function
        attn_mask_1= np.zeros((self.args.code_length+self.args.data_flow_length,
                        self.args.code_length+self.args.data_flow_length),dtype=bool)
        #calculate begin index of node and max length of input
        node_index=sum([i>1 for i in self.examples[item].position_idx])
        max_length=sum([i!=1 for i in self.examples[item].position_idx])
        #sequence can attend to sequence
        attn_mask_1[:node_index,:node_index]=True
        #special tokens attend to all tokens
        for idx,i in enumerate(self.examples[item].input_ids):
            if i in [0,2]:
                attn_mask_1[idx,:max_length]=True
        #nodes attend to code tokens that are identified from
        for idx,(a,b) in enumerate(self.examples[item].dfg_to_code):
            if a<node_index and b<node_index:
                attn_mask_1[idx+node_index,a:b]=True
                attn_mask_1[a:b,idx+node_index]=True
        #nodes attend to adjacent nodes 
        for idx,nodes in enumerate(self.examples[item].dfg_to_dfg):
            for a in nodes:
                if a+node_index<len(self.examples[item].position_idx):
                    attn_mask_1[idx+node_index,a+node_index]=True  
                    
        out_input_ids = torch.tensor(self.examples[item].input_ids)
        out_position_ids =  torch.tensor(self.examples[item].position_idx)
        out_attn_mask =   torch.tensor(attn_mask_1)
        out_labels = torch.tensor(self.examples[item].label)
        #out_bytecode_embedding = torch.tensor(self.examples[item].bytecode_embedding) # la 1 tensor co shape la [20,768]
        #out_opcode_tensor = self.examples[item].opcode_tensor
        # logger.info('out_input_ids',out_input_ids.shape)
        # logger.info('out_postion_ids',out_position_ids.shape)
        # logger.info('out_attn_mask',out_attn_mask.shape)
        # logger.info('out_labels',out_labels.shape)
        # logger.info(f'bytecode embedding {item} : {out_bytecode_embedding.shape}')
        # logger.info('Processed source code')
        return (torch.tensor(self.examples[item].input_ids),
                torch.tensor(self.examples[item].position_idx),
                torch.tensor(attn_mask_1),    
                #out_bytecode_embedding,
                #out_opcode_tensor,         
                torch.tensor(self.examples[item].label))

def saved_extract_dataset(dataset, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(dataset,f)

def load_extract_dataset(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


import joblib

def load_dataset_from_pickle(file_path):
    dataset = joblib.load(file_path)
    return dataset

def train(args,train_dataset, model, tokenizer):
    """ Train the model """
    
    # extract_data_file = 'extract_data.pkl'

    # if os.path.exists(extract_data_file):
    #     print('Loading extracted data...')
    #     train_dataset = load_extract_dataset(extract_data_file)
    # else:
    #     print('Extracting and saving data')
    #     saved_extract_dataset(train_dataset, extract_data_file)
    #build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)
    
    args.max_steps=args.epochs*len( train_dataloader)
    args.save_steps=len( train_dataloader)//5
    print('save step', args.save_steps)
    args.warmup_steps=args.max_steps//5
    model.to(args.device)
    
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate*0.1, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        print(f'Using {args.n_gpu} gpu')

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d",args.train_batch_size*args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    
    global_step=0
    tr_loss, logging_loss,avg_loss,tr_nb,tr_num,train_loss = 0.0, 0.0,0.0,0,0,0
    best_f1=0

    model.zero_grad()
 
    for idx in range(args.epochs): 
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        tr_num=0
        train_loss=0
        for step, batch in enumerate(bar):
            (inputs_ids,position_idx,attn_mask,
            labels)=[x.to(args.device)  for x in batch]
            model.train()
            loss,logits = model(inputs_ids,position_idx,attn_mask, labels)

            if args.n_gpu > 1:
                loss = loss.mean()
                
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num+=1
            train_loss+=loss.item()

            if avg_loss==0:
              avg_loss=tr_loss
                
            avg_loss=round(train_loss/tr_num,5)
            bar.set_description("epoch {} loss {}".format(idx,avg_loss))
              
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                output_flag=True
                avg_loss=round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4) 

                if global_step % args.save_steps == 0:
                    results = evaluate(args, model, tokenizer, eval_when_training=True)    
                    
                    # Save model checkpoint
                    if results['eval_f1']>best_f1:
                        best_f1=results['eval_f1']
                        logger.info("  "+"*"*20)  
                        logger.info("  Best f1:%s",round(best_f1,4))
                        logger.info("  "+"*"*20)                          
                        
                        checkpoint_prefix = 'checkpoint-best-f1'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)                        
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format('model2.bin')) 
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)
                        logger.info("Saved model")
                        
def evaluate(args, model, tokenizer, eval_when_training=False):
    #build dataloader
    eval_dataset = ModelDataset(tokenizer, args, file_path=args.eval_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,batch_size=args.eval_batch_size,num_workers=4)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]  
    y_trues=[]
    all_labels = []
    all_preds = []
    for batch in tqdm(eval_dataloader,desc="Evaluating"):
        (inputs_ids,position_idx,attn_mask,
        labels)=[x.to(args.device)  for x in batch]
        with torch.no_grad():
            lm_loss,logit = model(inputs_ids,position_idx,attn_mask, labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1
    
    #calculate scores
    logits=np.concatenate(logits,axis = 0)
    y_trues=np.concatenate(y_trues,axis = 0)
    best_threshold=0.5
    best_f1=0

    #y_preds=logits[:,1]>best_threshold
    y_preds = logits
    #print('prediction',y_preds)
    #print('label truth',y_trues)
    recall=recall_score(y_trues, y_preds, average='weighted')
    precision=precision_score(y_trues, y_preds,  average='weighted')   
    f1=f1_score(y_trues, y_preds, average='weighted')      
    accuracy = accuracy_score(y_trues, y_preds)
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_accuracy": float(accuracy),
        "eval_threshold":best_threshold
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],4)))

    return result

def test(args, model, tokenizer, best_threshold=0):
    #build dataloader
    eval_dataset = ModelDataset(tokenizer, args, file_path=args.test_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=4)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    #Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]  
    y_trues=[]
    for batch in tqdm(eval_dataloader,desc="Testing:"):
        (inputs_ids,position_idx,attn_mask,
        labels)=[x.to(args.device)  for x in batch]
        with torch.no_grad():
            lm_loss,logit = model(inputs_ids,position_idx,attn_mask, labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1
    
    #output result
    logits=np.concatenate(logits, axis = 0)
    y_trues=np.concatenate(y_trues,axis = 0)
    y_preds=logits
    
    weighted_recall=recall_score(y_trues, y_preds, average='weighted')
    weighted_precision=precision_score(y_trues, y_preds,  average='weighted')   
    weighted_f1=f1_score(y_trues, y_preds, average='weighted')   

    micro_recall=recall_score(y_trues, y_preds, average='micro')
    micro_precision=precision_score(y_trues, y_preds,  average='micro')   
    micro_f1=f1_score(y_trues, y_preds, average='micro') 
    
    macro_recall=recall_score(y_trues, y_preds, average='macro')
    macro_precision=precision_score(y_trues, y_preds,  average='macro')   
    macro_f1=f1_score(y_trues, y_preds, average='macro') 
    accuracy = accuracy_score(y_trues, y_preds)
    result = {
        "Test_micro_recall": float(micro_recall),
        "Test_micro_precision": float(micro_precision),
        "Test_micro_f1": float(micro_f1),
        "Test_macro_recall": float(macro_recall),
        "Test_macro_precision": float(macro_precision),
        "Test_macro_f1": float(macro_f1),
        "Test_weighted_recall": float(weighted_recall),
        "Test_weighted_precision": float(weighted_precision),
        "Test_weighted_f1": float(weighted_f1),
        "Test_accuracy": float(accuracy)
    }

    logger.info("***** Test Result *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],4)))

    return result
                                                
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
                    
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    parser.add_argument("--data_flow_length", default=64, type=int,
                        help="Optional Data Flow input sequence length after tokenization.") 
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")    
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--train_batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=1,
                        help="training epochs")

    args = parser.parse_args()

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0

    args.device = device
    # Setup CPU
    # device = torch.device("cpu")
    # args.n_gpu = 0  # No GPUs are being used

    # args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s",device, args.n_gpu,)


    # Set seed
    set_seed(args)
    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    config.num_labels=10
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, clean_up_tokenization_spaces = False)# su dung tokenizer tu pretrainmodel
    #model = RobertaModel.from_pretrained(args.model_name_or_path, config=config)  # Dùng encoder từ mô hình pre-trained
    model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path,config=config) #encoder ma hoa model ma hoa
    #return loss and logits co shape la 1 tensor
    from torchvision import models
    model=Model(model,config,tokenizer,args)

    from torchsummary import summary
    #summary(model,input_size= (20))

    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        train_dataset = ModelDataset(tokenizer, args, file_path=args.train_data_file)
        train(args, train_dataset, model, tokenizer)

    # Evaluation
    results = {}
    if args.do_eval:
        checkpoint_prefix = 'checkpoint-best-f1/model2.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir, weights_only = True))
        model.to(args.device)
        result=evaluate(args, model, tokenizer)
        
    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-f1/model2.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir, weights_only = True))
        model.to(args.device)
        test(args, model, tokenizer,best_threshold=0.5)

    return results


if __name__ == "__main__":
    main()

