import json
import logging
import os
import traceback
import pandas as pd
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import re
from tqdm import tqdm
import warnings

from transformers import AutoModel, AutoTokenizer
try:
    from experiments.utils.graph_utils import vuln_to_label, OWASP_VULN
    from experiments.utils.logger import setup_logger
except ImportError:
    from utils.graph_utils import vuln_to_label, OWASP_VULN
    from utils.logger import setup_logger

os.makedirs("Logs", exist_ok=True)
logger = setup_logger("Logs/raw_code_processing.log")


class RawCodeFeatureExtractor:
    """
    Object-oriented processor for raw code data using CodeBERT embeddings.
    """
    
    def __init__(self, tokenizer, model, device=None , batch_size = 4, is_check_dim=False):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = 512  # Maximum sequence length input for CodeBERT
        self.emb_out_dim = 768 # CodeBERT embedding output dimension
        self.batch_size = batch_size  # Default batch size for encoding (reduced to 4 to avoid CUDA memory issues)
        #################################
        self.is_check_dim = is_check_dim  # Boolean to activate token dimension checking
        self.stats ={
            'token_max': 0,    # Maximum tokens in a sample (no padding) - GLOBAL across all batches
            'token_min': float('inf'),  # Minimum tokens in a sample (no padding) - GLOBAL across all batches
            'token_avg': 0.0,   # Average tokens across samples (no padding) - GLOBAL across all batches
            'max_len_reached': 0,  # Count of sequences that hit max_length (truncated) - GLOBAL
            'total_samples': 0,  # Total number of samples processed - GLOBAL
        }
        self.global_token_counts = []  # Track all token counts across all batches/dataframes
        # Local batch stats (reset per batch)
        self.batch_stats = {
            'max_len_reached': 0,  # Count of truncated sequences in current batch
            'total_samples': 0,  # Total samples in current batch
        }
        #################################
        self.dtype = torch.float32  # Default data type for tensors
        self.model.eval()  # Set model to evaluation mode
        self.model.to(self.device)
        # Use the tokenizer's own separator token; don't force-add a new token that the model can't embed
        self.sep_token = getattr(self.tokenizer, 'sep_token', None) or '</s>'
        
    def fetch_all_feat_dim(self):
        return {
            # Code input dimension (CodeBERT embeddings)
            "CODE_IN_DIM": self.emb_out_dim,  # Fixed for CodeBERT
            "OUTPUT_DIM": len(OWASP_VULN),  # OWASP
        }
    
    def preprocess_code(self, code: str) -> str:
        """
        Preprocess raw code by cleaning and normalizing it.
        
        Args:
            code (str): Raw code string
            
        Returns:
            str: Preprocessed code
        """
        
        # Convert to string if not already
        code = str(code)
        
        # More robust regex to remove C-style comments (single-line and multi-line)
        # This will remove // ... and /* ... */, which may contain leaked labels.
        # re.DOTALL makes . match newlines, for multi-line /* ... */
        # re.MULTILINE makes $ match end-of-line, for // ...
        code = code = re.sub(r'//.*?$|/\*.*?\*/|/\*.*|\*/.*?(?=\n|$)', '', code, flags=re.DOTALL | re.MULTILINE)
        # Remove excessive whitespace (now do this *after* comment removal)
        code = re.sub(r'\s+', ' ', code)
        
        # Normalize common patterns
        code = re.sub(r'\b0x[a-fA-F0-9]+\b', '<ADDRESS>', code)  # Replace addresses
        code = re.sub(r'\b\d+\b', '<NUMBER>', code)  # Replace numbers
        
        # Strip and clean
        code = code.strip()
        logger.debug(f"\n==============CLEANED CODE============\n{code}\n==========================")  
        return code
    
    def create_combined_features(self, row: pd.Series) -> str:
        """
        Create combined feature string from row data.
        
        Args:
            row (pd.Series): Row from the dataframe
            
        Returns:
            str: Combined feature string
        """
        features = []
        #DataFrame columns: ['filename', 'contract', 'func', 'func_raw_code', 'vuln_id', 'vuln_desc', 'vuln_line_code', 'item_type']
        # Add metadata features
        if not pd.isna(row['contract']):
            features.append(f"CONTRACT: {row['contract']}")
        
        if not pd.isna(row['func']):
            features.append(f"FUNCTION: {row['func']}")
        
        # Add main code
        if not pd.isna(row['func_raw_code']):
            clean_code = self.preprocess_code(row['func_raw_code'])
            if clean_code:
                features.append(f"CODE: {clean_code}")
        
        return f" {self.sep_token} ".join(features)
    
    def encode_text_batch(self, texts: List[str]) -> torch.Tensor:
        all_embeddings = []
        
        # Reset batch stats for this encoding pass
        self.batch_stats['max_len_reached'] = 0
        self.batch_stats['total_samples'] = 0
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            try:
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",

                ).to(self.device)
                
                # Track token dimensions if enabled
                if self.is_check_dim:
                    input_ids = inputs['input_ids']
                    for seq in input_ids:
                        # Count actual tokens (excluding padding)
                        token_count = (seq != self.tokenizer.pad_token_id).sum().item()
                        self.global_token_counts.append(token_count)
                        self.stats['token_max'] = max(self.stats['token_max'], token_count)
                        self.stats['token_min'] = min(self.stats['token_min'], token_count)
                        
                        # Increment global sample count
                        self.stats['total_samples'] += 1
                        self.batch_stats['total_samples'] += 1
                        
                        # Check if truncated (sequence length == max_length and last token is not pad)
                        is_truncated = (len(seq) == self.max_length and 
                                        seq[-1] != self.tokenizer.pad_token_id)
                        if is_truncated:
                            self.stats['max_len_reached'] += 1
                            self.batch_stats['max_len_reached'] += 1

                with torch.no_grad():
                    # Set model to eval mode to ensure consistent behavior
                    self.model.eval()
                    # Try passing only essential inputs if token_type_ids causes issues
                    try:
                        outputs = self.model(**inputs)
                    except RuntimeError as inner_e:
                        logger.error(f"RuntimeError during model forward pass (batch start {i}): {inner_e}")
                        # Skip this batch to avoid using undefined outputs
                        continue
                    
                    # Use CLS token embeddings (first token)
                    embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu()  # Shape: (batch_size, hidden_size)
                    all_embeddings.append(embeddings)
            
            except RuntimeError as e:
                logger.error(f"RuntimeError during encoding batch starting at index {i}: {e}")
                logger.error(batch_texts)
                logger.error(traceback.print_exc())
                continue
            except Exception as e:
                logger.error(f"Unexpected error during encoding batch starting at index {i}: {e}")
                logger.error(traceback.print_exc())
                continue
            
        # Finalize token statistics if enabled
        if self.is_check_dim and self.global_token_counts:
            # Handle edge cases
            if self.stats['token_min'] == float('inf'):
                self.stats['token_min'] = 0
            # Calculate average from ALL global token counts
            self.stats['token_avg'] = sum(self.global_token_counts) / len(self.global_token_counts)
            logger.info(f"Token stats (GLOBAL - all batches): min={self.stats['token_min']}, max={self.stats['token_max']}, avg={self.stats['token_avg']:.2f}")
            logger.info(f"Truncation stats (GLOBAL): max_len_reached={self.stats['max_len_reached']}, total_samples={self.stats['total_samples']}")
            logger.info(f"Truncation stats (BATCH): max_len_reached={self.batch_stats['max_len_reached']}, total_samples={self.batch_stats['total_samples']}")
        
        # Concatenate all embeddings
        if len(all_embeddings) == 0:
            logger.error("No embeddings were generated. Returning empty tensor.")
            return torch.empty((0, self.model.config.hidden_size), device=self.device)
        
        # Concatenate on CPU first, then move to device if needed
        final_embeddings = torch.cat(all_embeddings, dim=0)
        
        # Move to device for consistency with rest of pipeline
        final_embeddings = final_embeddings.to(device=self.device, dtype=self.dtype)
        logger.info(f"Generated embeddings shape: {final_embeddings.shape}")
        
        return final_embeddings
    
    def create_labels(self, df: pd.DataFrame) -> torch.Tensor:
        labels = []
        for _, row in df.iterrows():
            # Support either 'vuln_id' or fallback to 'vuln' column
            vuln_field = row.get('vuln_id', None)
            if vuln_field is None and 'vuln' in row:
                vuln_field = row.get('vuln', None)
            if isinstance(vuln_field, str) and pd.notna(vuln_field) and vuln_field.strip():
                vuln_vec = vuln_to_label(vuln_field.split(';'))
            else:
                vuln_vec = vuln_to_label(None)
            vuln_tens = torch.tensor(vuln_vec, dtype=self.dtype, device=self.device)
            labels.append(vuln_tens)
        return torch.stack(labels)
    
    def pipeline_process_code_csv_to_features(self, df_list : List) :
        try:
            results = []
            logger.info("Starting raw code processing pipeline")
            for (project_name,df) in tqdm(df_list, desc="Processing dataframes"):

                # Create combined features
                logger.info(f"Processing dataframe {project_name}: Step 1 - Creating combined features")
                combined_texts = []
                for _, row in df.iterrows():
                    combined_text = self.create_combined_features(row)
                    combined_texts.append(combined_text)
                
                # Encode features
                logger.info(f"Processing dataframe {project_name}: Step 2 - Encoding features with CodeBERT")
                features = self.encode_text_batch(combined_texts)
                
                # Create labels
                logger.info(f"Processing dataframe {project_name}: Step 3 - Creating labels")
                labels = self.create_labels(df)
            
                # Compile statistics
                logger.info(f"Dataframe {project_name} - feature_shape: {features.shape}, label_shape: {labels.shape}")
                
                logger.info(f"Dataframe {project_name} completed successfully!")
                logger.info(f"Statistics: {json.dumps(self.stats, indent=2)}")
            
            logger.info(f"Pipeline completed! Processed {len(results)} dataframes")
            return results
        except Exception as e:
            logger.error(f"Error in pipeline_process_code_csv_to_features: {e}")
            logger.error(traceback.print_exc())
            return []

if __name__ == "__main__":
    # IMPORTANT: don't override sep_token with a new string unless you also resize embeddings.
    tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base', use_fast=True)
    embedding_model = AutoModel.from_pretrained('microsoft/codebert-base')
    # Keep embeddings size in sync with tokenizer (safe even if sizes already match)
    try:
        embedding_model.resize_token_embeddings(len(tokenizer))
    except Exception:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create a dummy DataFrame
    df = pd.DataFrame([
        {
            "filename": "file1.sol",
            "contract": "MyContract",
            "func": "foo",
            "func_raw_code": "function foo() public {}",
            "vuln_id": "VULN1;VULN2"
        },
        {
            "filename": "file2.sol",
            "contract": "OtherContract",
            "func": "bar",
            "func_raw_code": "function bar() public {}",
            "vuln_id": ""
        }
    ])

    # Instantiate extractor with dummy tokenizer/model
    extractor = RawCodeFeatureExtractor(
        tokenizer=tokenizer,
        model=embedding_model
    )

    # Run pipeline
    results = extractor.pipeline_process_code_csv_to_features([df])
    for k, v in results.items():
        print("Features shape:", v['features'].shape)
        print("Labels shape:", v['labels'].shape)
        print("Features:", v['features'])
        print("Labels:", v['labels'])
