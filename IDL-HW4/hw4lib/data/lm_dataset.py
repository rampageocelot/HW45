from typing import Tuple, List
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset 
from torch.nn.utils.rnn import pad_sequence
from .tokenizer import H4Tokenizer

'''
TODO: Implement this class.

Specification:
- Dataset for training and evaluating language models
- Loads text data from files and tokenizes them
- Handles data subsetting based on configuration
- Creates shifted and golden (target) versions of sequences
- Tracks dataset statistics (chars, tokens, lengths)
- Provides collation function for batching
- Supports random prompt sampling for generation

Key Requirements:
- Each sequence should start with SOS token in shifted version
- Each sequence should end with EOS token in golden version
- Padding should use the designated pad token
- Sequences within a batch must be padded to same length
- Must track character and token counts for perplexity calculation
- Must verify alignment between shifted and golden sequences
'''
    
class LMDataset(Dataset):
    """
    Dataset for Language Model training/evaluation.
    """
    def __init__(
            self, 
            partition: str, 
            config: dict, 
            tokenizer: H4Tokenizer
    ):
        """
        Initializes the Language Model Dataset for training language models on text data.

        Args:
            partition (str): Data partition subdirectory under root (e.g., 'train', 'test')
            config (dict): Configuration dictionary containing dataset settings
            tokenizer (H4Tokenizer): Tokenizer for encoding/decoding text
        """
        # Store configuration and other args
        # DO NOT MODIFY
        self.config    = config
        self.partition = partition
        self.tokenizer = tokenizer

        # Get tokenizer ids for special tokens (eos, sos, pad)
        self.eos_token = tokenizer.eos_id
        self.sos_token = tokenizer.sos_id
        self.pad_token = tokenizer.pad_id

        # ---------- PATHS & FILE LIST (MATCH p2) ----------
        # Join root and partition to get the text directory
        self.text_dir = os.path.join(config["root"], partition)

        # Get all text files in the text directory in sorted order  
        # (no extension filtering, like p2)
        self.text_files = sorted(os.listdir(self.text_dir))

        # Take subset as a FRACTION of files (subset \in [0,1])
        subset_size = int(config["subset"] * len(self.text_files))
        self.text_files = self.text_files[:subset_size]

        # Initialize lists to store transcripts
        self.transcripts_shifted = []
        self.transcripts_golden  = []
        
        # Initialize tracking variables
        # DO NOT MODIFY 
        self.total_chars  = 0
        self.total_tokens = 0
        self.text_max_len = 0
        
        print(f"Loading transcripts for {partition} partition...")
        for file in tqdm(self.text_files):
            # Load the transcript (match p2 behavior)
            transcript_arr = np.load(os.path.join(self.text_dir, file), allow_pickle=True)

            if len(transcript_arr.shape) > 0:
                # Flatten and join with spaces
                transcript = " ".join(transcript_arr.astype(str).reshape(-1))
            else:
                transcript = str(transcript_arr)
            
            # Track character count (before tokenization)
            # DO NOT MODIFY
            self.total_chars += len(transcript)
            
            # Use tokenizer to encode the transcript
            tokenized = tokenizer.encode(transcript)
            
            # Track token count (excluding special tokens)
            # DO NOT MODIFY
            self.total_tokens += len(tokenized)
            
            # Track max length (add 1 for the sos/eos tokens)
            # DO NOT MODIFY
            self.text_max_len = max(self.text_max_len, len(tokenized)+1)
            
            # Create shifted and golden versions by adding sos and eos tokens
            shifted = [self.sos_token] + tokenized
            golden  = tokenized + [self.eos_token]

            self.transcripts_shifted.append(shifted)
            self.transcripts_golden.append(golden)

        # Calculate average characters per token
        # DO NOT MODIFY
        self.avg_chars_per_token = self.total_chars / self.total_tokens if self.total_tokens > 0 else 0
        
        # Verify data alignment
        # DO NOT MODIFY
        if not (len(self.transcripts_shifted) == len(self.transcripts_golden)):
            raise ValueError("Shifted and golden transcripts are misaligned")
        
        # Store the length of the dataset (either is fine; they’re equal)
        self.length = len(self.transcripts_golden)
        
    def get_avg_chars_per_token(self) -> float:
        '''
        Get the average number of characters per token. Used to calculate character-level perplexity.
        DO NOT MODIFY
        '''
        return self.avg_chars_per_token
    
    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """
        Get a single sample from the dataset.

        Args:
            idx (int): Sample index

        Returns:
            tuple: (shifted_transcript, golden_transcript) where:
                - shifted_transcript: LongTensor starting with SOS token
                - golden_transcript: LongTensor ending with EOS token
        """
        shifted = self.transcripts_shifted[idx]
        golden  = self.transcripts_golden[idx]
        return torch.tensor(shifted, dtype=torch.long), torch.tensor(golden, dtype=torch.long)
    
    
    def collate_fn(self, batch: List[Tuple[torch.LongTensor, torch.LongTensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collate and pad a batch of samples to create a batch of fixed-length padded shifted and golden transcripts.

        Args:
            batch (list): List of (shifted, golden) transcript pairs

        Returns:
            tuple: (padded_shifted, padded_golden, lengths) where:
                - padded_shifted: Tensor of shape (batch, max_len) with SOS prefixes
                - padded_golden: Tensor of shape (batch, max_len) with EOS suffixes
                - lengths: Original sequence lengths before padding
        """
        shifted_transcripts, golden_transcripts = zip(*batch)
        
        # p2 uses lengths from golden_transcripts (same length as shifted, but matching exactly)
        lengths = torch.tensor([transcript.shape[0] for transcript in golden_transcripts])  # (B,)

        padded_shifted = pad_sequence(
            shifted_transcripts, batch_first=True, padding_value=self.pad_token
        )
        padded_golden  = pad_sequence(
            golden_transcripts, batch_first=True, padding_value=self.pad_token
        )

        return padded_shifted, padded_golden, lengths

    def sample_prompts(self, num_samples: int, prompt_length: int, seed: int = None) -> Tuple[torch.LongTensor, List[torch.LongTensor]]:
        """
        Sample random prompts of fixed length from the dataset and return their original sequences.
        DO NOT MODIFY
        
        Args:
            num_samples: Number of prompts to sample
            prompt_length: Exact number of tokens for each prompt
            seed: Random seed for reproducibility. If None, no seed is set.
            
        Returns:
            tuple: (prompts, originals) where:
                - prompts: torch.LongTensor of tokenized prompts
                - originals: List of torch.LongTensor containing complete original sequences
        """
        # Set random seed if provided
        if seed is not None:
            # Save current random state
            np_state = np.random.get_state()
            # Set seed for sampling
            np.random.seed(seed)
            
        prompts = []
        originals = []
        attempts = 0
        max_attempts = num_samples * 10  # Prevent infinite loops
        
        while len(prompts) < num_samples and attempts < max_attempts:
            # Sample random transcript
            idx = np.random.randint(0, len(self))
            tokens = self.transcripts_shifted[idx][1:]  # remove sos token
            
            # Skip if transcript is too short
            if len(tokens) < prompt_length:
                attempts += 1
                continue
                
            # Get exactly prompt_length tokens
            prompt_tokens = tokens[:prompt_length]
            
            # Store prompt and original sequence
            prompts.append(torch.LongTensor([self.sos_token] + prompt_tokens))
            originals.append(torch.LongTensor(tokens + [self.eos_token]))
            
            attempts += 1
            
        if len(prompts) < num_samples:
            print(f"Warning: Could only sample {len(prompts)} valid prompts")
        
        # Restore random state if seed was set
        if seed is not None:
            np.random.set_state(np_state)
            
        # No need for another LongTensor conversion since prompts are already tensors
        return torch.stack(prompts), originals
