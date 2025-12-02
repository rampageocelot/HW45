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
        # TODO: Implement __init__
        
        
        # Store configuration and other args
        # DO NOT MODIFY
        self.config    = config
        self.partition = partition
        self.tokenizer = tokenizer

        # Special token IDs from H4Tokenizer
        self.eos_token = tokenizer.eos_id
        self.sos_token = tokenizer.sos_id
        self.pad_token = tokenizer.pad_id

        # Set up data paths
        root = config.get("root", config.get("root_dir"))
        if root is None:
            raise ValueError("Config must contain 'root' or 'root_dir' for LMDataset.")
        self.text_dir = os.path.join(root, partition)

        # Get all .npy text files in the text directory in sorted order
        all_files = os.listdir(self.text_dir)
        self.text_files = sorted(
            os.path.join(self.text_dir, f)
            for f in all_files
            if f.endswith(".npy")
        )

        self.transcripts_shifted = []
        self.transcripts_golden  = []

        # Take subset if requested
        subset_size = self.config.get("subset_size", self.config.get("subset", None))
        if subset_size is not None:
            subset_size = int(subset_size)
            self.text_files = self.text_files[:subset_size]

        # Tracking variables
        self.total_chars  = 0
        self.total_tokens = 0
        self.text_max_len = 0

        print(f"Loading transcripts for {partition} partition...")
        for file in tqdm(self.text_files):
            # Load the transcript: np.load -> list -> join to string
            arr = np.load(file, allow_pickle=True)
            transcript = "".join(arr.tolist())

            # Track character count (before tokenization)
            self.total_chars += len(transcript)

            # Tokenize transcript
            tokenized = self.tokenizer.encode(transcript)

            # Track token count (excluding special tokens)
            self.total_tokens += len(tokenized)

            # Track max length (+1 for sos/eos)
            self.text_max_len = max(self.text_max_len, len(tokenized) + 1)

            # Create shifted and golden sequences
            shifted = [self.sos_token] + tokenized         # starts with SOS
            golden  = tokenized + [self.eos_token]         # ends with EOS

            self.transcripts_shifted.append(shifted)
            self.transcripts_golden.append(golden)

        # Average characters per token
        self.avg_chars_per_token = (
            self.total_chars / self.total_tokens if self.total_tokens > 0 else 0
        )

        # Verify alignment
        if len(self.transcripts_shifted) != len(self.transcripts_golden):
            raise ValueError("Shifted and golden transcripts are misaligned")

        # Dataset length
        self.length = len(self.transcripts_shifted)
        
    def get_avg_chars_per_token(self) -> float:
        '''
        Get the average number of characters per token. Used to calculate character-level perplexity.
        DO NOT MODIFY
        '''
        return self.avg_chars_per_token
    
    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        # TODO: Implement __len__
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
        # TODO: Implement __getitem__
        # Make sure you convert to the right type
        shifted = torch.LongTensor(self.transcripts_shifted[idx])
        golden  = torch.LongTensor(self.transcripts_golden[idx])
        return shifted, golden
    
    
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

        lengths = torch.LongTensor([t.size(0) for t in shifted_transcripts])

        padded_shifted = pad_sequence(
            shifted_transcripts, batch_first=True, padding_value=self.pad_token
        )
        padded_golden = pad_sequence(
            golden_transcripts, batch_first=True, padding_value=self.pad_token
        )

        return padded_shifted, padded_golden, lengths

    def sample_prompts(self, num_samples: int, prompt_length: int, seed: int = None) -> Tuple[torch.LongTensor, List[torch.LongTensor]]:
        """
        Sample random prompts of fixed length from the dataset and return their original sequences.
        DO NOT MODIFY
        """
        if seed is not None:
            np_state = np.random.get_state()
            np.random.seed(seed)

        prompts = []
        originals = []
        attempts = 0
        max_attempts = num_samples * 10

        while len(prompts) < num_samples and attempts < max_attempts:
            idx = np.random.randint(0, len(self))
            tokens = self.transcripts_shifted[idx][1:]  # remove sos token

            if len(tokens) < prompt_length:
                attempts += 1
                continue

            prompt_tokens = tokens[:prompt_length]

            prompts.append(torch.LongTensor([self.sos_token] + prompt_tokens))
            originals.append(torch.LongTensor(tokens + [self.eos_token]))

            attempts += 1

        if len(prompts) < num_samples:
            print(f"Warning: Could only sample {len(prompts)} valid prompts")

        if seed is not None:
            np.random.set_state(np_state)

        return torch.stack(prompts), originals