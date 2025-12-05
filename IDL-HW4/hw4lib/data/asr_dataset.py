from typing import Literal, Tuple, Optional
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset 
from torch.nn.utils.rnn import pad_sequence
import torchaudio.transforms as tat
from .tokenizer import H4Tokenizer

'''
TODO: Implement this class.

Specification:
The ASRDataset class provides data loading and processing for ASR (Automatic Speech Recognition):

1. Data Organization:
   - Handles dataset partitions (train-clean-100, dev-clean, test-clean)
   - Features stored as .npy files in fbank directory
   - Transcripts stored as .npy files in text directory
   - Maintains alignment between features and transcripts

2. Feature Processing:
   - Loads log mel filterbank features from .npy files
   - Supports multiple normalization strategies:
     * global_mvn: Global mean and variance normalization
     * cepstral: Per-utterance mean and variance normalization
     * none: No normalization
   - Applies SpecAugment data augmentation during training:
     * Time masking: Masks random time steps
     * Frequency masking: Masks random frequency bands

3. Transcript Processing:
   - Similar to LMDataset transcript handling
   - Creates shifted (SOS-prefixed) and golden (EOS-suffixed) versions
   - Tracks statistics for perplexity calculation
   - Handles tokenization using H4Tokenizer

4. Batch Preparation:
   - Pads features and transcripts to batch-uniform lengths
   - Provides lengths for packed sequence processing
   - Ensures proper device placement and tensor types

Key Requirements:
- Must maintain feature-transcript alignment
- Must handle variable-length sequences
- Must track maximum lengths for both features and text
- Must implement proper padding for batching
- Must apply SpecAugment only during training
- Must support different normalization strategies
'''



class ASRDataset(Dataset):
    def __init__(
            self,
            partition: Literal['train-clean-100', 'dev-clean', 'test-clean'],
            config: dict,
            tokenizer: H4Tokenizer,
            isTrainPartition: bool,
            global_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        self.config = config
        self.partition = partition
        self.isTrainPartition = isTrainPartition
        self.tokenizer = tokenizer

        # special tokens
        self.eos_token = tokenizer.eos_id
        self.sos_token = tokenizer.sos_id
        self.pad_token = tokenizer.pad_id

        root = config["root"]
        subset = config["subset"]
        num_feats = config["num_feats"]

        # ---------- feature files ----------
        self.fbank_dir = os.path.join(root, partition, "fbank")
        fbank_files = sorted(os.listdir(self.fbank_dir))
        subset_size = int(subset * len(fbank_files))
        self.fbank_files = fbank_files[:subset_size]
        self.length = len(self.fbank_files)

        # ---------- transcript files (not for test-clean) ----------
        if self.partition != "test-clean":
            self.text_dir = os.path.join(root, partition, "text")
            text_files = sorted(os.listdir(self.text_dir))
            self.text_files = text_files[:subset_size]

            if len(self.fbank_files) != len(self.text_files):
                raise ValueError("Number of feature and transcript files must match")

        # storage
        self.feats: list = []
        self.transcripts_shifted: list = []
        self.transcripts_golden: list = []

        # stats
        self.total_chars = 0
        self.total_tokens = 0
        self.feat_max_len = 0
        self.text_max_len = 0

        # Welford accumulators for global MVN
        if self.config["norm"] == "global_mvn" and global_stats is None:
            if not isTrainPartition:
                raise ValueError(
                    "global_stats must be provided for non-training partitions when using global_mvn"
                )
            count = 0
            mean = torch.zeros(num_feats, dtype=torch.float64)
            M2 = torch.zeros(num_feats, dtype=torch.float64)

        # ---------- load everything ----------
        print(f"Loading data for {partition} partition...")
        for i in tqdm(range(self.length)):
            # features: (num_feats, time)
            feat_path = os.path.join(self.fbank_dir, self.fbank_files[i])
            feat_np = np.load(feat_path, allow_pickle=True)
            feat_np = feat_np[:num_feats, :]           # truncate to num_feats
            self.feats.append(feat_np)

            # track longest feature length
            self.feat_max_len = max(self.feat_max_len, feat_np.shape[1])

            # update global stats if needed
            if self.config["norm"] == "global_mvn" and global_stats is None:
                feat_tensor = torch.as_tensor(feat_np, dtype=torch.float32)  # (F, T)
                batch_count = feat_tensor.shape[1]
                count += batch_count

                delta = feat_tensor - mean.unsqueeze(1)
                mean += delta.mean(dim=1)
                delta2 = feat_tensor - mean.unsqueeze(1)
                M2 += (delta * delta2).sum(dim=1)

            # transcripts (skip for test-clean)
            if self.partition != "test-clean":
                text_path = os.path.join(self.text_dir, self.text_files[i])
                transcript_arr = np.load(text_path, allow_pickle=True)
                transcript_str = " ".join(map(str, transcript_arr.tolist()))

                # chars before tokenization
                self.total_chars += len(transcript_str)

                # tokenize
                token_ids = tokenizer.encode(transcript_str)

                # token stats (no specials)
                self.total_tokens += len(token_ids)
                self.text_max_len = max(self.text_max_len, len(token_ids) + 1)

                # shifted (SOS + tokens), golden (tokens + EOS)
                shifted = [self.sos_token, *token_ids]
                golden = token_ids + [self.eos_token]

                self.transcripts_shifted.append(shifted)
                self.transcripts_golden.append(golden)

        # avg chars per token
        self.avg_chars_per_token = (
            self.total_chars / self.total_tokens if self.total_tokens > 0 else 0
        )

        # alignment check
        if self.partition != "test-clean":
            if not (
                len(self.feats)
                == len(self.transcripts_shifted)
                == len(self.transcripts_golden)
            ):
                raise ValueError("Features and transcripts are misaligned")

        # finalize global stats
        if self.config["norm"] == "global_mvn":
            if global_stats is not None:
                self.global_mean, self.global_std = global_stats
            else:
                variance = M2 / (count - 1)
                self.global_std = torch.sqrt(variance + 1e-8).float()
                self.global_mean = mean.float()

        # SpecAugment transforms
        self.time_mask = tat.TimeMasking(
            time_mask_param=config["specaug_conf"]["time_mask_width_range"],
            iid_masks=True,
        )
        self.freq_mask = tat.FrequencyMasking(
            freq_mask_param=config["specaug_conf"]["freq_mask_width_range"],
            iid_masks=True,
        )

    def get_avg_chars_per_token(self):
        return self.avg_chars_per_token

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat_np = self.feats[idx]
        feat = torch.as_tensor(feat_np, dtype=torch.float32)

        # normalization
        if self.config["norm"] == "global_mvn":
            assert (
                self.global_mean is not None and self.global_std is not None
            ), "Global mean and std must be computed before normalization"
            feat = (feat - self.global_mean.unsqueeze(1)) / (
                self.global_std.unsqueeze(1) + 1e-8
            )
        elif self.config["norm"] == "cepstral":
            feat = (feat - feat.mean(dim=1, keepdim=True)) / (
                feat.std(dim=1, keepdim=True) + 1e-8
            )
        # 'none' => no change

        shifted_tensor, golden_tensor = None, None
        if self.partition != "test-clean":
            shifted_tensor = torch.as_tensor(
                self.transcripts_shifted[idx], dtype=torch.long
            )
            golden_tensor = torch.as_tensor(
                self.transcripts_golden[idx], dtype=torch.long
            )

        return feat, shifted_tensor, golden_tensor

    def collate_fn(
        self, batch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # features: list of (F, T) -> make (T, F) for pad_sequence
        feat_seq_list = [item[0].transpose(0, 1) for item in batch]  # (T, F)
        feat_lengths = torch.as_tensor(
            [seq.shape[0] for seq in feat_seq_list], dtype=torch.long
        )

        padded_feats = pad_sequence(feat_seq_list, batch_first=True)  # (B, T, F)

        padded_shifted, padded_golden, transcript_lengths = None, None, None
        if self.partition != "test-clean":
            shifted_list = [item[1] for item in batch]
            golden_list = [item[2] for item in batch]

            transcript_lengths = torch.as_tensor(
                [g.shape[0] for g in golden_list], dtype=torch.long
            )

            padded_shifted = pad_sequence(
                shifted_list, batch_first=True, padding_value=self.pad_token
            )
            padded_golden = pad_sequence(
                golden_list, batch_first=True, padding_value=self.pad_token
            )

        # SpecAugment only on training partition
        if self.config["specaug"] and self.isTrainPartition:
            padded_feats = padded_feats.permute(0, 2, 1)  # (B, F, T)

            if self.config["specaug_conf"]["apply_freq_mask"]:
                for _ in range(self.config["specaug_conf"]["num_freq_mask"]):
                    padded_feats = self.freq_mask(padded_feats)

            if self.config["specaug_conf"]["apply_time_mask"]:
                for _ in range(self.config["specaug_conf"]["num_time_mask"]):
                    padded_feats = self.time_mask(padded_feats)

            padded_feats = padded_feats.permute(0, 2, 1)  # (B, T, F)

        return padded_feats, padded_shifted, padded_golden, feat_lengths, transcript_lengths

