import torch
import torch.nn as nn
from transformers import AutoModel


class transformer_lstm(nn.Module):
    """
    SapBERT encoder followed by a bidirectional LSTM classification head.

    When load_sapbert=True  (on-the-fly mode):
        forward() receives the raw tokenizer dict  {input_ids, attention_mask, …}
        and runs SapBERT internally (frozen, no_grad).

    When load_sapbert=False (precomputed mode):
        forward() receives a dict with two keys:
            'embeddings'    – float tensor  (B, seq_len, 768)  [stored as fp16,
                               cast to fp32 here before entering the LSTM]
            'attention_mask'– int/bool tensor (B, seq_len)

    Architecture:
        SapBERT  →  packed BiLSTM  →  concat final fwd+bwd hidden states
                 →  Dropout  →  Linear(lstm_hidden*2, 1)
    """

    def __init__(
        self,
        lstm_hidden_size: int = 256,
        lstm_num_layers:  int = 2,
        dropout:          float = 0.2,
        bidirectional:    bool  = True,
        device=None,
        load_sapbert:     bool  = True,
    ):
        super().__init__()

        self.device       = device
        self.load_sapbert = load_sapbert
        self.bidirectional = bidirectional
        self.lstm_num_layers = lstm_num_layers

        # ------------------------------------------------------------------ #
        # Optional SapBERT encoder (frozen)
        # ------------------------------------------------------------------ #
        if self.load_sapbert:
            self.sapbert = AutoModel.from_pretrained(
                "../helper_code/sapBERT_local_save", local_files_only=True
            )
            # Keep SapBERT frozen — we only train the LSTM head
            for param in self.sapbert.parameters():
                param.requires_grad = False
        else:
            self.sapbert = None

        # ------------------------------------------------------------------ #
        # Bidirectional LSTM
        # ------------------------------------------------------------------ #
        # inter-layer dropout is only applied when num_layers > 1
        lstm_dropout = dropout if lstm_num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )

        # ------------------------------------------------------------------ #
        # Classifier head
        # ------------------------------------------------------------------ #
        # If bidirectional, the final state is the concatenation of the
        # last forward layer's hidden state and the last backward layer's
        # hidden state → size = lstm_hidden_size * 2
        head_input_size = lstm_hidden_size * 2 if bidirectional else lstm_hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_input_size, 1),
        )

    # ---------------------------------------------------------------------- #
    def forward(self, x):
        # ------------------------------------------------------------------ #
        # 1. Get sequence embeddings
        # ------------------------------------------------------------------ #
        if self.load_sapbert:
            # x is the raw tokenizer output dict
            with torch.no_grad():
                with torch.amp.autocast(device_type=self.device.type):
                    outputs = self.sapbert(**x)
            sequence_output = outputs.last_hidden_state          # (B, seq_len, 768)
            attention_mask  = x['attention_mask']                 # (B, seq_len)
        else:
            # x is a dict from the precomputed dataset
            # Cast fp16 → fp32 so LSTM weights (fp32) work cleanly with autocast
            sequence_output = x['embeddings'].float()             # (B, seq_len, 768)
            attention_mask  = x['attention_mask']                 # (B, seq_len)

        # ------------------------------------------------------------------ #
        # 2. Pack padded sequences → LSTM → unpack
        #    pack_padded_sequence skips padding positions, which both speeds
        #    up computation and prevents the LSTM from learning from padding.
        # ------------------------------------------------------------------ #
        lengths = attention_mask.sum(dim=1).cpu().long()
        # Clamp to ≥ 1 to avoid zero-length sequences causing a crash
        lengths = lengths.clamp(min=1)

        packed_input = nn.utils.rnn.pack_padded_sequence(
            sequence_output, lengths, batch_first=True, enforce_sorted=False
        )

        _, (h_n, _) = self.lstm(packed_input)
        # h_n shape: (num_layers * num_directions, B, lstm_hidden_size)

        # ------------------------------------------------------------------ #
        # 3. Extract final hidden state
        #    For a 2-layer bidirectional LSTM the layer ordering in h_n is:
        #      index 0 → layer 1  forward
        #      index 1 → layer 1  backward
        #      index 2 → layer 2  forward   ← top forward  hidden state
        #      index 3 → layer 2  backward  ← top backward hidden state
        # ------------------------------------------------------------------ #
        if self.bidirectional:
            # Top-layer forward and backward hidden states
            final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (B, hidden*2)
        else:
            final_hidden = h_n[-1]                                  # (B, hidden)

        # ------------------------------------------------------------------ #
        # 4. Classify
        # ------------------------------------------------------------------ #
        return self.classifier(final_hidden).float()               # (B, 1)
