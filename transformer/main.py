import json
import torch
import torch.nn as nn
import torch.optim as optim

###############################################################################
# 1. Load all_tickers_combined.json
###############################################################################
# The file structure is assumed to be:
# {
#   "tickers_order": ["VIX", "BTC-USD", ...],
#   "data": {
#       "2021-01-01": [val_ticker0, val_ticker1, ...],
#       "2021-01-02": [...],
#       ...
#   }
# }
#
# We'll treat each date as one "token," and the list of values across tickers
# as the token dimension.

JSON_FILE = "all_tickers_combined.json"

with open(JSON_FILE, "r") as f:
    combined = json.load(f)

tickers_order = combined["tickers_order"]  # e.g. ["VIX", "BTC-USD", ...]
data_dict = combined["data"]              # a dict of date_str -> list of floats

# Sort the dates for a consistent sequence
all_dates = sorted(data_dict.keys())

# Build a list of [val_ticker0, val_ticker1, ...] for each date in ascending order
sequence_data = []
for d in all_dates:
    sequence_data.append(data_dict[d])  # shape will be [#dates, #tickers]

# Convert to a torch Tensor of shape (#dates, #tickers)
sequence_data = torch.tensor(sequence_data, dtype=torch.float32)

# We want shape (batch_size=1, seq_len=#dates, ticker_dim=len(tickers))
BATCH_SIZE = 1
SEQ_LEN = sequence_data.shape[0]
TICKER_DIM = sequence_data.shape[1]

# Unsqueeze to get (1, seq_len, ticker_dim)
inputs = sequence_data.unsqueeze(0)  # (1, SEQ_LEN, TICKER_DIM)
# For demo, let's just predict the same sequence as targets
targets = inputs.clone()

print(f"Loaded {SEQ_LEN} tokens (dates), each of dimension {TICKER_DIM}.")
print(f"Input shape: {inputs.shape}, Target shape: {targets.shape}")

###############################################################################
# 2. Hyperparameters
###############################################################################
D_MODEL = 32      # Transformer model dimension
NHEAD = 4         # Number of attention heads
NUM_LAYERS = 2    # Number of decoder layers
LR = 1e-3         # Learning rate
EPOCHS = 5

###############################################################################
# 3. Positional Encoding
###############################################################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        seq_len = x.size(0)
        return x + self.pe[:seq_len, :]

###############################################################################
# 4. Decoder-Only Transformer
###############################################################################
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, input_dim):
        super(DecoderOnlyTransformer, self).__init__()
        
        # Project from ticker dimension -> d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            activation='relu'
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Final layer projects back from d_model -> input_dim
        self.output_projection = nn.Linear(d_model, input_dim)
        
    def generate_causal_mask(self, seq_len):
        # Upper-triangular mask for causal (autoregressive) attention
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        return mask

    def forward(self, x):
        """
        :param x: shape (batch_size, seq_len, input_dim)
        :return: predicted shape (batch_size, seq_len, input_dim)
        """
        batch_size, seq_len, input_dim = x.shape
        
        # (batch, seq_len, input_dim) -> (seq_len, batch, input_dim)
        x = x.permute(1, 0, 2)
        
        # Project to d_model
        x = self.input_projection(x)  # (seq_len, batch, d_model)
        
        # Positional Encoding
        x = self.pos_encoder(x)       # (seq_len, batch, d_model)
        
        # Causal Mask
        mask = self.generate_causal_mask(seq_len).to(x.device)  # (seq_len, seq_len)
        
        # Decode
        decoded = self.transformer_decoder(
            tgt=x, memory=x, tgt_mask=mask
        )  # (seq_len, batch, d_model)
        
        # Project back to input_dim
        out = self.output_projection(decoded)  # (seq_len, batch, input_dim)
        
        # Re-permute to (batch, seq_len, input_dim)
        out = out.permute(1, 0, 2)
        return out

###############################################################################
# 5. Instantiate Model
###############################################################################
model = DecoderOnlyTransformer(
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    input_dim=TICKER_DIM
)

###############################################################################
# 6. Setup Loss/Optimizer
###############################################################################
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

###############################################################################
# 7. Training
###############################################################################
model.train()
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    preds = model(inputs)  # forward pass
    loss = criterion(preds, targets)
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

###############################################################################
# 8. Quick Check
###############################################################################
model.eval()
with torch.no_grad():
    sample_out = model(inputs)
    print("Sample output shape:", sample_out.shape)
    # Should be (1, SEQ_LEN, TICKER_DIM)
