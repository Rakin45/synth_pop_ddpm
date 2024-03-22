import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerWithPositionalEncoding(nn.Module):
    def __init__(self, feature_size=8, num_layers=3, dropout=0.1, max_time_steps=144):
        super().__init__()
        self.feature_size = feature_size
        self.time_embedding = nn.Embedding(max_time_steps, feature_size)

        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=3, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, feature_size)                                                                                                    
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.time_embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, timesteps):
        timesteps = timesteps + 1
        #print("Original src shape:", src.shape)
        #print("Timesteps values:", timesteps)
        assert timesteps.max() <= self.time_embedding.num_embeddings, "Timestep exceeds max_time_steps."
        assert timesteps.min() >= 1, "Timestep is below 1."  # Adjust based on your actual timestep range
        time_encodings = self.time_embedding(timesteps)  # Shape: [batch_size, feature_size]
        #print("Time encodings shape:", time_encodings.shape)
        
    
        # Ensure time_encodings is broadcastable to src's shape
        time_encodings = time_encodings.unsqueeze(1)  # Add sequence length dim: [batch_size, 1, feature_size]
        #print("Time encodings shape after unsqueeze:", time_encodings.shape)
        time_encodings = time_encodings.expand(-1, src.size(1), -1)  # Expand to match src: [batch_size, sequence_length, feature_size]
        #print("Time encodings shape after expand:", time_encodings.shape)

        # Apply positional encoding to the input features
        src = self.pos_encoder(src + time_encodings)
        #print("Src shape after adding time encodings:", src.shape)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        #print("PE shape:", self.pe[:x.size(0), :].shape)
        x = x + self.pe[:x.size(0), :]
        #print("X shape after adding PE:", x.shape)
        return x