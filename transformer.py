import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):
    def __init__(self, time_dim, device, c_in=1, c_out=1, max_time_steps=5000):
        super(TransformerModel, self).__init__()
        self.device = device
        # Assuming embedding_dim is what you want for both the sequence embeddings and the time embeddings
        self.embedding_dim = time_dim
        self.nhead = c_in  # Adjust if needed; ensure it divides embedding_dim
        self.num_encoder_layers = 4
        self.dim_feedforward = 2048

        # Adjusted encoder to take the correct input size
        self.encoder = nn.Linear(time_dim, self.embedding_dim)  # Assuming input is already at time_dim size
        self.pos_encoder = PositionalEncoding(self.embedding_dim)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim, 
            nhead=self.nhead, 
            dim_feedforward=self.dim_feedforward
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=self.num_encoder_layers)
        
        self.decoder = nn.Linear(self.embedding_dim, c_out)
        # Adjusted time embedding to match the embedding dimension of sequence elements
        self.time_embedding = nn.Embedding(max_time_steps, self.embedding_dim)

        self.init_weights()
        
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.encoder.bias.data.zero_()
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.time_embedding.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, src, timesteps):
        # Adjust src to remove channel dimension and correctly match dimensions
        src = src.squeeze(1)  # Remove channel dimension, assuming it's always 1
        
        # Get timestep embeddings with correct shape
        time_embeddings = self.time_embedding(timesteps)  # Shape: (batch_size, seq_len, embedding_dim)
        
        # Concatenation is not needed if we are directly embedding to the correct dimension
        src = torch.cat((src, time_embeddings), dim=2)  # Assuming you want to add them instead
        
        # Assuming you want to add or use time embeddings directly
        #src = src + time_embeddings  # Element-wise addition of time embeddings
        
        src = self.encoder(src)
        src = src * torch.sqrt(torch.tensor(self.embedding_dim, device=self.device))
        src = src.transpose(0, 1)  # Transformer expects (seq_len, batch, features)
        src = self.pos_encoder(src)
        
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        output = output.transpose(0, 1)  # Reshape output to (batch_size, seq_len, features)
        return output



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x



