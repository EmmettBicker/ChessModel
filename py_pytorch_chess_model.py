import torch 
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel

from py_get_bert_word_embeddings import EmbeddingFromSentence

MAX_SEQUENCE_LENGTH = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pytorch's   positional encoding implementaiton
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
    
class CausalMHAAndMLPBlock(nn.Module):
    def __init__(self, max_sequence_length, n_head=8, d_embed=768, feed_forward_dim=2048, dropout=0):
        super().__init__()
        
        self.positional_encoding = PositionalEncoding(d_embed)
        self.mha = nn.MultiheadAttention(d_embed, n_head, dropout=dropout, batch_first=True)
        self.causal_mha = self.CausalMHABlock(max_sequence_length)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_embed, feed_forward_dim),
            nn.GELU(),
            nn.Linear(feed_forward_dim, d_embed)
        ) 
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(normalized_shape=d_embed)
        self.norm2 = nn.LayerNorm(normalized_shape=d_embed)
        

        
    def __call__(self, l_embeds: torch.Tensor, padding_mask: torch.Tensor):
        return self.forward(l_embeds, padding_mask)
    
    def forward(self, l_embeds: torch.Tensor, padding_mask: torch.Tensor):
        l_embeds = self.positional_encoding(l_embeds)
        l_embeds = self.dropout(self.causal_mha(l_embeds, padding_mask)) + l_embeds
        l_embeds = self.norm1(l_embeds)
        
        original_shape = l_embeds.shape
        valid_tokens = ~padding_mask
        
        feed_forward_output = self.feed_forward(l_embeds[valid_tokens])
        updated_embeddings = torch.zeros(original_shape, device=l_embeds.device, dtype=l_embeds.dtype)
        updated_embeddings[valid_tokens] = feed_forward_output
        
        l_embeds = self.dropout(updated_embeddings) + l_embeds
        l_embeds = self.norm2(l_embeds)
        return l_embeds 


    def CausalMHABlock(self, max_sequence_length: int):
        def causalMHABlock(embeddings_matrix: torch.Tensor, padding_mask: torch.Tensor):
            causal_mask = torch.triu(torch.ones(max_sequence_length, max_sequence_length), diagonal=1)
            causal_mask = causal_mask.bool().to(embeddings_matrix.device)
            
            output, _ = self.mha.forward(embeddings_matrix, embeddings_matrix, embeddings_matrix, key_padding_mask=padding_mask, need_weights=False, attn_mask=causal_mask, is_causal=True)
            return output[0] 
        
        return causalMHABlock
    

class ChessModel(nn.Module):
    def __init__(self, max_sequence_length, vocab_size=370, num_layers=3, n_head=8, d_embed=768, feed_forward_dim = 2048, dropout=0):
        super().__init__()
        self.d_embed = d_embed
        self.embedder = EmbeddingFromSentence(max_sequence_length=MAX_SEQUENCE_LENGTH)

        self.out_linear = nn.Linear(self.d_embed, vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        self.layers = nn.ModuleList(
            [CausalMHAAndMLPBlock(max_sequence_length,n_head, d_embed, feed_forward_dim, dropout) for _ in range(num_layers)]
        )
        self.vocab_size = vocab_size
        self.one_hot_embedder = nn.Linear(vocab_size, d_embed)
        
       
        
    def forward_text(self, sentences: torch.Tensor):
        embeddings, attention_mask, _ = self.embedder.get_embeddings_from_sentence(sentences)
        padding_mask = attention_mask.eq(0)
        for layer in self.layers:
            embeddings = layer(embeddings, padding_mask)


        embeddings = self.out_linear(embeddings)
        embeddings = self.softmax(embeddings)
        return embeddings
    
    def forward_embeddings(self, embeddings: torch.Tensor, attention_mask: torch.Tensor):
        padding_mask = attention_mask.eq(0)
        for layer in self.layers:
            embeddings = layer(embeddings, padding_mask)


        embeddings = self.out_linear(embeddings)
        embeddings = self.softmax(embeddings)
        return embeddings
    
    def forward_one_hot(self, one_hot: torch.Tensor, attention_mask: torch.Tensor):
        embeddings = self.one_hot_embedder(one_hot)
        return self.forward_embeddings(embeddings, attention_mask)
    
    def forward_chess_tokens(self, tokens: torch.Tensor, attention_mask: torch.Tensor):
        one_hot = F.one_hot(tokens, self.vocab_size).float()
        return self.forward_one_hot(one_hot, attention_mask)
    
    def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor):
        return self.forward_chess_tokens(tokens, attention_mask)
        
    def __call__(self, tokens, padding_mask):
        return self.forward(tokens, padding_mask)

