from enum import Enum
import torch
from transformers import PreTrainedModel, PretrainedConfig
from py_pytorch_chess_model import ChessModel
class ChessModelConfig(PretrainedConfig):
    model_type = "chess_model"
    
    def __init__(self, max_sequence_length=512, vocab_size=370, num_layers=3, n_head=8, d_embed=768, feed_forward_dim = 2048, dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.max_sequence_length=max_sequence_length
        self.vocab_size=vocab_size
        self.num_layers=num_layers
        self.n_head=n_head
        self.d_embed=d_embed
        self.feed_forward_dim=feed_forward_dim
        self.dropout=dropout
        
class ChessModelHF(PreTrainedModel):
    class InputTypes(Enum):
        TOKENS = 1,
        ONE_HOTS = 2,
        BERT_EMBEDDINGS = 3,
        TEXT_LS = 4
    
    def __init__(self, config):
        super().__init__(config)
        self.model = ChessModel(
            max_sequence_length=config.max_sequence_length,
            vocab_size=config.vocab_size,
            num_layers=config.num_layers,
            n_head=config.n_head,
            d_embed=config.d_embed,
            feed_forward_dim=config.feed_forward_dim,
            dropout=config.dropout
        )
        
    def forward(self, input_type: InputTypes, input_mask: torch.Tensor, input_tokens: torch.Tensor=None, input_one_hots: torch.Tensor=None, input_embeddings: torch.Tensor=None, input_text_ls: list[str] = None):
        if input_type == ChessModelHF.InputTypes.TOKENS:
            return self.model.forward_chess_tokens(input_tokens, input_mask)
        elif input_type == ChessModelHF.InputTypes.ONE_HOTS:
            return self.model.forward_one_hot(input_one_hots, input_mask)
        elif input_type == ChessModelHF.InputTypes.BERT_EMBEDDINGS:
            return self.model.forward_embeddings(input_embeddings, input_mask)
        elif input_type == ChessModelHF.InputTypes.TEXT_LS:
            return self.model.forward_text(input_text_ls)
        else:
            raise ValueError("No input type specified to ChessHFModel")
        
