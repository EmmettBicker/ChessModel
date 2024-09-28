import torch
from transformers import BertModel, BertTokenizer

class EmbeddingFromSentence():
    def __init__(self, max_sequence_length=512, chess_vocab_size = 370):
        self.max_sequence_length = max_sequence_length
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased', clean_up_tokenization_spaces=True)
        self.bert_model: BertModel= BertModel.from_pretrained('bert-base-uncased')
        
        self.vocab_size = chess_vocab_size
        # extremely hack-y implementation of vocab
        vocab = set()
        with open("games.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                for move in line.split(" "):
                    vocab.add(move)
        vocab.remove("\n")
        token_to_idx = {}
        vc = sorted(list(vocab))
        for idx, word in enumerate(vc):
            token_to_idx[word] = idx + 1
        self.token_to_idx = token_to_idx

    def get_embeddings_from_sentence(self, word_ls: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            encoded_text = self.tokenizer(word_ls, max_length=self.max_sequence_length, truncation=True, padding='max_length', return_tensors="pt")
            word_embeddings = self.bert_model.get_input_embeddings()
            attn_mask, embeddings = (encoded_text['attention_mask'], word_embeddings(encoded_text['input_ids']))
            return embeddings, attn_mask, encoded_text['input_ids']
        
    def get_one_hot_from_sentence(self, word_ls: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            encoded_text = self.tokenizer(word_ls, max_length=self.max_sequence_length, truncation=True, padding='max_length', return_tensors="pt")
            word_embeddings = self.bert_model.get_input_embeddings()
            attn_mask, embeddings = (encoded_text['attention_mask'], word_embeddings(encoded_text['input_ids']))
            return embeddings, attn_mask, encoded_text['input_ids']
    
    def one_hot_from_sentence(self, sequence_list: list[str]):
        batch_size = len(sequence_list)
        output_tensor = torch.zeros(batch_size, self.max_sequence_length, self.vocab_size, dtype=torch.float)
        attn_mask = torch.zeros(batch_size, self.max_sequence_length, dtype=torch.long)
        ids = torch.zeros(batch_size, self.max_sequence_length, dtype=torch.long)
        for seq_idx, sequence in enumerate(sequence_list):
            last_idx = 0
            for word_idx, word in enumerate(sequence.split(" ")):
                last_idx = word_idx
                if word_idx >= self.max_sequence_length:
                    break
                if word in self.token_to_idx:
                    id = self.token_to_idx[word]
                    ids[seq_idx, word_idx] = id
                    output_tensor[seq_idx, word_idx, id] = 1
                else:
                   id = 0
                   ids[seq_idx, word_idx] = id
                   output_tensor[seq_idx, word_idx, id] = 1
                attn_mask[seq_idx, word_idx] = 1
            if last_idx < self.max_sequence_length:
                id = self.vocab_size - 1
                ids[seq_idx, last_idx+1] = id #eos will be treated a 369 (there is no 370 because of off by one )
                attn_mask[seq_idx, word_idx] = 1
                output_tensor[seq_idx, word_idx, id] = 1
        return output_tensor, attn_mask, ids 
                