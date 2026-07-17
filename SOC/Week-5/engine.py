import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import tiktoken

# 1. Setup Local Device Configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Running inference locally on: {device}")

# 2. Setup Tokenizer and Core Constants
enc = tiktoken.get_encoding("gpt2")
PAD = enc.n_vocab
BOS = enc.n_vocab + 1
EOS = enc.n_vocab + 2
VOCAB_SIZE = enc.n_vocab + 3
MAX_LEN = 64

# ==========================================\n# 3. EXACT MODEL ARCHITECTURE CODE
# ==========================================\n
class Head(nn.Module):
    def __init__(self, d_model, head_size):
        super().__init__()
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.key   = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        self.head_size = head_size
        self.dropout = nn.Dropout(0.1)

    def forward(self, q, k, v, mask=None):
        B, T, C = q.shape
        Q = self.query(q)
        K = self.key(k)
        V = self.value(v)
        wei = Q @ K.transpose(-2, -1) * (self.head_size ** -0.5)
        if mask is not None:
            wei = wei.masked_fill(mask == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ V

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()
        head_size = d_model // num_heads
        self.heads = nn.ModuleList([Head(d_model, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, q, k, v, mask=None):
        out = torch.cat([h(q=q, k=k, v=v, mask=mask) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(0.1)
        )
    def forward(self, x):
        return self.net(x)

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attn = MultiHeadAttention(num_heads, d_model)
        self.ff = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        norm_x = self.norm1(x)
        x = x + self.attn(q=norm_x, k=norm_x, v=norm_x, mask=mask)
        x = x + self.ff(self.norm2(x))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attn = MultiHeadAttention(num_heads, d_model)
        self.cross_attn = MultiHeadAttention(num_heads, d_model)
        self.ff = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_out, self_mask=None, cross_mask=None):
        norm_x = self.norm1(x)
        x = x + self.attn(q=norm_x, k=norm_x, v=norm_x, mask=self_mask)
        norm_x = self.norm2(x)
        x = x + self.cross_attn(q=norm_x, k=encoder_out, v=encoder_out, mask=cross_mask)
        x = x + self.ff(self.norm3(x))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_len=64):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len)
        self.encoder_layers = nn.ModuleList([EncoderBlock(d_model, num_heads) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderBlock(d_model, num_heads) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_causal_mask(self, size, device):
        return (torch.triu(torch.ones(size, size, device=device), diagonal=1) == 0).float()

    def forward(self, src, tgt):
        device = src.device
        _, tgt_seq_len = tgt.shape
        src_mask = (src != PAD).unsqueeze(1)
        tgt_causal = self.generate_causal_mask(tgt_seq_len, device)
        tgt_pad = (tgt != PAD).unsqueeze(1)
        tgt_mask = tgt_pad * tgt_causal

        enc_out = self.pos_emb(self.token_emb(src))
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, mask=src_mask)

        dec_out = self.pos_emb(self.token_emb(tgt))
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out, self_mask=tgt_mask, cross_mask=src_mask)
        return self.fc_out(dec_out)

# ==========================================\n# 4. TRANSLATION ENGINE & RUNTIME
# ==========================================\n
def translate(model, sentence, enc, max_len=64, temperature=0.3, top_k=5):
    model.eval()
    tokens = enc.encode(sentence) + [EOS]
    tokens = tokens[:max_len]
    tokens += [PAD] * (max_len - len(tokens))
    src = torch.tensor([tokens], dtype=torch.long, device=device)
    tgt_tokens = [BOS]

    with torch.no_grad():
        for _ in range(max_len):
            tgt_tensor = torch.tensor([tgt_tokens], dtype=torch.long, device=device)
            outputs = model(src, tgt_tensor)
            logits = outputs[0, -1, :] / temperature
            v, ix = torch.topk(logits, top_k)
            logits[logits < v[-1]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            if next_token == EOS or next_token == PAD or len(tgt_tokens) >= max_len:
                break
            tgt_tokens.append(next_token)

    output_tokens = [t for t in tgt_tokens if t < enc.n_vocab]
    return enc.decode(output_tokens) if output_tokens else "[No translation generated]"

# 5. Load Trained Weights & Execute Interactive Test Loop
if __name__ == "__main__":
    # Corrected keyword argument names mapped directly to architecture definition
    model = Transformer(vocab_size=VOCAB_SIZE, d_model=512, num_heads=8, num_layers=3).to(device)
    
    try:
        model.load_state_dict(torch.load('transformer_translator.pt', map_location=device))
        print("🎉 Success: Model weights loaded locally!")
    except FileNotFoundError:
        print("❌ Error: Could not find 'transformer_translator.pt' in this folder.")
        exit()

    print("\n--- Interactive Local Translator (Type 'exit' to quit) ---")
    while True:
        user_input = input("\nEnter English sentence: ")
        if user_input.strip().lower() == 'exit':
            break
        result = translate(model, user_input, enc, temperature=0.5)
        print(f"French Translation: {result}")