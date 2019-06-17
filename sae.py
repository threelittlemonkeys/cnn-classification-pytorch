from utils import *
import numpy as np

class sae(nn.Module): # self attentive encoder
    def __init__(self, vocab_size, embed_size = 512):
        super().__init__()
        dim = embed_size
        num_layers = 1

        # architecture
        self.embed = nn.Embedding(vocab_size, dim, padding_idx = PAD_IDX)
        self.pe = pos_encoder(dim)
        self.layers = nn.ModuleList([enc_layer(dim) for _ in range(num_layers)])

        if CUDA:
            self = self.cuda()

    def forward(self, x):
        mask = maskset(x)
        x = self.embed(x)
        h = x + self.pe(x.size(1))
        for layer in self.layers:
            h = layer(h, mask[0])
        return y

class pos_encoder(nn.Module): # positional encoder
    def __init__(self, dim, maxlen = 1000):
        super().__init__()
        self.pe = Tensor(maxlen, dim)
        pos = torch.arange(0, maxlen, 1.).unsqueeze(1)
        k = torch.exp(-np.log(10000) * torch.arange(0, dim, 2.) / dim)
        self.pe[:, 0::2] = torch.sin(pos * k)
        self.pe[:, 1::2] = torch.cos(pos * k)

    def forward(self, n):
        return self.pe[:n]

class enc_layer(nn.Module): # encoder layer
    def __init__(self, dim):
        super().__init__()

        # architecture
        self.attn = attn_mh(dim)
        self.ffn = ffn(dim)

    def forward(self, x, mask):
        z = self.attn(x, x, x, mask)
        z = self.ffn(z)
        return z

class attn_mh(nn.Module): # multi-head attention
    def __init__(self, dim):
        super().__init__()
        self.D = dim # dimension of model
        self.H = 8 # number of heads
        self.Dk = self.D // self.H # dimension of key
        self.Dv = self.D // self.H # dimension of value

        # architecture
        self.Wq = nn.Linear(self.D, self.H * self.Dk) # query
        self.Wk = nn.Linear(self.D, self.H * self.Dk) # key for attention distribution
        self.Wv = nn.Linear(self.D, self.H * self.Dv) # value for context representation
        self.Wo = nn.Linear(self.H * self.Dv, self.D)
        self.dropout = nn.Dropout(DROPOUT)
        self.norm = nn.LayerNorm(self.D)

    def attn_sdp(self, q, k, v, mask): # scaled dot-product attention
        c = np.sqrt(self.Dk) # scale factor
        a = torch.matmul(q, k.transpose(2, 3)) / c # compatibility function
        a = a.masked_fill(mask, -10000) # masking in log space
        a = F.softmax(a, -1)
        a = torch.matmul(a, v)
        return a # attention weights

    def forward(self, q, k, v, mask):
        x = q # identity
        q = self.Wq(q).view(BATCH_SIZE, -1, self.H, self.Dk).transpose(1, 2)
        k = self.Wk(k).view(BATCH_SIZE, -1, self.H, self.Dk).transpose(1, 2)
        v = self.Wv(v).view(BATCH_SIZE, -1, self.H, self.Dv).transpose(1, 2)
        z = self.attn_sdp(q, k, v, mask)
        z = z.transpose(1, 2).contiguous().view(BATCH_SIZE, -1, self.H * self.Dv)
        z = self.Wo(z)
        z = self.norm(x + self.dropout(z)) # residual connection and dropout
        return z

class ffn(nn.Module): # position-wise feed-forward networks
    def __init__(self, dim):
        super().__init__()
        dim_ffn = 2048

        # architecture
        self.layers = nn.Sequential(
            nn.Linear(dim, dim_ffn),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(dim_ffn, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        z = x + self.layers(x) # residual connection
        z = self.norm(z) # layer normalization
        return z

def maskset(x):
    mask = x.data.eq(PAD_IDX)
    return (mask.view(BATCH_SIZE, 1, 1, -1), x.size(1) - mask.sum(1)) # set of mask and lengths
