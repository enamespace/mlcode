from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchtext.data.functional import to_map_style_dataset
import torchtext.datasets as datasets
import os
import spacy
from os.path import exists
import altair as alt
import time
from torchtext.vocab import build_vocab_from_iterator
import pandas as pd
from torch.optim.lr_scheduler import LambdaLR
import math
import copy
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad


class EncoderDecoder(nn.Module):

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator) -> None:
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt,  tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)


class Generator(nn.Module):
    def __init__(self, d_model, vocab) -> None:
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # please refer to https://github.com/harvardnlp/annotated-transformer/issues/110
        # pre-layer norm is the correct implementation
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        # must be lambda function not x
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn,  feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

        self.feed_forward = feed_forward
        self.size = size

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.self_attn(
            x, memory, memory, src_mask))
        x = self.sublayer[2](x, self.feed_forward)
        return x


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(
        attn_shape), diagonal=1).type(torch.uint8)

    return subsequent_mask == 0


def attention(q, k, v, mask=None, dropout=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, v), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        # q_m, k_m, v_m + linear
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.h = h
        self.d_k = d_model // h
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v,  mask=None):
        nbatches = q.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)

        q, k, v = [
            linear(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for linear, x in zip(self.linears, (q, k, v))
        ]

        x, self.attn = attention(q, k, v, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(
            nbatches, -1, self.h * self.d_k)

        x = self.linears[-1](x)

        del q
        del k
        del v
        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()

        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) / d_model * (-math.log(10000))
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)


def make_model(
        src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):

    model = EncoderDecoder(
        Encoder(
            EncoderLayer(
                d_model,
                MultiHeadAttention(h, d_model, dropout),
                PositionwiseFeedForward(d_model, d_ff),
                dropout
            ), N
        ),
        Decoder(
            DecoderLayer(
                d_model,
                MultiHeadAttention(h, d_model, dropout),
                MultiHeadAttention(h, d_model, dropout),
                PositionwiseFeedForward(d_model, d_ff),
                dropout
            ), N
        ),
        nn.Sequential(
            Embeddings(
                d_model,
                src_vocab
            ),
            PositionalEncoding(d_model, dropout)
        ),
        nn.Sequential(
            Embeddings(
                d_model,
                tgt_vocab
            ),
            PositionalEncoding(d_model, dropout)
        ),
        Generator(d_model, tgt_vocab)
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


def inference_test():
    test_model = make_model(11, 11, 2)
    test_model.eval()

    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)

    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)

    for i in range(9):
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    print("Example Untrained Model Prediction:", ys)


RUN_EXAMPLES = True


def show_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        return fn(*args)


def run_tests():
    for _ in range(10):
        inference_test()

# show_example(run_tests)
# tmp_model = make_model(30000, 30000, 6)
# src_vocab_size=30000, tgt_vocab_size=30000, N=6
# None
# for name, param in tmp_model.named_parameters():
#     if param.requires_grad:
#         print (name, param.data.shape)
#     else:
#         print ('no gradient necessary', name, param.data.shape)


class Batch:
    def __init__(self, src, tgt=None, pad=0):
        self.src = src
        # pad represents the unified placeholder
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            # illustrated transformer： use src + tgt to predict tgt_y
            self.tgt = tgt[:, :-1]  # first N-1 word
            self.tgt_y = tgt[:, 1:]  # last N-1 word
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        # shape like: [2, 1, 3] & [1, 3, 3] then broadcast to [2, 3, 3]
        # for every postion in one example, there is a mask to mask future words.
        tgt_mask = tgt_mask & subsequent_mask(
            tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask


class TrainState:
    step: int = 0
    accum_step: int = 0
    samples: int = 0
    tokens: int = 0


def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode='train',
    accum_iter=1,
    train_state=TrainState(),
):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0

    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.tgt,
                            batch.src_mask, batch.tgt_mask)
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                # 多次backward是什么状态
                optimizer.step()  # update parameter by grad and lr
                optimizer.zero_grad(set_to_none=True)  # clean grad
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Token / Sec: %7.1f| Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state


def rate(step, model_size, factor, warmup):
    if step == 0:
        step = 1
    return factor * model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))


def example_learning_scheudle():
    opts = [
        [512, 1, 4000],  # example 1
        [512, 1, 8000],  # example 2
        [256, 1, 4000],  # example 3
    ]

    dummy_model = torch.nn.Linear(1, 1)
    learning_rates = []

    for idx, example in enumerate(opts):
        optimizer = torch.optim.Adam(
            dummy_model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9
        )

        lr_scheduler = LambdaLR(
            optimizer=optimizer, lr_lambda=lambda x: rate(x, *example)
        )

        tmp = []

        for step in range(20000):
            tmp.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            lr_scheduler.step()

        learning_rates.append(tmp)
    learning_rates = torch.tensor(learning_rates)

    # enable altair to handle more than 5000 rows
    alt.data_transformers.disable_max_rows()

    opts_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Learning Rate": learning_rates[warmup_idx, :],
                    "model size: warmup": ["512:4000", "512:8000", "256:4000"][warmup_idx],
                    "step": range(20000)
                }
            )

            for warmup_idx in [0, 1, 2]
        ]
    )

    return (
        alt.Chart(opts_data)
        .mark_line()
        .properties(width=600)
        .encode(x="step", y="Learning Rate", color="model size\:warmup:N")
        .interactive()
    )

# example_learning_scheudle()


class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0) -> None:
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, tgt):
        assert self.size == x.size(1)

        # label smoothing explanation refer to https://zhuanlan.zhihu.com/p/116466239
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, tgt.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0

        mask_idx = torch.nonzero(tgt.data == self.padding_idx)
        if mask_idx.dim() > 0:
            true_dist.index_fill(0, mask_idx.squeeze(), 0.0)

        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


def example_label_smoothing():
    crit = LabelSmoothing(5, 0, 0.4)
    predict = torch.FloatTensor(
        [
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],

        ]
    )
    crit(x=predict.log(), tgt=torch.LongTensor([2, 1, 0, 3]))
    LS_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "target distribution": crit.true_dist[x, y].flatten(),
                    "columns": y,
                    "rows": x,
                }
            )
            for y in range(4)
            for x in range(4)
        ]
    )

    return (
        alt.Chart(LS_data)
        .mark_rect(color="Blue", opacity=1)
        .properties(height=200, width=200)
        .encode(
            alt.X("columns:O", title=None),
            alt.Y("rows:O", title=None),
            alt.Color(
                "target distribution:Q", scale=alt.Scale(scheme="viridis")
            ),
        )
        .interactive()
    )


# show_example(example_label_smoothing)


def data_gen(V, batch_size, nbatches):
    for i in range(nbatches):
        data = torch.randint(1, V, size=(batch_size, 10))
        data[:, 0] = 1
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src, tgt, 0)


class SimpleLossCompute:
    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = self.criterion(
            x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm

        return sloss.data * norm, sloss


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)

    for i in range(max_len-1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        # last word
        prob = model.generator(out[:, -1])
        _, next_word_index = torch.max(prob, dim=1)
        next_word_index = next_word_index.data[0]  # only 1 sample
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word_index)], dim=1
        )
    return ys


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None


def example_simple_model():
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)

    model = make_model(V, V, N=2)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )

    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400)
    )

    batch_size = 80

    for epoch in range(20):
        model.train()
        run_epoch(
            data_gen(V, batch_size, 20),
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer,
            lr_scheduler,
            mode='train'
        )

        model.eval()
        run_epoch(
            data_gen(V, batch_size, 5),
            model,
            SimpleLossCompute(model.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode='eval'
        )[0]
    model.eval()
    src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len)

    print(greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0))

# example_simple_model()


def load_tokenizers():
    try:
        spacy_de = spacy.load("de_core_news_sm")
    except IOError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_de, spacy_en


def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]


def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])


def build_vocabulary(spacy_de, spacy_en):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    print("Building Germen Vocabulary ...")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))

    vocab_src = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_de, index=0),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"]
    )

    print("Building English Vocabulary")

    train, val, test = datasets.Multi30k(language_pair=("de", "en"))

    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_en, index=1),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"]
    )

    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt


def load_vocab(spacy_de, spacy_en):
    if not exists("vocab.pt"):
        vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en)
        torch.save((vocab_src, vocab_tgt), "vocab.pt")
    else:
        vocab_src, vocab_tgt = torch.load("vocab.pt")
    print(
        f"Vocabulary Size: len(src)={len(vocab_src)}, len(tgt)={len(vocab_tgt)}")
    return vocab_src, vocab_tgt

spacy_de, spacy_en = load_tokenizers()
vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)
# print(vocab_src["<s>"], vocab_src["</s>"])


def collate_batch(
        batch,
        src_pipeline,
        tgt_pipeline,
        src_vocab,
        tgt_vocab,
        device,
        max_padding=128,
        pad_id=2
):
    bos_id = torch.tensor([0], device=device)  # <s>
    eos_id = torch.tensor([1], device=device)  # </s>

    src_list, tgt_list = [], []
    for (_src, _tgt) in batch:
        processed_src = torch.cat(
            [
                bos_id,
                torch.tensor(
                    src_vocab(src_pipeline(_src)),
                    dtype=torch.int64,
                    device=device
                ),
                eos_id
            ]
        )

        processed_tgt = torch.cat(
            [
                bos_id,
                torch.tensor(
                    tgt_vocab(src_pipeline(_tgt)),
                    dtype=torch.int64,
                    device=device
                ),
                eos_id
            ]
        )

        src_list.append(
            pad(processed_src, (0, max_padding-len(processed_src)), value=pad_id)
        )

        tgt_list.append(
            pad(processed_tgt, (0, max_padding-len(processed_tgt)), value=pad_id)
        )

        src = torch.stack(src_list)
        tgt = torch.stack(tgt_list)
        return src, tgt


def create_dataloaders(
        device,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=12000,
        max_padding=128,
        is_distributed=True
):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenize_de,
            tokenize_en,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()['<blank>']
        )

    train_iter, valid_iter, test_iter = datasets.Multi30k(
        language_pair=('de', 'en'))

    train_iter_map = to_map_style_dataset(
        train_iter
    )

    train_sampler = (
        DistributedSampler(train_iter_map) if is_distributed else None
    )

    valid_iter_map = to_map_style_dataset(
        valid_iter
    )

    valid_sampler = (
        DistributedSampler(valid_iter_map) if is_distributed else None
    )

    train_dataloader = DataLoader(
        train_iter_map,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn
    )

    valid_dataloader = DataLoader(
        valid_iter_map,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn
    )

    return train_dataloader, valid_dataloader

import GPUtil
def train_worker(
        gpu,
        ngpus_per_node,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        config,
        is_distributed=False
):
    torch.cuda.set_device(gpu)
    pad_idx = vocab_tgt['<blank>']
    d_model = 512
    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.cuda(gpu)
    module = model
    is_main_process = True

    criterion = LabelSmoothing(
        size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1
    )

    criterion.cuda(gpu)

    train_dataloader, valid_dataloader = create_dataloaders(
        gpu,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=config["batch_size"],
        max_padding=config["max_padding"],
        is_distributed=is_distributed
    )


    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9
    )

    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, d_model, factor=1, warmup=config["warmup"]
        )
    )

    train_state = TrainState()

    for epoch in range(config["num_epochs"]):
        model.train()
        print(f"[GPU{gpu}] Epoch {epoch} Training ====", flush=True)
        _, train_state = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=config["accum_iter"],
            train_state=train_state
        )

        GPUtil.showUtilization()

        if is_main_process:
            file_path = "%s%.2d.pt" % (config["file_prefix"], epoch)
            torch.save(module.state_dict(), file_path)
        torch.cuda.empty_cache()

        print(f"[GPU{gpu}] Epoch {epoch} Validation ====", flush=True)

        model.eval()

        sloss = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval"
        )

        print(sloss)
        torch.cuda.empty_cache()

    if is_main_process:
        file_path = "%sfinal.pt" % config["file_prefix"]
        torch.save(module.state_dict(), file_path)


def train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    train_worker(0, 1, vocab_src, vocab_tgt, spacy_de, spacy_en, config, False)


def load_trained_model():
    config = {
        "batch_size": 32,
        "distributed": False,
        "num_epochs":8,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": 72,
        "warmup": 3000,
        "file_prefix": "multi30k_model_"
    }

    model_path = "multi30k_model_final.pt"
    if not exists(model_path):
        train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config)

    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(torch.load("multi30k_model_final.pt"))

model = load_trained_model()