# commonlit-readability-competition

microsoft/deberta-large:
params: 3 epochs, dropout: 0, wd: 0.1, lr: 1e-5, bs: 8, warmup: 150, scheduler: linear, random seed: 0
CV: 0.457
LB: 0.473

roberta-large:
params: 10 epochs, dropout: 0.1, wd 0.1, lr: 1e-5, bs 8, warmup: 150, scheduler: cosine, random seed: 69,
CV: 0.47
LB: 0.473

xlnet-large-cased:
params: 5 epochs, dropout: 0, wd 0.0, lr: 3e-5, bs 8, warmup: 150, scheduler: cosine, random seed: 0
CV: 0.478
LB: 0.48

funnel-transformer/large-base:
params: 10 epochs, dropout: 0.1, wd 0.1, lr: 1e-5, bs 8, warmup: 150, scheduler: cosine, random seed: 69
CV: 0.47
LB: 0.485

electra-large:
params: 4 epochs, dropout: 0, wd 0.1, lr: 1e-5, bs 8, warmup: 100, scheduler: cosine, random seed: 0
CV: 0.468
LB: 0.486
