# CS224n Assignment 5
Public version of assignment 5 from CS224n Winter 2020 coures website, [`a5_publc.zip`](http://web.stanford.edu/class/cs224n/assignments/a5_public.zip). For some reason the assignment handout is the same across both versions. Not sure what's the difference.


## Unit tests:
Test cases for modules are included in files prefix `test`. Run them by:

```python test_highway.py```
```python test_cnn.py```


## Notes:
- Changed padding token to `<pad>` from `'∏'` in `vocab.py` and `sanity_check.py` to match `char_vocab_sanity_check.json`.
- Changed padding token to `<unk>` from `'Û'` in `vocab.py` and `sanity_check.py` to match `char_vocab_sanity_check.json`.
- Official handout `a5.pdf` does not state clearly (or reminds students) to copy functions from `assignment 4`. It's pretty easy to miss one or a few of them. The following are copied from `assignment 4`:
    - `pad_sents` in `utils.py`
    - `__init__` in `nmt_model.py`
    - `encode` in `nmt_model.py`
    - `decode` in `nmt_model.py`
    - `step` in `nmt_model.py`

- Seemingly compatbility problem. Without using `ceil_mode=True`, `sh run.sh train_local_q1` will fail to run. After changing `ceil_mode=True` in `cnn.py`, our network can pass `--no-char-decoder` with <1 loss and >99 BLEU score.
```
#############################
# Setting ceil_mode=True, otherwise it throws
# RuntimeError: Given input size: (256x1x12). Calculated output size: (256x1x0). Output size is too small
# References:
# https://github.com/pytorch/pytorch/issues/28625#issue-512206689
# https://github.com/pytorch/pytorch/issues/26610#issue-496710180
#############################
self.maxpool = nn.MaxPool1d(kernel_size=max_len - kernel_size + 1, ceil_mode=True)
```

- Modification of provided code in `nmt_model.py`:
```
###########
# 2020/02/15 This line throws an runtime error:
# RuntimeError: view size is not compatible with input tensor's size and stride 
# (at least one dimension spans across two contiguous subspaces). 
# Use .reshape(...) instead.
# target_chars = target_padded_chars[1:].view(-1, max_word_len)
target_chars = target_padded_chars[1:].reshape(-1, max_word_len)
###########
```            
- Problem 2(d) trains fine. But when running tests it throws the error.
```
epoch 200, iter 1000, cum. loss 0.18, cum. ppl 1.01 cum. examples 200   
begin validation ...
validation: iter 1000, dev. ppl 1.002426
save currently the best model to [model.bin]
save model parameters to [model.bin]
```
```
(pytorch) PS B:\user\cs224n\a5_public> python run.py decode model.bin ./en_es_data/test_tiny.es ./en_es_data/test_tiny.en outputs/test_outputs_local_q2.txt
[nltk_data] Downloading package punkt to
[nltk_data]     C:\Users\HCL\AppData\Roaming\nltk_data...
[nltk_data]   Package punkt is already up-to-date!
load test source sentences from [./en_es_data/test_tiny.es]
load test target sentences from [./en_es_data/test_tiny.en]
load model from model.bin
Decoding:   0%|                                  | 0/4 [00:00<?, ?it/s] 
  File "run.py", line 350, in <module>
    main()
  File "run.py", line 344, in main
    decode(args)
  File "run.py", line 286, in decode
    max_decoding_time_step=int(args['--max-decoding-time-step']))       
  File "run.py", line 317, in beam_search
    example_hyps = model.beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)
  File "B:\user\cs224n\a5_public\nmt_model.py", line 358, in beam_search
    y_t_embed = self.model_embeddings_target(y_tm1)
  File "B:\miniconda\envs\pytorch\lib\site-packages\torch\nn\modules\module.py", line 532, in __call__
    result = self.forward(*input, **kwargs)
  File "B:\user\cs224n\a5_public\model_embeddings.py", line 89, in forward
    X_highway = self.highway(X_conv_out)
  File "B:\miniconda\envs\pytorch\lib\site-packages\torch\nn\modules\module.py", line 532, in __call__
    result = self.forward(*input, **kwargs)
  File "B:\user\cs224n\a5_public\highway.py", line 41, in forward       
    X_proj = F.relu(self.proj(X_conv_out))
  File "B:\miniconda\envs\pytorch\lib\site-packages\torch\nn\modules\module.py", line 532, in __call__
    result = self.forward(*input, **kwargs)
  File "B:\miniconda\envs\pytorch\lib\site-packages\torch\nn\modules\linear.py", line 87, in forward
    return F.linear(input, self.weight, self.bias)
  File "B:\miniconda\envs\pytorch\lib\site-packages\torch\nn\functional.py", line 1372, in linear
    output = input.matmul(weight.t())
RuntimeError: size mismatch, m1: [1280 x 2], m2: [256 x 256] at C:\w\1\s\tmp_conda_3.6_095855\conda\conda-bld\pytorch_1579082406639\work\aten\src\TH/generic/THTensorMath.cpp:136
```