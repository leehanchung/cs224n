# CS224n Assignment 5
Public version of assignment 5 from CS224n Winter 2020 coures website, [`a5_publc.zip`](http://web.stanford.edu/class/cs224n/assignments/a5_public.zip). For some reason the assignment handout is the same across both versions. Not sure what's the difference.


## Unit tests:
Test cases for modules are included in `test_modules.py`. The test cases are written for CPU only. Run them by:

```python test_modules.py```


## Notes:
- Official handout `a5.pdf` does not state clearly (or reminds you) to copy functions from `assignment 4`. It's pretty easy to miss one or a few of them. The following are copied from `assignment 4`:
    - `pad_sents` in `utils.py`
    - `__init__` in `nmt_model.py`
    - `encode` in `nmt_model.py`
    - `decode` in `nmt_model.py`
    - `step` in `nmt_model.py`
- The assignment is done using Pytorch 1.4 instead of Pytorch 1.0 specified in the assignment `local_env.yml`. Here's an compatibility breaking change encountered in `cnn.py` and how it's resolved.
```
#############################
# Setting ceil_mode=True, otherwise it throws
# RuntimeError: Given input size: (256x1x12). Calculated output size: (256x1x0). Output size is too small
# References:
# https://github.com/pytorch/pytorch/issues/28625#issue-512206689
# https://github.com/pytorch/pytorch/issues/26610#issue-496710180
#############################
self.maxpool = nn.MaxPool1d(kernel_size=max_len - kernel_size + 1,
                            ceil_mode=True)
```

## Errata:
- Changed padding token to `<pad>` from `'∏'` in `vocab.py` and `sanity_check.py` to match `char_vocab_sanity_check.json`.
- Changed padding token to `<unk>` from `'Û'` in `vocab.py` and `sanity_check.py` to match `char_vocab_sanity_check.json`.
