# CS224n Assignment 5
Public version of assignment 5 from CS224n Winter 2020 coures website, [`a5_publc.zip`](http://web.stanford.edu/class/cs224n/assignments/a5_public.zip). For some reason the assignment handout is the same across both versions. Not sure what's the difference.

### Unit tests:
Test cases for modules are included in `test_modules.py`. The test cases are written for CPU only. Run them by:

```python test_modules.py```

### Errata:
- Changed padding token to `<pad>` from `'∏'` in both `sanity_check.py` to match `char_vocab_sanity_check.json`.
- Changed padding token to `<unk>` from `'Û'` in both `sanity_check.py` to match `char_vocab_sanity_check.json`.

