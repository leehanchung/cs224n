# Stanford CS 224n Natural Language Processing with Deep Learning

Self study on Stanford CS 224n, Winter 2020. Special thanks to Stanford and Professor Chris Manning for making this great resources online and free to the public.

[Lecture Videos, CS 224n, Winter 2019](https://www.youtube.com/playlist?list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z)

[Lecture slides, CS 224n, Winter 2019](./slides)

[Lecture notes, CS 224n, Winter 2019](./notes)


## Assignment 1 :heavy_check_mark:
Constructed count vectorized embeddings using co-occurance matrix and used Gensim word2vec to study predictions and language biases.

## Assignment 2 :heavy_check_mark:
Implemented and trained word2vec in Numpy.

Written: [Understanding word2vec](./a2/a2_written.pdf)

Coding: [Implementing word2vec](./a2/README.md)

![word_vectors](./a2/word_vectors.png)


## Assignment 3 :heavy_check_mark:

[Written and Coding](./a3/README.md)


## Assignment 4 :heavy_check_mark:

Coding: [Neural Machine Translation with RNN](./a4/README.md)

Left local Windows 10 machine with RTX 2080 Ti training overnight. Hit early stopping at around 11 hours. Test BLEU score 35.89.

![Train](./a4/outputs/train.png)

![Test](./a4/outputs/test.png)

Written: [Analyzing NMT Systems](./a4/a4_written.pdf)


## Assignment 5 Public :heavy_check_mark:

2020/02/16 - This network is much harder to train despite of consistently having around 1200 words/second on average vs 700 words/second on average in assignment 4. By 15 hours mark, its only at epoch 15 with loss hovering around 90 using RTX 2080 Ti. VSCode screwed up and exited at around 20 hours mark at epoch 19 with loss hovering around 80-90, test BLEU score 24.35.

2020/02/17 - Restarted run, using `batch_size=64` vs default 32. GPU memory at 10/11GB. Will see how it goes.

Coding: [Neural Machine Translation with RNN](./a5_public/README.md)

![Test](./a5_public/assets/training1.png)

Written: [Neural Machine Translation with RNN](./a5/a5_written.pdf)

## LICENSE
All slides, notes, assignments, and provided code scaffolds are owned by Stanford University.

You can use my solutions under the open CC BY-SA 3.0 license and cite it as:
```
@misc{leehanchung,
  author = {Lee, Hanchung},
  title = {CS224n Solutions},
  year = {2020},
  howpublished = {Github Repo},
  url = {https://github.com/leehanchung/cs224n}
}
```