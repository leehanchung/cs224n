# CS 224n Assignment #3: Dependency Parsing

## 1 Machine Learning & Neural Networks (8 points)

### (a) (4 points)  Adam Optimizer
#### i.  (2 points)  Briefly explain (you don’t needto prove mathematically, just give an intuition) how using m stops the updates from varying as much and why this low variance may be helpful to learning, overall.

**Answer:** The momentum smooths the update rate so it helps pointing the gradients towards the long term direction and reduces contributions from gradients that change directions. Overall it would reduce oscilations and helps with faster convergence.

 #### ii. (2 points)  Adam extends the idea ofmomentumwith the trick ofadaptive  learning  ratesbykeeping track ofv, a rolling average of the magnitudes of the gradients:m←β1m+ (1−β1)∇θJminibatch(θ)v←β2v+ (1−β2)(∇θJminibatch(θ) ∇θJminibatch(θ))θ←θ−αm/√vwhereand/denote elementwise multiplication and division (sozzis elementwise squaring)andβ2is a hyperparameter between 0 and 1 (often set to 0.99).  Since Adam divides the updateby√v,  which  of  the  model  parameters  will  get  larger  updates?   Why  might  this  help  withlearning?

**Answer:** With $\sqrt{v}$, Adam normalizes the parameter updates, reducing high gradients parameters learning rate and increasing low gradients low gradients parameters learning rate through the element wise multiplication between $\alpha% and $\frac{m}{\sqrt{v}}$. The normalization helps parameters learn at a similar pace and helps with faster convergence.

### (b)  (4 points)  Dropout3is a regularization technique.  During training,  dropout randomly sets unitsin the hidden layerhto zero with probabilitypdrop(dropping different units each minibatch), andthen multiplieshby a constantγ.  We can write this ashdrop=γdhwhered∈ {0,1}Dh(Dhis the size ofh) is a mask vector where each entry is 0 with probabilitypdropand 1 with probability (1−pdrop).γis chosen such that the expected value ofhdropish:Epdrop[hdrop]i=hifor alli∈{1,...,Dh}.

#### i.  (2 points)  What mustγequal in terms ofpdrop?  Briefly justify your answer.

**Answer:** $\gamma = \frac{1}{p_drop}$. If we turn off $p%$ neurons we need to offset those 'lost' values by scaling the keep neurons to maintain the total sum of the layer.

#### ii.  (2 points)  Why should we apply dropout during training but not during evaluation?

**Answer:** We apply dropout during training to improve robustness of the model on out of training data sets, i.e., adding regularization. At evaluation time, we want to use the robust model, thus we dont have to add drop out during evaluation.


## 2. Neural Transition-Based Dependency Parsing (44 points)
### (a)  (4 points)  Go through the sequence of transitions needed for parsing the sentence“I  parsed  thissentence  correctly”.  The dependency tree for the sentence is shown below.  

**Answer:**

| Stack                                 | Buffer                                     | New dependency    | Transition |step|
|---------------------------------------|--------------------------------------------|-------------------|------------|----|
| [ROOT]                                |     [I, parsed, this, sentence, correctly] |                   | Init       |0   |
| [ROOT, I]                             |     [parsed, this, sentence, correctly]    |                   | Shift      |1   |
| [ROOT, I, parsed]                     |     [this, sentence, correctly]            |                   | Shift      |2   |
| [ROOT, parsed]                        |     [this, sentence, correctly]            | I <- parsed       | Left-Arc   |3   |
| [ROOT, parsed, this]                  |     [ sentence, correctly]                 |                   | Shift      |4   |
| [ROOT, parsed, this, sentence]        |               [correctly]                  |                   | Shift      |5   |
| [ROOT, parsed, sentence]              |               [correctly]                  |this <- sentence   | Left-Arc   |6   |
| [ROOT, parsed]                        |               [correctly]                  |parsed -> sentence | Right-Arc  |7   |
| [ROOT, parsed, correctly]             |               []                           |                   | Shift      |8   |
| [ROOT, parsed]                        |               []                           |parsed -> correctly| Right-Arc  |9   |
| [ROOT]                                |               []                           |Root -> parsed     | Right-Arc  |10  |

### (b)  (2 points)  A sentence containingnwords will be parsed in how many steps (in terms ofn)?  Brieflyexplain why.

**Answer:** 2n steps. It takes n steps to move from buffer to stack, and n steps to map n dependency.
