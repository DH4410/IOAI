---
title: Batch Normalization and Dropout
track: nn
order: 6
estimatedTime: 40
difficulty: intermediate
---

# Batch Normalization and Dropout

You can now build and train a network. But training *deep* networks well takes two more tricks that appear in almost every modern architecture: **Batch Normalization** (which makes training faster and more stable) and **Dropout** (which fights overfitting). Both are simple to implement and enormously effective, and both behave differently during training vs inference — a subtlety that trips up beginners.

> **Big picture:** BatchNorm keeps the numbers flowing through the network at a healthy scale so training is fast and stable. Dropout randomly silences neurons during training so the network can't rely on any single one, making it generalize better.

---

## Part 1: Batch Normalization

### The internal covariate shift problem

Here's a headache with deep networks. Each layer's input is the *previous* layer's output. During training, every layer's weights are constantly changing — so the distribution of inputs each layer sees keeps *shifting* underneath it. Layer 5 spends its effort adapting to layer 4's ever-moving output instead of learning the actual task. This moving-target problem was named **internal covariate shift**.

Concretely, as training proceeds, the values feeding into a layer might drift to become huge, or tiny, or lopsided. Big values push activations like sigmoid/tanh into their saturated flat regions (vanishing gradients again), and wildly varying scales force you to use a small, cautious learning rate. Deep networks became slow and finicky to train.

**The idea of BatchNorm:** what if, at each layer, we *re-normalize* the inputs to a stable distribution (mean 0, variance 1) before passing them on? Then every layer always sees inputs on a consistent, well-behaved scale, no matter what the earlier layers are doing. That's Batch Normalization.

> Note: researchers now debate whether "internal covariate shift" is really *why* BatchNorm helps (a leading theory is that it smooths the loss landscape). But the mechanism and benefits are not in dispute, and this is the intuition it was invented with.

### The BatchNorm formula

For each feature, using the statistics of the current **mini-batch**:

**Step 1 — normalize** to zero mean and unit variance:
$$\hat{x} = \frac{x - \mu_{\text{batch}}}{\sqrt{\sigma^2_{\text{batch}} + \epsilon}}$$

*In English:* subtract the batch's mean $\mu_{\text{batch}}$ (centering the values at zero) and divide by the batch's standard deviation $\sqrt{\sigma^2 + \epsilon}$ (scaling them to have spread 1). The tiny $\epsilon$ (e.g. $10^{-5}$) just prevents division by zero when the variance is near zero. After this, the values are "standardized."

**Step 2 — scale and shift** with two *learnable* parameters $\gamma$ (gamma) and $\beta$ (beta):
$$y = \gamma\hat{x} + \beta$$

*In English:* multiply by a learnable scale $\gamma$ and add a learnable shift $\beta$. Why undo the normalization we just did? Because forcing *every* layer to have exactly mean-0, variance-1 might not be ideal — maybe this layer works best with a different scale. So we let the network *learn* the best scale and shift via $\gamma$ and $\beta$. If normalization is perfect, it can learn $\gamma=1, \beta=0$; if it prefers the original, it can learn to undo the normalization. The network gets the stability of normalization *plus* the flexibility to adjust.

The mean and variance are computed **per feature, across the examples in the batch** — hence "*batch* normalization."

### Why BatchNorm helps

- **Faster training.** Stable input distributions let you use a **much larger learning rate** without diverging — often several times faster convergence.
- **Less sensitive to initialization.** The network is more forgiving of how you set the initial weights.
- **Mild regularization.** Because each example's normalization depends on the random other examples in its batch, there's a little noise injected, which slightly reduces overfitting (a small bonus, not a replacement for dropout).
- **Combats vanishing/exploding activations.** Keeping values well-scaled keeps activations out of saturated regions.

### Training vs inference mode — the crucial difference

This is the part everyone gets wrong at least once. BatchNorm behaves **differently** during training and during inference (prediction).

- **During training:** we compute $\mu$ and $\sigma^2$ from the *current mini-batch*. Fine — the batch has many examples.
- **During inference:** we often predict on a *single* example, or we want predictions to be *deterministic*. You can't compute a meaningful "batch mean" from one example, and you don't want your prediction for an image to depend on which other images happen to be in the batch with it!

**The solution: running (moving) averages.** During training, BatchNorm keeps a running estimate of the mean and variance across all batches it has seen:

$$\mu_{\text{running}} \leftarrow (1-\rho)\,\mu_{\text{running}} + \rho\,\mu_{\text{batch}}$$
$$\sigma^2_{\text{running}} \leftarrow (1-\rho)\,\sigma^2_{\text{running}} + \rho\,\sigma^2_{\text{batch}}$$

*In English:* slowly blend each batch's statistics into a long-term average ($\rho$, the momentum, is small like 0.1). At **inference time**, we freeze these running statistics and use them instead of batch statistics — so every prediction is deterministic and works even for a single example.

> **The classic bug:** forgetting to switch modes. In PyTorch you must call `model.eval()` before validation/inference (so BatchNorm uses running stats) and `model.train()` before training. Forget `eval()` and your validation numbers will be wrong and jittery. This is one of the most common real-world mistakes — remember it.

### numpy implementation of BatchNorm

```python
import numpy as np

class BatchNorm:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.gamma = np.ones(num_features)      # learnable scale
        self.beta = np.zeros(num_features)      # learnable shift
        self.eps = eps
        self.momentum = momentum
        # running stats used at inference time
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x, training=True):
        # x has shape (batch_size, num_features)
        if training:
            mu = x.mean(axis=0)                 # per-feature mean over the batch
            var = x.var(axis=0)                 # per-feature variance over the batch
            # update the running estimates for later inference use
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            # inference: use the frozen running statistics
            mu = self.running_mean
            var = self.running_var

        x_hat = (x - mu) / np.sqrt(var + self.eps)   # normalize
        return self.gamma * x_hat + self.beta        # scale and shift

# Demo
rng = np.random.default_rng(0)
bn = BatchNorm(num_features=3)
X = rng.normal(loc=5.0, scale=2.0, size=(8, 3))   # messy inputs (mean 5, std 2)

out = bn.forward(X, training=True)
print("input  mean per feature:", np.round(X.mean(axis=0), 3))
print("output mean per feature:", np.round(out.mean(axis=0), 3))   # ~0 (gamma=1,beta=0)
print("output std  per feature:", np.round(out.std(axis=0), 3))    # ~1
```

Run it: the messy inputs (mean ≈ 5, std ≈ 2) come out with mean ≈ 0 and std ≈ 1. The layer *standardized* them. With $\gamma$ and $\beta$ starting at 1 and 0, the output is just the normalized values; as training proceeds the network tunes $\gamma,\beta$ to whatever scale works best.

---

## Part 2: Dropout

### The overfitting problem

A big network has enough capacity to **memorize** the training data — nailing every training example while failing on new data. That's **overfitting**: great training accuracy, poor test accuracy. One cause is **co-adaptation** — neurons become overly dependent on specific other neurons, forming fragile chains that work only for the training set.

**Dropout** is a wonderfully simple fix: during training, **randomly turn off (set to zero) a fraction of the neurons on each forward pass.**

### How dropout works

On every training step, for each neuron independently, flip a biased coin: with probability $p$ (the **dropout rate**), set that neuron's output to zero for this pass; otherwise keep it. The set of dropped neurons changes every step.

```
Full layer:        With dropout (p=0.5), one step:
 ● ● ● ● ● ●          ● ○ ● ○ ○ ●      (○ = dropped this step)
                     next step:
                      ○ ● ● ● ○ ○      (a different random subset)
```

**Why does randomly breaking your own network help?!** Two intuitions:

1. **No neuron can rely on any specific other neuron**, because any of them might vanish at any moment. So the network is forced to spread its knowledge across many neurons, learning **robust, redundant** features instead of fragile co-adapted chains. Redundancy generalizes better.
2. **It's like training an ensemble.** Each random dropout mask defines a slightly different "thinned" sub-network. Over training you're effectively training a huge ensemble of sub-networks that share weights, and at test time you average them — and ensembles famously generalize better than single models.

### The dropout rate p

$p$ is the probability a neuron is dropped. Common values:
- $p = 0.5$ for fully-connected hidden layers (the classic value).
- $p = 0.1$ to $0.3$ for convolutional layers or when you want gentler regularization.
- $p = 0$ means no dropout.

Higher $p$ = stronger regularization (more neurons killed) but also more disruption to training. It's a dial: turn it up if you're overfitting, down if you're underfitting.

### The scaling detail (inverted dropout)

Here's a subtlety. If we zero out half the neurons during training, the total signal reaching the next layer is, on average, only half as big. At **test time** we use *all* neurons, so the signal would suddenly be twice as big — a mismatch the network wasn't trained for.

The fix, called **inverted dropout** (what everyone uses), is to **scale up the surviving neurons during training** by dividing by the keep-probability $(1-p)$. Then the *expected* total signal stays constant, and at test time we can use all neurons with **no** scaling — dropout simply turns off. Clean and efficient.

$$\text{keep prob} = 1 - p, \qquad \text{surviving outputs} \times \frac{1}{1-p}$$

### Training vs inference — again

Like BatchNorm, dropout is **active only during training**:
- **Training:** randomly drop neurons (and scale up survivors).
- **Inference:** use **all** neurons, no dropping. (With inverted dropout, no scaling needed either — it just becomes the identity.)

Same lesson: `model.train()` turns dropout on, `model.eval()` turns it off. Forgetting `eval()` at test time makes your predictions randomly worse and non-deterministic.

### numpy implementation of Dropout

```python
import numpy as np

def dropout(x, p=0.5, training=True, rng=None):
    """Inverted dropout. p = probability of dropping a neuron."""
    if not training or p == 0.0:
        return x                       # inference: pass through unchanged
    if rng is None:
        rng = np.random.default_rng()
    keep_prob = 1.0 - p
    # mask: 1 where we keep, 0 where we drop
    mask = (rng.random(x.shape) < keep_prob).astype(float)
    # scale survivors up by 1/keep_prob so the expected value is unchanged
    return x * mask / keep_prob

# Demo
rng = np.random.default_rng(0)
x = np.ones((2, 6))                    # all ones for clarity

train_out = dropout(x, p=0.5, training=True, rng=rng)
test_out = dropout(x, p=0.5, training=False)

print("training (some zeroed, survivors doubled):")
print(train_out)
print("inference (unchanged):")
print(test_out)
print("mean during training:", train_out.mean(), " (~1.0, preserved)")
```

Run it: during training you'll see roughly half the entries zeroed and the survivors bumped to 2.0 (since $1/(1-0.5)=2$), keeping the average near 1. During inference every value stays 1.0. That preserved average is exactly why inverted dropout lets training and inference agree.

---

## BatchNorm and Dropout together

Both are regularization/stabilization tools, and both appear in real architectures — but with nuances:

- **Order in a layer:** a common pattern is `Linear → BatchNorm → ReLU → Dropout`. BatchNorm stabilizes, the activation adds non-linearity, dropout regularizes.
- **They can interact awkwardly.** BatchNorm's statistics get noisier when dropout is randomly zeroing things, so some architectures use one or the other. Modern CNNs (ResNet) lean heavily on BatchNorm and use little/no dropout; older fully-connected nets used lots of dropout.
- **Both must respect train/eval mode.** This is the unifying takeaway: any layer that behaves stochastically or tracks running statistics *must* be switched between training and inference modes. In PyTorch, `model.train()` / `model.eval()` handles both at once.

| | BatchNorm | Dropout |
|---|---|---|
| Purpose | Stabilize & speed up training | Reduce overfitting |
| Training | Uses batch mean/var; updates running stats | Randomly zeros neurons, scales survivors |
| Inference | Uses frozen running stats | Uses all neurons, no dropping |
| Learnable params | $\gamma$, $\beta$ | none |
| Key hyperparameter | momentum, $\epsilon$ | dropout rate $p$ |

---

## Why this matters for IOAI

- **They're in every serious model.** ResNets (CV track) are packed with BatchNorm; transformers use LayerNorm (a cousin) and dropout. You'll add these layers constantly, so knowing what they do prevents cargo-cult copying.
- **The train/eval bug is a score-killer.** Forgetting `model.eval()` during validation gives wrong, noisy metrics — you might discard a good model thinking it failed. Recognizing this instantly is worth real points.
- **Overfitting is the central battle.** IOAI datasets are limited; models overfit fast. Dropout (and knowing how to dial $p$) is a primary weapon, alongside data augmentation and weight decay.
- **BatchNorm lets you train faster.** With limited compute time, being able to crank up the learning rate because BatchNorm keeps things stable means you fit more experiments into the competition window.
- **Understanding running stats matters when fine-tuning.** When you fine-tune a pretrained model with frozen BatchNorm layers, knowing how running statistics work helps you avoid subtle accuracy bugs.

---

## Sort BatchNorm vs Dropout

```widget
{
  "type": "concept-sort",
  "title": "BatchNorm or Dropout? Match the Description",
  "categories": [
    { "name": "Batch Normalization", "color": "#5B5BD6" },
    { "name": "Dropout", "color": "#F97316" }
  ],
  "items": [
    { "text": "Normalizes activations to mean=0, std=1", "category": "Batch Normalization" },
    { "text": "Randomly zeros neurons during training", "category": "Dropout" },
    { "text": "Fights internal covariate shift", "category": "Batch Normalization" },
    { "text": "Forces network to not rely on single neurons", "category": "Dropout" },
    { "text": "Has learnable gamma and beta parameters", "category": "Batch Normalization" },
    { "text": "At inference, scales outputs by (1-p) implicitly", "category": "Dropout" }
  ]
}
```

---

## Practice Questions

**Quick check:** You forget to call `model.eval()` before running validation. What effect does this have on BatchNorm and Dropout?
> BatchNorm will use the current mini-batch statistics instead of the running averages — this injects noise and gives unreliable validation metrics. Dropout will also randomly zero neurons — artificially reducing capacity. Both cause validation metrics to be lower AND noisier than the true performance. Always call `model.eval()` before evaluating.

**Quick check:** After applying BatchNorm, the learnable parameters γ=1 and β=0 initially. Why are these necessary if BatchNorm already normalizes to mean=0, std=1?
> Because mean=0, std=1 might not be the optimal scale for every layer. γ and β let the network learn the optimal scale and shift for each feature. With just γ and β, the network can effectively "undo" the normalization if needed, giving it full flexibility.

**Quick check:** You're using dropout with p=0.5 at training and get 90% train accuracy. What accuracy do you expect if you forget to call model.eval() at test time?
> Significantly lower — roughly 70-80%. With p=0.5 dropout active, the network uses only 50% of neurons, halving its effective capacity. The network was calibrated to assume all neurons are present at inference (via the 1/(1-p) scaling).

---

## Summary

- Deep nets suffer from shifting input distributions (**internal covariate shift**) that slow training.
- **BatchNorm** normalizes each feature to mean 0, variance 1 using batch statistics: $\hat{x}=\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}$, then applies a **learnable** scale and shift $y=\gamma\hat{x}+\beta$.
- It enables **larger learning rates**, faster convergence, and less sensitivity to initialization.
- BatchNorm uses **batch stats during training** but **running (moving-average) stats during inference** — you must switch modes (`model.eval()`).
- **Dropout** randomly zeros a fraction $p$ of neurons during training, forcing robust, redundant features and acting like training an ensemble — which reduces **overfitting**.
- **Inverted dropout** scales survivors by $\frac{1}{1-p}$ during training so inference needs no change.
- Dropout is **off at inference**. Both layers depend critically on **train vs eval mode**.
