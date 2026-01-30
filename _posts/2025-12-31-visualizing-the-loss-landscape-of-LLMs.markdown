---
layout: post 
title:  "Visualizing the Loss Landscape of LLMs"
date:   2026-01-28 21:36:48 +0100
categories: jekyll update
---

## Background

Some time ago, I came across the paper [*Visualizing the Loss Landscape of Neural Nets*][visualizing-paper] by Li et al. (2018), which I found quite interesting.

The paper proposes, well as the name suggests, a method for visualizing the loss landscape of deep neural networks. This is very useful because, at its core, machine learning is about minimizing a loss function. Being able to visualize this landscape makes it much easier to build intuition for why a model behaves the way it does. Such visualizations can guide architectural choices and suggest modifications that lead to better-conditioned loss surfaces, and help motivate training methodologies that work particularly well in a given setting.

The original work focuses on computer vision models, but since loss functions are a universal concept, the approach can be adapted to any kind of machine learning model trained with gradient-based optimization. In my opinion, complex loss landscapes are far more insightful than the relatively simple ones that arise from classical statistical models with low-dimensional parameter spaces. For this reason, I chose to apply the method to LLMs. This is particularly interesting because their autoregressive training differs fundamentally from typical CV tasks. The interactive plot below shows one of the loss-landscapes resulting from this methodology. Feel free to explore it

<iframe
  src="/assets/loss_landscape/loss_landscape_optimal_viridis.html"
  width="100%"
  height="600"
  style="border:none;">
</iframe>

In the following, I will first describe the surprisingly simple yet effective methodology behind this approach. I will then present a small experiment in which the method is applied to multiple layers of LLaMA-3.2, to investigate whether the complexity of the loss landscape changes depending on the depth of the layer.

---

## Methodology
The methodology consists of several components. The core idea is to evaluate the loss of a trained model over a grid of parameter perturbations. In addition, a few important modifications are required to ensure that the resulting visualizations are meaningful and interpretable.

#### **1. Grid Definition**

The key idea is to visualize the loss function as a three-dimensional surface over a two-dimensional slice of the model’s high-dimensional parameter space.

Let $$ \theta^*$$ denote the trained model parameters. Two direction vectors $$\delta$$ and $$\eta$$, each having the same dimensionality as $$\theta^*$$, are sampled from a normal distribution. These vectors span a two-dimensional affine subspace within the full parameter space.

A grid over two scalar vectors $$\alpha$$ and $$\beta$$ is then defined, typically over a range such as
$$
\alpha, \beta \in [-3, 3].
$$ This grid represents the $$x$$ and $$y$$ coordinates of the 3D-surface. 

The z-value at a specific point on the grid is given by the loss achieved by the corresponding parameter perturbation:

$$
f(\alpha, \beta) = L\bigl(\theta^* + \alpha \delta + \beta \eta\bigr).
$$

By iterating over all grid points and evaluating the loss at each $$ \alpha $$ and $$ \beta $$ combination, one obtains a three-dimensional surface that represents a local slice of the loss landscape around $$\theta^* $$.


#### **2. Layer-wise normalization**

To ensure that the scale of the visualization is meaningful, the direction vectors $$\delta$$ and $$\eta$$ are typically normalized. Without normalization, the magnitude of these random directions can arbitrarily stretch or shrink the loss surface.

Following common practice, normalization can be performed layer-wise so that each direction has the same norm as the corresponding parameters:

$$\delta \leftarrow \frac{\delta}{\lVert \delta \rVert} \lVert \theta^* \rVert, \quad
\eta \leftarrow \frac{\eta}{\lVert \eta \rVert} \lVert \theta^* \rVert$$.

This ensures that variations in $$\alpha$$ and $$\beta$$ correspond to comparable relative perturbations across layers, rather than being dominated by differences in parameter scale.

#### **3. Hessian-based Directions**
Earlier I mentioned that the direction vectors $$\delta$$ and $$\eta$$ are sampled from a normal distribution. Since the visualization is heavily dependent on this choice, one may ask whether there is a more deterministic way of finding the most meaningful vectors.

Well, first I should maybe define what “meaningful” even means in the context of a loss landscape. Essentially, a direction is meaningful if moving along it changes the loss significantly, that is, if it is steep. One can measure this steepness using the Hessian, which captures the curvature of the loss. The directions of greatest change, or the steepest directions, are given by the top eigenvectors of the Hessian. The result is a slice that is much more non-convex and interesting than the usual random-direction plots. These eigenvectors are approximated with power iteration and Hessian vector products, and normalized just like before so that the scale is meaningful.

The loss landscape below stems from random sampling from a normal distribution and looks relatively flat, whereas the non-convex one above uses directions from the eigenvectors and reveals much sharper curvature.

<iframe
  src="/assets/loss_landscape/loss_landscape_random_viridis.html"
  width="100%"
  height="400"
  style="border:none;">
</iframe>

---

## Practical Implementation

In the original loss landscape paper, the authors visualized the training loss of a classification network by evaluating it on a fixed dataset and measuring the cross-entropy loss against the ground-truth class labels.

For LLMs, the setting is slightly different.


#### **1. Causal next-token prediction**
LLMs are trained to predict the next token in a sequence, given all previous tokens. Formally, given a token sequence $$ x_1, x_2, ..., x_N $$ , the model defines the probability of the sequence as:

$$ P_\theta(x_1, ..., x_N) = \prod_{t=1}^N P_\theta(x_t \mid x_1, ..., x_{t-1}) $$

where \theta represents the model weights.


#### **2. Loss Function**
The standard training loss is the cross-entropy between the predicted token probabilities and the reference tokens:

$$ L(\theta) = - \frac{1}{N} \sum_{t=1}^{N} \log P_\theta(x_t \mid x_1, ..., x_{t-1}) $$

At each position in the sequence, the model predicts the next token, and the loss compares this prediction against the actual token in the reference sequence. This is equivalent to the standard training objective of causal language models.

#### **3. Reference Sequences**
Because of the autoregressive training, separate labels: are not needed each sequence serves as its own reference. For computational simplicity, I use three short prompts:

-	“Explain recursion in simple terms.”
-	“What is gradient descent?”
-	“Describe how transformers work.”

The model processes each prompt token by token. As mentioned earlier, at each step $$x_t$$, it predicts the next token $$x_{t+1}$$ based on the prior tokens $$x_1, ..., x_{t-1}$$, and the cross-entropy loss is computed against the actual next token in the reference sequence. This gives a scalar loss for each prompt, which is averaged across the three prompts for evaluation.


#### **4. Experimental Setup**

For evaluation, I apply the methodology to three different layers of the Transformer [LLaMA-3.2-3B-Instruct][llama-model]. The model consists of 28 layers, each with the following most relevant components:

```
Transformer Layer

Input
  │
  ├─ Self-Attention
  │
  ├─ MLP
  │    ├─ up_proj (Linear)
  │    ├─ activation (SwiGLU)
  │    └─ down_proj (Linear)
  │
Output
```


For comparison, I focus on the up-projection layer of the MLP, where the hidden representation is expanded and therefore strongly affects the model’s internal geometry. By analyzing this layer, one can observe how the depth of a layer influences the loss landscape.

The three layers selected for the experiment are:
1.	**First layer** (layer 1)
2.	**Middle layer** (layer 14)
3.	**Last layer** (layer 28)

These layers provide a meaningful depth-wise comparison while avoiding the prohibitive computational cost of evaluating all 28 layers individually.

---
## Results

Based on the methodology and experimental setup, the vizualization of the loss landscapes of the three selected layers is presented below:
#### **1. First Layer**

<iframe
  src="/assets/loss_landscape/loss_landscape_layer0_viridis.html"
  width="100%"
  height="400"
  style="border:none;">
</iframe>

#### **2. Middle Layer**

<iframe
  src="/assets/loss_landscape/loss_landscape_layer14_viridis.html"
  width="100%"
  height="400"
  style="border:none;">
</iframe>

#### **3. Last Layer**

<iframe
  src="/assets/loss_landscape/loss_landscape_layer27_viridis.html"
  width="100%"
  height="400"
  style="border:none;">
</iframe>

---
[visualizing-paper]: https://arxiv.org/abs/1712.09913
[llama-model]: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct

