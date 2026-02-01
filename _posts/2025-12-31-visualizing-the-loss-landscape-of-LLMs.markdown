---
layout: post 
title:  "Visualizing the Loss Landscape of Large Language Models"
date:   2026-01-28 21:36:48 +0100
categories: jekyll update
---

**TL;DR**  
I adapt the loss landscape visualization method of Li et al. (2018) to LLMs, exploring how the geometry of the loss surface varies across layers in LLaMA-3.2. The post shows visualizations of three representative layers, with a brief analysis highlighting how their landscapes differ depending on layer depth.

---

## Background

Some time ago, I came across the paper [*Visualizing the Loss Landscape of Neural Nets*][visualizing-paper] by Li et al. (2018), which I found quite interesting.

The paper proposes, well as the name suggests, a method for visualizing the loss landscape of deep neural networks. This is very useful because, at its core, machine learning is about minimizing the loss function. Being able to visualize this landscape makes it much easier to build intuition for why a model behaves the way it does and can guide architectural choices and suggest modifications that lead to better-conditioned loss surfaces. 

The original work focuses on computer vision models, but since loss functions are a universal concept, the approach can be adapted to any kind of machine learning model trained with gradient-based optimization. In my opinion, complex loss landscapes are far more insightful than the relatively simple ones that arise from classical statistical models with low-dimensional parameter spaces. I tried visualizing those as well, but they were almost always trivially bowl-shaped. For this reason, I chose to apply the method to Large Language Models. This is particularly interesting because their autoregressive training differs fundamentally from typical CV tasks. The interactive plot below shows one of the loss-landscapes resulting from this methodology. Feel free to explore it!

<iframe
  src="/assets/loss_landscape/loss_landscape_optimal_viridis.html"
  width="100%"
  height="600"
  style="border:none;">
</iframe>

In the following, I will first describe the surprisingly simple yet effective methodology behind this approach. I will then present a small experiment in which I adapt the method to LLMs and apply it to multiple layers of LLaMA-3.2, in order to investigate how the complexity of the loss landscape changes with layer depth.

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

Well, first I should maybe define what “meaningful” even means in the context of a loss landscape. Essentially, a direction is meaningful if moving along it changes the loss significantly, that is, if it is steep. One can measure this steepness using the Hessian matrix, capturing the curvature of the loss. The directions of greatest change (or steepest directions) are given by the top eigenvectors of the Hessian matrix. The result is a slice that is much more non-convex and interesting than the usual random-direction plots. These eigenvectors are approximated with power iteration and Hessian vector products, and normalized just like before so that the scale is meaningful.

The loss landscape below stems from random sampling from a normal distribution and looks relatively flat, whereas the non-convex one above uses directions from the eigenvectors and reveals much sharper and insightful curvature.

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


#### **2. Loss Function**
The standard training loss is the cross-entropy loss between the predicted token probabilities and the reference tokens:

$$ L(\theta) = - \frac{1}{N} \sum_{t=1}^{N} \log P_\theta(x_t \mid x_1, ..., x_{t-1}) $$

At each position in the sequence, the model predicts the next token, and the loss compares this prediction against the actual token in the reference sequence. 
#### **3. Reference Sequences**
Because of the autoregressive training, separate labels are not needed and each sequence serves as its own reference. For computational simplicity, I use three short prompts:

-	“Explain recursion in simple terms.”
-	“What is gradient descent?”
-	“Describe how transformers work.”

The model processes each prompt token by token. As mentioned earlier, it predicts the next token $$x_{t}$$ based on the prior tokens $$x_1, ..., x_{t-1}$$, and the cross-entropy loss is computed against the actual next token in the reference sequence. This gives a scalar loss for each prompt, which is averaged across the three prompts for evaluation.


#### **4. Experimental Setup**

For evaluation, I apply the methodology to three different layers of the Transformer [LLaMA-3.2-3B-Instruct][llama-model] to compare how the complexity and shape of the loss landscape change with layer depth. The model is chosen because it is representative of general-purpose LLMs while remaining feasible under our computational constraints. It consists of 28 layers, each with the following most relevant components:

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
The loss landscape of the first layer exhibits a largely flat outer region, with sharp, localized spikes near the origin along one principal direction.

#### **2. Middle Layer**

<iframe
  src="/assets/loss_landscape/loss_landscape_layer14_viridis.html"
  width="100%"
  height="400"
  style="border:none;">
</iframe>
The middle layer shows a comparatively smoother surface, with only a small number of localized spikes and a predominantly flat surrounding region.

#### **3. Last Layer**

<iframe
  src="/assets/loss_landscape/loss_landscape_layer27_viridis.html"
  width="100%"
  height="400"
  style="border:none;">
</iframe>
Unlike the previous layers, the final layer displays a more gradual change in loss, with less pronounced flat regions and a smoother overall slope.

---

## Analysis

Looking at the three loss landscapes side by side, a few consistent patterns emerge. All layers show large, nearly flat regions away from the center. This suggests that the trained model sits inside a fairly wide basin of the loss surface, meaning that large perturbations in many directions do not immediately harm performance. This kind of flatness is commonly associated with robustness and is in line with observations from earlier work on loss landscapes.

The more interesting behavior appears close to the original parameters. The __first layer__ shows the strongest local structure, with sharp spikes in the loss along a specific direction. This indicates that the loss is highly sensitive to small perturbations in this layer. Intuitively, this makes sense: early layers operate directly on token embeddings, and even small changes at this stage can propagate through the entire network and significantly alter downstream representations.

In the __middle layer__, the landscape becomes noticeably smoother. While there are still localized spikes, they are fewer and less pronounced. This suggests that intermediate layers are more stable to small weight perturbations. By this stage, representations are already well-formed, and neighboring layers can compensate for small changes, leading to a flatter and more robust local geometry.

The __last layer__ shows a different behavior again. Instead of sharp spikes or large flat regions, the loss changes more gradually as the parameters are perturbed. Since this layer is closest to the output logits, its weights have a more direct influence on the predicted token probabilities. As a result, perturbations tend to affect the loss in a smoother but more consistent way, rather than producing isolated regions of high curvature.

--- 

## Conclusion

Overall, these visualizations suggest that the geometry of the loss landscape evolves with depth in the Transformer. Early layers appear highly sensitive and sharply curved, middle layers are comparatively stable, and later layers exhibit smoother but more directly loss-relevant behavior. This highlights that different parts of a LLM operate under very different optimization regimes, even though they are trained with the same objective.

Congrats, you made it to the end! I sincerely appreciate your effort in reading this post and hope it was helpful :)

[visualizing-paper]: https://arxiv.org/abs/1712.09913
[llama-model]: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct

