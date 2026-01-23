---
layout: post 
title:  "Visualizing the Loss Landscape of LLMs"
date:   2025-12-31 21:36:48 +0100
categories: jekyll update
---

## Background

Some time ago, I came across the paper[*Visualizing the Loss Landscape of Neural Nets*][visualizing-paper] by Li et al. (2018) and found it quite interesting.

They propose a method to visualize the loss landscape of deep neural networks, which is highly relevant since supervised learning aims to find good local or global minima. These visualizations show how architectural choices affect the curvature and smoothness of the loss surface, helping practitioners stabilize training and improve convergence.

The original work focused on computer vision models, but I wondered how this would look for LLMs, given their autoregressive training differs from CV tasks. The result turned out to be quite insightful. The interactive plot below shows the loss landscape of one small layer in LLaMA-3.2 (the first linear layer inside its MLP). Feel free to explore it!

<iframe
  src="/assets/loss_landscape/loss_landscape_optimal_viridis.html"
  width="100%"
  height="500"
  style="border:none;">
</iframe>


In the following, I will talk about the methodology of the approach and provide a few insights 


## Methodology of the Paper

To understand our approach, it is helpful to first examine how Li et al. construct their loss landscape visualizations. From there, we can adapt the methodology to large language models (LLMs), identifying the necessary modifications required by their autoregressive training objective.

**Grid Definition**

The key idea is to visualize the loss function as a 3D surface over a two-dimensional slice of the model’s high-dimensional parameter space. Let $$\theta^*$$ denote the trained model parameters. Two direction vectors $$\delta$$ and $$\eta$$, having the same dimensionality as $$\theta^*$$, are sampled from a normal distribution. These vectors span a two-dimensional affine subspace within the full parameter space.

A grid over two scalar coordinates $$\alpha$$ and $$\beta$$ is then defined, typically over a range such as
$$
\alpha, \beta \in [-1, 1].
$$ This grid represents the $$x$$ and $$y$$ coordinates of our 3D-surface. 

For each grid point $$ (\alpha, \beta) $$, the model parameters are perturbed according to

$$
\theta(\alpha, \beta) = \theta^* + \alpha \delta + \beta \eta.
$$

The loss evaluated at these perturbed parameters defines the third ($$z$$-coordinate) dimension of the visualization:

$$
f(\alpha, \beta) = L\bigl(\theta^* + \alpha \delta + \beta \eta\bigr).
$$

Evaluating this loss over the entire grid yields a scalar value for each coordinate pair $$ (\alpha, \beta) $$, producing a 3D surface where the horizontal axes correspond to the direction coefficients $$\alpha$$ and $$\beta$$, and the vertical axis corresponds to the loss value.

**Filter-wise Normalization**

A crucial detail in the construction of meaningful loss landscape visualizations is the normalization of the direction vectors $$\delta$$ and $$\eta$$. Neural networks exhibit various scale invariances, for example, scaling the weights of one layer and inversely scaling the weights of a subsequent layer may leave the network’s output unchanged, while significantly altering the raw parameter values. As a result, using unnormalized random directions can lead to misleading or uninformative visualizations.

To address this issue, Li et al. propose filter-wise normalization of the direction vectors. The idea is to ensure that perturbations applied to each filter (or neuron) are proportional to the scale of the corresponding trained parameters, rather than being dominated by arbitrary differences in parameter magnitudes across layers.

Let $$\theta^*$$ denote the trained model parameters, and let $$\delta$$ be a randomly sampled direction vector with the same structure as $$\theta^*$$. We decompose both $$\theta^*$$ and $$\delta$$ into their constituent filters. For the $$j$$-th filter in the $$i$$-th layer, let $$\theta^{i,j}$$ and $$\delta{i,j}$$ denote the corresponding parameter tensors.

Filter-wise normalization rescales each filter of the direction vector according to:

$$
\delta_{i,j} \leftarrow \frac{\delta_{i,j}}{|\delta_{i,j}|} , |\theta^*_{i,j}|.
$$

An analogous normalization is applied to the second direction vector $$\eta$$.

After this normalization, each filter of the direction vectors has the same norm as the corresponding filter in the trained model. Consequently, perturbations along different layers and filters are applied at comparable relative scales, preventing any single layer from disproportionately influencing the loss variation due to parameter magnitude alone.

This normalization step is essential for producing interpretable loss landscape visualizations, as it ensures that the observed curvature and flatness reflect intrinsic geometric properties of the loss surface rather than artifacts arising from scale invariances in the network parameterization.


[visualizing-paper]: https://arxiv.org/abs/1712.09913


