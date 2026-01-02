---
layout: post 
title:  "Visualizing the Loss Landscape of LLMs"
date:   2025-12-31 21:36:48 +0100
categories: jekyll update
---

## Background

Some time ago, I came across the paper [*Visualizing the Loss Landscape of Neural Nets*][visualizing-paper] by Li et al. (2018), and it truly blew my mind. The authors introduce a method to effectively visualize the loss landscape of deep learning models, which is a significant breakthrough. While the ultimate goal of supervised learning is to find parameter values that minimize the loss function—ideally reaching a global minimum or a good local minimum—visualizing the loss surface provides insight into the shape of the landscape and the dynamics of optimization. By inspecting these visualizations, one can see how hyperparameters such as network depth, number of filters, or architectural choices affect the curvature, smoothness, and connectivity of the loss landscape. This understanding enables informed adjustments that can make training more stable and help the model converge more efficiently.

Li et al. demonstrate the power of their approach by comparing the loss surfaces of ResNet-56 with and without skip connections, showing that skip connections significantly smooth the loss surface and alter the optimization dynamics.

![Loss landscape](/assets/loss_landscape/loss-landscape-paper.png)

---

## Project Formulation

As one might expect, I found this approach fascinating, but I didn’t want to simply replicate their work on vision models. Instead, my idea is to extend the methodology to NLP, and in particular to large language models (LLMs). This is especially interesting because Li et al. visualized loss landscapes for models trained on classification tasks, whereas autoregressive LLMs are trained with a fundamentally different objective. The loss calculation in LLMs differs due to their sequential, token-by-token training, which introduces unique challenges and opportunities for visualization. Exploring these differences requires some creative adaptations, and as we all know, real growth often comes from stepping off the conventional path. After all, growth is exactly the purpose of this website 🤝

---

## Methodology of the Paper

To understand our approach, it is helpful to first examine how Li et al. construct their visualizations. From there, we can adapt the methodology to LLMs, identifying the necessary modifications to handle their autoregressive training objective and implementing the visualization accordingly.


[visualizing-paper]: https://arxiv.org/abs/1712.09913


$$ \nabla_\boldsymbol{x} J(\boldsymbol{x}) $$