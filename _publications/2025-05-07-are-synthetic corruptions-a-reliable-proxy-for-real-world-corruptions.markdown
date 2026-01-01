---
layout: publication 
title:  "Are Synthetic Corruptions A Reliable Proxy For Real-World Corruptions?"
date:   2025-05-07 
authors: "Shashank Agnihotri, David Schader, Nico Sharei, Mehmet Ege Kaçar, Margret Keuper"
venue: "CVPR 2025"
arxiv: "https://arxiv.org/abs/2505.04835"
categories: jekyll update
---

![corruptions comparison](/assets/synthetic_corruptions/corruptions_comparison.png)
Although deep learning models for computer vision have demonstrated remarkable performance in real-world applications, 
they remain vulnerable to distribution shifts that deviate from their original training data.
In particular, models deployed in outdoor environments encounter conditions rarely seen during training,
such as rain, fog, snow, or nighttime scenarios. While these conditions occur less frequently than standard daylight settings, 
a robust deep learning system must be able to generalize to such out-of-distribution examples.

Unfortunately, collecting real-world training data under these rare conditions is challenging and time-consuming, 
as it requires waiting for specific weather events and capturing a sufficient number of images. 
To address this, researchers have developed the widely used ACDC dataset (“Adverse Conditions Dataset with Correspondences for 
Robust Semantic Driving Scene Perception”), which covers night, rain, fog, and snow conditions. Numerous studies validate their models using this dataset.

An alternative approach involves synthetically simulating out-of-distribution shifts by applying corruptions to in-distribution images. 
This allows researchers to generate diverse, realistic examples of rare conditions with minimal effort. 
If synthetic corruptions can reliably represent real-world corruptions, they could significantly accelerate model development and evaluation. 
In this work, we hypothesize that a high correlation between model performance on real-world corruptions and their synthetic counterparts 
indicates that synthetic corruptions effectively mirror real-world performance trends.

The image below shows that in all metrics they do show a high to very high correlation, proving our hypothesis. 

![general corruptions](/assets/synthetic_corruptions/correlation_general.png)



[paper]: https://arxiv.org/abs/1712.09913

