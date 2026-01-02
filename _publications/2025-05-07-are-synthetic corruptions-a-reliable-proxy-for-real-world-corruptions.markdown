---
layout: publication 
title:  "Are Synthetic Corruptions A Reliable Proxy For Real-World Corruptions?"
date:   2025-05-07 
authors: "Shashank Agnihotri, David Schader, Nico Sharei, Mehmet Ege Kaçar, Margret Keuper"
venue: "CVPR 2025 - SynData4CV Workshop"
arxiv: "https://arxiv.org/abs/2505.04835"
categories: jekyll update
---

**TL;DR**  
We show that semantic segmentation performance under synthetic corruptions
strongly correlates with performance under real-world adverse conditions (ACDC),
supporting the use of synthetic corruptions as a scalable robustness benchmark.
Full details and additional results are available in [the paper][paper].

---

## Motivation

Deep learning-based semantic segmentation models achieve strong performance under standard conditions, yet remain sensitive to distribution shifts relative to their training data.

In outdoor driving scenarios, models frequently encounter adverse environmental conditions such as rain, fog, snow, and nighttime scenes. These conditions are often underrepresented or entirely absent during training.

Although such scenarios occur less frequently than standard daytime settings, robust semantic segmentation systems must generalize reliably to these out-of-distribution conditions.

---

## Adverse Conditions and Real-World Data
To address this challenge, the **ACDC dataset** (Adverse Conditions Dataset with Correspondences for Robust Semantic Driving Scene Perception) was introduced. It provides pixel-wise annotated driving scenes captured under night, rain, fog, and snow conditions.

ACDC is explicitly designed as an extension of the popular **Cityscapes** dataset with natural adverse conditions and has become a standard benchmark for robustness evaluation in semantic segmentation.

While highly valuable, ACDC is limited to urban environments, and many other real-world settings with adverse conditions remain underrepresented. This limitation reflects the inherent difficulty and cost of collecting semantic segmentation data under rare or extreme conditions. Specifically:

- Weather events are unpredictable and infrequent  
- Dense pixel-level annotations are expensive to obtain  
- Large-scale coverage across conditions is challenging  

---

## Synthetic Corruptions

To overcome the limitations of real-world data collection, an increasingly popular approach is to synthetically simulate distribution shifts by applying controlled corruptions to in-distribution images.

This strategy enables:

- Efficient generation of rare conditions  
- No additional annotation cost  
- High scalability for evaluation  

As a result, synthetic corruptions can be used for robustness analysis, model development, and comparative evaluation in semantic segmentation.

---

## Hypothesis

> If synthetic corruptions are faithful proxies for real-world adverse conditions,
> then semantic segmentation performance under synthetic corruptions will strongly
> correlate with performance under corresponding real-world corruptions.

Under this hypothesis, relative robustness trends observed on synthetically corrupted data should reflect those observed under natural adverse conditions.

A high correlation between performance metrics measured on real-world data (e.g., ACDC) and synthetically corrupted data would therefore validate synthetic corruptions as a reliable and scalable proxy for robustness evaluation.

---

## Experimental Setup

To test this hypothesis, we compare model performance on:

- **ACDC**, containing real-world adverse conditions  
- **Cityscapes**, augmented with corresponding synthetic corruptions  

These datasets are intentionally chosen, as ACDC is explicitly designed as an extension of Cityscapes with natural corruptions. This enables a controlled comparison between real-world and synthetic distribution shifts for semantic segmentation.

---

## Results

The figure below shows the correlation between segmentation performance on real-world and synthetic corruptions across all evaluated metrics.

![general corruptions](/assets/synthetic_corruptions/correlation_general.png)

Across all metrics, we observe high to very high correlation between real-world and synthetic corruptions, thereby supporting our hypothesis.

---
If you are interested in a more detailed discussion, additional results, and further insights,
feel free to check out the full paper:

[paper]: https://arxiv.org/abs/1712.09913

