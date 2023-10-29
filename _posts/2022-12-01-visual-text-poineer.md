---
layout: distill
title: Visual Text Poineer
description: In this blog post, we intorduce three ways of integrating visual information into LLMs.
  We also conduct some detailed analysis and comparison among them.
date: 2022-12-01
htmlwidgets: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Huang-Ru Liao
    url: "https://en.wikipedia.org/wiki/Albert_Einstein"
    affiliations:
      name: University of Texas at Austin
  - name: Muhammad Muaz
    url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
    affiliations:
      name: University of Texas at Austin
  - name: Yi-Jen Shih
    url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
    affiliations:
      name: University of Texas at Austin

# must be the exact same name as your blogpost
bibliography: 2022-12-01-visual-text-poineer.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Readme
  - name: Introduction
  - name: Methods
    subsections:
    - name: Query-based
    - name: Projection-based
    - name: Parameter-Efficient Tuning

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

## Readme

> I will delete this section when submmiting for our project.

We will put our blog content here. The example usage of the markdown can be found in `_posts/2022-12-01-distill-example.md`.

The files are organized as the following: ([SUBMISSION NAME]=`visual-text-pioneer`)
* Create a markdown file in the _posts/ directory with the format _posts/2022-12-01-visual-text-poineer.md.

* Add any static image assets will be added to assets/img/2022-12-01-visual-text-poineer/.

* Add any interactive HTML figures will be added to assets/html/2022-12-01-visual-text-poineer/.

* Put your citations into a bibtex file in assets/bibliography/2022-12-01-visual-text-poineer.bib.

## Introduction

Recent papers have shown the success of adapting LLMs to vision-language tasks with a small amount of fine-tuning required.
There are different methods to combine visual and text input. 
In this blog post, we will focus on methods that leverage a learnable interface to establish connections between images and text. 
These methods fall into three categories: query-based, projection-based, and parameter-efficient tuning approaches.
We will first introduce a representative work for each method and conduct some analysis, visualization, or interpretation on each of them(refer to $$*$$ below). 
Lastly, we will compare them on two benchmarks: ScienceQA<d-cite key="lu2022ScienceQA"></d-cite> and MME<d-cite key="fu2023mme"></d-cite> and make some conclusions about our findings.

## Methods


### Query-based
In BLIP-2<d-cite key="li2023blip2"></d-cite> , they propose a generic and compute-efficient training strategy that bootstraps from off-the-shelf frozen pre-trained vision and language models. To bridge the modality gap, they introduce a lightweight Querying Transformer(Q-Former) that employs a set of learnable queries to extract the most relevant visual features for the LLM to output the desired text. 

The Q-Former is pretrained in two stages: 
1. Vision-language representation learning stage with a frozen image encoder 

    They connect Q-Former to a frozen image encoder to jointly optimize three pre-training objectives by modeling interactions of image-text pairs.

2. Vision-to-language generative learning stage with a frozen LLM. 

    The output queries are projected and prepended to the text input as “soft visual prompts” and trained with language modeling loss. 

($$*$$) In our analysis of this paper, we aim to assess the efficacy of the acquired visual queries by visualizing the feature space alongside text tokens. Additionally, a noteworthy aspect of their method is that it extracts a fixed number of queries, irrespective of the input image resolution. This raises questions about potential information loss, a topic we plan to investigate thoroughly.

### Projection-based

In LLAVA paper<d-cite key="liu2023llava"></d-cite> , the authors presented an attempt (first of its kind) to use language-only GPT-4 model to generate multi-modal language-visual instruction following data. Apart from dataset generation, they introduced LLAVA which is an end-to-end trained multi-modal model that brings together the capabilities of both a LLM and a vision encoder model for general-purpose visual and language instruction. This way, using LLAVA model, the authors aimed to project (map) the vision features generated by the pre-trained vision encoder model to the same space as the word token embedding. The authors achieved this transformation by learning a single linear (transformation) layer. Early experimental analysis showed that the LLAVA model performance is at par with the multimodal GPT-4 model and demonstrates impressive performance on OOD data. The authors also evaluated the performance on ScienceQA<d-cite key="lu2022ScienceQA"></d-cite> dataset and showed that LLAVA achieved $$90.92 \%$$ accuracy which is quite close to SOTA model having $$91.68 \%$$ accuracy. However, the paper did not explored other ways of learning this mapping from visual features space to token embedding space. ($$*$$) We plan to explore different architectures of learning this mapping and see which design leads can lead to an efficient and robust mapping.

### Parameter-Efficient Tuning

In LLaMA-Adapter<d-cite key="zhang2023llamaadapter"></d-cite> , they incorporate visual features by adding representation from the CLIP visual encoder to the learnable adapter prompt prepended at the beginning of the text sequence. 
Zero-init Attention technique is proposed to mitigate the early stage training instability and they show good performance on image-captioning and visual question answering tasks.
($$*$$) For this type of modality bridging, we want to investigate what the adaption prompt learned and how the model attends to the adaption prompt when inference.

