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
    url: "https://www.cs.utexas.edu/~yjshih/"
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

In this section we choose LLaMA Adapter<d-cite key="zhang2023llamaadapter"></d-cite> as the representative. 
To incoporate visual information into pretrained LLMs, we can also use adatpers. 
Adapters are common techiniques for fintuning large model for downstream tasks. 
The main concept of adapters is that rather than tunning the entire model, we inject learnable light weight parameters in different layer of the large model. 
By doing so, we can steer the pretrained model to a new dowstream task. 
The advantage is that, by injecting adapters into deep layers of LLM, we can change the representation in different depth of the model without the need to update deep layers.

In LLaMA Adapter, they propose a new training method **Zero-init Gated Attention**. 
When fintuning with adapter in early training stage, it is often unstable. 
The reason is that the pretrained model has not yet learned how to utilize the newly inject adapter modules. 
To this end, in LLaMA adapter, the author intorduce a gate factor $$\alpha$$ on the adapter part of attention score before performing multiplying the values in attention layer.
Their ablation studies further substantiate the advantage of this proposed method.

The contributions of LLaMA can be summarized as follow:
* 1.2M tunable params required to achieve comparable abilities of 7B Alpaca(Fully fintuned on LLaMA 7B)
* Train less than 1 hour on 8 A100 GPU 3x faster than Alpaca
* More flexible to switch adapters than train the whole model 
* Enable to perform multimodal tasks: ScienceQA and COCO Captions
* Zero-initialized Attention can mitigate early stage unstability and can be generalized to other traditional vision and language models


{% include figure.html path="assets/img/2022-12-01-visual-text-poineer/llama_adapter_overview.png" class="img-fluid" %}
<div class="caption">
The overview of LLaMA-Adapter Architecture
</div>

#### Zero-Init Attention

To mitigate early stage disturbance of adatption prompt. The author introduce a learnable gate factor on the attention score of adaption prompts’ positions.
Specifically, let’s take a closer look at the attention layer of LLaMA.

{% include figure.html path="assets/img/2022-12-01-visual-text-poineer/llama_adapter_attention.png" class="img-fluid" %}

Suppose we have $$K$$ adaption prompts prepended in the beginning of the original sequence (length=$$M+1$$), $$C$$ indicates the hidden dimension of the model. 
Now, that's consider the last timestep for attention calculation. 
$$\mathbf{Q}_t$$ is the query vector of the last timestep,
$$\mathbf{K}_l$$ are the key vectors of the entire input sequence ($$K+M+1$$),
$$\mathbf{V}_l$$ are the value vectors of the entire input sequence ($$K+M+1$$)

To calculalte the attention score for the last timestep query to all keys $$\mathbf{S}_l$$, we simply do a dot product $$\mathbf{Q}_l$$ and $$\mathbf{K}_l$$ and normalize it by $$\sqrt{C}$$.

Notice that the upper part(first $$K$$ rows) of the attention scores $$\mathbf{S}_l$$ is affected by adaption prompt ($$[\mathbf{P}_1...\mathbf{P}_k]$$) while the rest of them aren't.
To let the model to gradually utilize the adaption prompt, the author multiply the upper part with a learnable gating factor $$\mathbf{g}_l$$. 
The softmax function is applied separately for the upper and lower parts of $$\mathbf{S}_l$$ rather than on the whole vector. The reason is that we don't want the values on the lower part(original attention) be affected by the values on the upper part(adapter prompts).

{% include figure.html path="assets/img/2022-12-01-visual-text-poineer/llama_adapter_attention1.png" class="img-fluid" %}

<div class="caption">
In their implemention code, the gating factor is different for each layer and each attention head.
</div>


Finally, the modified attention scores are then used to perform weighted sum over the entire values vector sequence to get the final hidden state for the last timestep.
To sum up, if the gating factor is 0, it is a oridinary attention calculation. The author initialized the gating factor as 0, which is also the reason why its dubbed as **zero-init** attention.

The author further conduct some ablation experiments to justify the effectiveness of **zero-init** attention.

1. The final performance on ScienceQA

    There is a performance gap about 43% on ScienceQA between w/ and w/o zero-init attention.

    | Setting | Val Acc (%) |
    | -------- | -------- |
    | Rand-Init Attention     | 40.77     | 
    | Zero-Init Attention     | 83.85     | 
    | Gain     | +43.08     |
  

2. Robustness to overfit

    As the model trains more epochs on ScienceQA, we see that it hardly shows overfitting.

    | Epoch | Train Loss |  Val Loss   | Val Acc (%)  |
    | -------- | -------- | --- | -------- |
    | 15     | 0.022     | 0.136    | 82.08    |
    | 30     | 0.004     | 0.241    | 83.85    |
    | 60     | 0.001     | 0.282    | **83.94**    |


3. Convergence

    The loss curve is also shown in the paper. Zero-init attention not only converge faster but also lower.



{% include figure.html path="assets/img/2022-12-01-visual-text-poineer/llama_adapter_loss_curve.png" class="img-fluid" %}


In addition to the analysis provided in the manuscript, we are curious how the learning factor grows thoughout the training process.
Hence, we train the LLaMA-Adapter for 5 epochs on Alplaca 52K Instruction dataset and we visulize the gating factor for each layer and each head. Notice that we only draw the absolute value of gating factor since only the magnitude matters.
We create a interatve visulization in the following.
    

<div class="l-page">
  <iframe src="{{ 'assets/html/2022-12-01-visual-text-poineer/llama_adapter_alpha_animation.html' | relative_url }}" frameborder='0' scrolling='no' height="600px" width="100%"></iframe>
</div>

As expected, the gating factor gradually grows thorughout the training process. We also observed a trend that the gating factors in upper layers tend to have a higher value. This may be resonable because the representations in upper layers are more task specific. So the role of adapters are more crucial to them.

LLaMA-Adapter incoporate visual modality by adding image embeddings(from CLIP<d-cite key="clip2021"></d-cite>) on to each learnble adapter prompts. 

($$*$$) For this type of modality bridging, we want to investigate what the adaption prompt learned and how the model attends to the adaption prompt when inference.

