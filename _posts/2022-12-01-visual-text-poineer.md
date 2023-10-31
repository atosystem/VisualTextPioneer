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
<!-- In BLIP-2<d-cite key="li2023blip2"></d-cite> , they propose a generic and compute-efficient training strategy that bootstraps from off-the-shelf frozen pre-trained vision and language models. To bridge the modality gap, they introduce a lightweight Querying Transformer(Q-Former) that employs a set of learnable queries to extract the most relevant visual features for the LLM to output the desired text. 

The Q-Former is pretrained in two stages: 
1. Vision-language representation learning stage with a frozen image encoder 

    They connect Q-Former to a frozen image encoder to jointly optimize three pre-training objectives by modeling interactions of image-text pairs.

2. Vision-to-language generative learning stage with a frozen LLM. 

    The output queries are projected and prepended to the text input as “soft visual prompts” and trained with language modeling loss. 

($$*$$) In our analysis of this paper, we aim to assess the efficacy of the acquired visual queries by visualizing the feature space alongside text tokens. Additionally, a noteworthy aspect of their method is that it extracts a fixed number of queries, irrespective of the input image resolution. This raises questions about potential information loss, a topic we plan to investigate thoroughly. -->

In this section, we choose BLIP-2<d-cite key="li2023blip2"></d-cite> and its extention InstructBLIP<d-cite key="instructblip"></d-cite> as the represented papers. 
In BLIP-2, they propose a lightweight Querying Transformer (Q-Former) to bridge the gap between image and text modalities. The learnable queries learn to extract text-related features from the image in pre-trained stage.
In InstructBLIP, they formulate instruction-tuning dataset and propose a Instruction-aware Visual Feature Extraction to extend Q-Former from BLIP-2.
#### Q-Former
Q-Former is a trainable module to bridge the gap between a frozen image encoder and a frozen LLM. There are two transformer submodules (1) an image transformer and (2) a text transformer that share the same self-attention layers. A set of learnable queries is sent as input to the image transformer, interacting with frozen image features through cross-attention layers. Q-Former is pre-trained in two stages as belows:
##### Bootstrap Vision-Language Representation Learning from a Frozen Image Encoder

> Extract visual representation that is most informative of the text.
 
In the representation learning stage, Q-Former is connected to a frozen image encoder and perform pre-training using image-text pairs. The goal is to enable queries to extract visual representation that is most relevent to the text. There are three learning objectives, each one employs different attention masking strategy between queries and text.

{% include figure.html path="assets/img/2022-12-01-visual-text-poineer/BLIP2_Q-Former_stage1.png" class="img-fluid" %}
<div class="caption">
(Left) Model architecture of Q-Former and BLIP-2’s first-stage vision-language representation learning objectives. (Right) The self-attention masking strategy for each objective to control query-text interaction.
</div>

##### Image-Text Contrastive Learning (ITC) 
> Contrastive learning on the output of image and text transformer.

It learns to align image and text representations by contrasting the image-text similarity of a positive pairs against negative pairs. Specifically, it aligns the output query representation Z and output [CLS] token in text transformer. Then, compute pair-wise similarity between each query output and [CLS] and select the highest one as the image-text similarity.

##### Image-grounded Text Generation (ITG)
> Given image as condition, generate the text. 

It learns to generate the text from image features extracted by the queries. So the queries are forced to extract visual features that capture all the information about the text. Here it employs causual self-attention mask where the queries cannot attend to the text tokens while the text tokens can attend to all queries and previuos text tokens.

##### Image-Text Matching (ITM)
> Binary classification whether an image-text pair is matched.

It learns the fine-grained alignment betwen image and text. It uses a bi-directional attention where all queries and text tokens can attend to each other and the output query embeddings with multimodal information will be fed into linear classifier to predict whether the image-text pair is matched(positive) or unmatched(negative). 

##### Bootstrap Vision-to-Language Generative Learning from a Frozen LLM

In the generative pre-training stage, we connect QFormer(with the frozen image encoder attached) to a frozen LLM. The output query embeddings are projected to the same dimension as the text embeddings of the LLM. Then, the projected query embeddings are prepended to the input text embeddings as 'soft prompt'. 
There are two types of LLMs: decoder-based and encoder-decoder based as the below figure shows. 

{% include figure.html path="assets/img/2022-12-01-visual-text-poineer/BLIP2_Q-Former_stage2.png" class="img-fluid" %}
<div class="caption">
BLIP-2’s second-stage vision-to-language generative pre-training.
(Top) Bootstrapping a decoder-based LLM (e.g. OPT). (Bottom) Bootstrapping an encoder-decoder-based LLM (e.g. FlanT5).
</div>

#### InstructBLIP
In this paper, the authors conduct a systematic and comprehensive study on vision-language instruction tuning based on the pretrained BLIP-2 models. 
1. **Instruction-tuning.** They use 13 held-in datasets for instruction tuning and 13 held-out datasets for zero-shot evaluation. 
2. **Instruction-aware Visual Feature Extraction.** They proposes an instruction-aware Q-former module, which extend the Q-Former in BLIP-2 to take in the instruction text tokens as additional input. The instruction interacts with the query embeddings through self-attention layers of the Q-Former, and encourages the extraction of task-relevant image features.
3. **Balanced Data Sampling.** Due to the significant differences in the size of each dataset, mixing them uniformly could cause the model to overfit smaller datasets and underfit larger datasets. As a result, they propose to sample datasets with probabilities proportional to the square root of the numbers of training samples. 

#### Experiment

{% include figure.html path="assets/img/2022-12-01-visual-text-poineer/InstructBLIP_zero_shot.png" class="img-fluid" %}
*  Zero-shot Evaluation
    * Achieve new zero-shot SOTA results.
    * Demonstrate the effectiveness of vision-language instruction tuning.

{% include figure.html path="assets/img/2022-12-01-visual-text-poineer/BLIP2_ablation.png" class="img-fluid" %}
* **BLIP-2.** Effect of Vision-Language Representation Learning
The author shows that without representation learning, which means solely relying on vision-to-language generative learning to bridge the modality gap gives substantially lower performance on zero-shot VQA. 
 
{% include figure.html path="assets/img/2022-12-01-visual-text-poineer/InstructBLIP_ablation.png" class="img-fluid" %}
* **InstructBLIP.** Effect of Instruction-aware Visual Feature Extraction and Balanced Data Sampling strategy
    * Removing instruction awareness from visual features significantly degrades performance across all datasets.This decline is particularly notable in datasets involving spatial (e.g., ScienceQA) or temporal (e.g., iVQA) visual reasoning, where instructions guide attention to informative image regions.
    * The absence of the data balancing strategy leads to unstable and uneven training.


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

### Method Comparisons
#### Qualitative Analysis - Different types of Image and Prompts
In this section, we perform qualitative analysis by utilizing various images and prompts to explore distinct forms of visual reasoning across three methods. Within this analysis, we categorize our visual reasoning into six distinct types, as outlined below:

1. Emotion / Atmosphere Inference
    In this category, we present the model with an image carefully chosen to evoke specific emotions or atmospheric qualities. The challenge for the model is to understand the underlying emotional tone or ambiance depicted in the image. This category tests the model's ability to utilize visual cues such as lighting, colors, or even implicit reasoning to analyze the emotional atmosphere captured in the image.
2. Create a Backstory:
    In this category, we prompt the model to construct a compelling and detailed backstory for the characters, places, or objects depicted. This category tests the model's creativity and imagination to contextualize visual information and create engaging stories using the visual cues provided in the image.

4. Predict Future:
    In this category, we prompt the model to predict the future. It evaluates the model's foresight and its ability to infer future scenarios, demonstrating its understanding of causal relationships and the dynamics of the depicted scene.

5. Explain Object-Object Relationship:
    In this category, the model is prompted to reason the connections, interactions, or dependencies between different objects. This category tests the model's ability to reason the probable relationship between objects in the image.

6. Explain Human-Object Relationship:

    In this category, the model is prompted to reason the connections, interactions, or dependencies between humans and objects. This category tests the model's ability to understand gestures, expressions, and body language of human and their interactions with surrounding objects in the image.

7. Explain Human-Human Relationship:

    In this category, the model is expected to capture the nature of human's relationship — whether it is friendly, adversarial, familial, romantic, or professional. This task assesses the model's proficiency in understanding human emotions and social cues, enabling it to discern complex human relationships.

8. Confusing Image:
    In this category, the model is presented with an confusing image that might be unusual compared to commonsense. This task evaluates the model's capacity to capture the uncommon part and interpret it.

9. Implicit Relationship Inference:
    In this advanced category, the model is presented with subtle visual cues and need to infer the implicit relationships that are not immediately apparent. This evaluates the model's ability of in-depth thinking and complex visual understanding.