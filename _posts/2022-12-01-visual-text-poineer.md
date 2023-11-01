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
  - name: Introduction
  - name: Methods
    subsections:
    - name: Query-based
    - name: Projection-based
    - name: Parameter-Efficient Tuning
  - name: Method Comparisons
    subsections:
    - name: Quantitative Analysis-ScienceQA
    - name: Qualitative Analysis-Different types of Image and Prompts 
    - name: Qualitative Analysis-Robustness

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

<!-- ## Readme

> I will delete this section when submmiting for our project.

We will put our blog content here. The example usage of the markdown can be found in `_posts/2022-12-01-distill-example.md`.

The files are organized as the following: ([SUBMISSION NAME]=`visual-text-pioneer`)
* Create a markdown file in the _posts/ directory with the format _posts/2022-12-01-visual-text-poineer.md.

* Add any static image assets will be added to assets/img/2022-12-01-visual-text-poineer/.

* Add any interactive HTML figures will be added to assets/html/2022-12-01-visual-text-poineer/.

* Put your citations into a bibtex file in assets/bibliography/2022-12-01-visual-text-poineer.bib. -->

## Introduction

<!-- Recent papers have shown the success of adapting LLMs to vision-language tasks with a small amount of fine-tuning required.
There are different methods to combine visual and text input. 
In this blog post, we will focus on methods that leverage a learnable interface to establish connections between images and text. 
These methods fall into three categories: query-based, projection-based, and parameter-efficient tuning approaches.
We will first introduce a representative work for each method and conduct some analysis, visualization, or interpretation on each of them(refer to $$*$$ below). 
Lastly, we will compare them on two benchmarks: ScienceQA<d-cite key="lu2022ScienceQA"></d-cite> and MME<d-cite key="fu2023mme"></d-cite> and make some conclusions about our findings. -->

Recent papers have shown the success of adapting LLMs to vision-language tasks with a small amount of fine-tuning required.
Various methods are proposed for bridging visual and text modalities.
Nonetheless, the comparison and analysis among them are rarely studied.
To this end, we plan to write a blog post to give a kind introduction and summary of recent approaches for vision-language adaption for LLM.
Although there are lots of related works in recent years, they basically fall into three categories: *query-based*, *projection-based*, and *parameter-efficient tuning* approaches (following definitions in<d-cite key="yin2023survey"></d-cite>).
For each of the categories, we select one representative work for our blog post.


Our blog post consists of two main parts.
In the first part, we will give a detailed introduction to each representative work individually.
We will also demonstrate some additional analysis done by ourselves and some conclusions.
For the second part, we will try to make comparisons between the three methods quantitatively and qualitatively.

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
In this section, we chose the works  LLaVA <d-cite key="liu2023llava"></d-cite> and Llava-1.5 <d-cite key="liu2023improved"></d-cite> as the representative papers. In LLaVA <d-cite key="liu2023llava"></d-cite>, the authors proposed an end-to-end multimodal model that bridges the gap between an off-the-shelf vision encoder and Large Language Model (LLM) by mapping the representations of the vision model to the same higher dimensional textual embedding space. The authors also curated a Multimodal instruction following data by utilizing language-only GPT-4 model. In Llava-1.5 <d-cite key="liu2023improved"></d-cite> the authors claim that due to the fact that LLaVA has not been pre-trained on large-scale data, as other similar approaches like query-based methods do (add citation here), the performance of Llava model is undermined. Moreover,  LLaVA model suffers from balancing between short- and long-form VQA tasks. To address these issues, the authors used prompt engineering by formatting the response formats and also, increased the complexity of projector layers.

#### Contributions of LLaVA <d-cite key="liu2023llava"></d-cite>
* The paper highlights that the amount of multimodal data (text, image) pairs have increased over time. However, the amount of instruction following multimodal data is still limited because the process is time-consuming and not well structured for human crowd source data collection. Therefore, the paper utilized language only GPT-4 model to generate three different type of data (conversation styled, detailed description, compelx reasoning) by only prompting the model with image description and the bounding box of the object present in the image.
* Secondly, the authors presented an end-to-end model using off-the-shelf image encoder (Clip Visual encoder ViT-L/14) and LLM (Llama) by mapping the representations of the vision model to the same higher dimensional textual embedding space using a linear projection layer.



### Parameter-Efficient Tuning

In this section, we choose the LLaMA Adapter<d-cite key="zhang2023llamaadapter"></d-cite> as the representative. 
To incorporate visual information into pretrained LLMs, we can also use adapters. 
Adapters are common techniques for finetuning large models for downstream tasks. 
The main concept of adapters is that rather than tuning the entire model, we inject learnable lightweight parameters in different layers of the large model. 
By doing so, we can steer the pretrained model to a new downstream task. 
The advantage is that, by injecting adapters into deep layers of LLM, we can change the representation in different depths of the model without the need to update deep layers.
The visual information is integrated into the LLaMA model by adding different scales of CLIP image encoder<d-cite key="clip2021"></d-cite>  outputs to the learnable adapter prompts.


In the LLaMA Adapter, they propose a new training method **Zero-init Gated Attention**. 
When fintuning with adapters in the early training stage, it is often unstable. 
The reason is that the pretrained model has not yet learned how to utilize the newly injected adapter modules. 
To this end, in the LLaMA adapter, the authors introduce a gating factor $$\alpha$$ on the adapter part of the attention score before multiplying the values in the attention layer.
Their ablation studies further substantiate the advantage of this proposed method.

The contributions of LLaMA can be summarized as the following:
* 1.2M tunable params required to achieve comparable abilities of 7B Alpaca(Fully fintuned on LLaMA 7B)
* Train less than 1 hour on 8 A100 GPU 3x faster than Alpaca
* More flexible to switch adapters than train the whole model 
* Enable to perform multimodal tasks: ScienceQA and COCO Captions
* Zero-initialized Attention can mitigate early-stage unstability and can be generalized to other traditional vision and language models


{% include figure.html path="assets/img/2022-12-01-visual-text-poineer/llama_adapter_overview.png" class="img-fluid" %}
<div class="caption">
The overview of LLaMA-Adapter Architecture.
</div>

#### Zero-Init Attention

To mitigate early-stage disturbance of adapter prompt. The author introduces a learnable gate factor on the attention score of adaption prompts’ positions.
Specifically, let’s take a closer look at the attention layer of LLaMA.

{% include figure.html path="assets/img/2022-12-01-visual-text-poineer/llama_adapter_attention.png" class="img-fluid" %}

Suppose we have $$K$$ adaption prompts prepended at the beginning of the original sequence (length=$$M+1$$), $$C$$ indicates the hidden dimension of the model. 
Now, let's consider the last timestep for attention calculation. 
$$\mathbf{Q}_t$$ is the query vector of the last timestep,
$$\mathbf{K}_l$$ are the key vectors of the entire input sequence ($$K+M+1$$),
$$\mathbf{V}_l$$ are the value vectors of the entire input sequence ($$K+M+1$$)

To calculate the attention score for the last timestep query to all keys $$\mathbf{S}_l$$, we simply do a dot product $$\mathbf{Q}_l$$ and $$\mathbf{K}_l$$ and normalize it by $$\sqrt{C}$$.

Notice that the upper part(first $$K$$ rows) of the attention scores $$\mathbf{S}_l$$ is affected by adapter prompt ($$[\mathbf{P}_1...\mathbf{P}_k]$$) while the rest of them aren't.
To let the model gradually utilize the adaption prompt, the author multiplies the upper part with a learnable gating factor $$\mathbf{g}_l$$. 
The softmax function is applied separately for the upper and lower parts of $$\mathbf{S}_l$$ rather than on the whole vector. The reason is that we don't want the values on the lower part(original attention) to be affected by the values on the upper part(adapter prompts).

{% include figure.html path="assets/img/2022-12-01-visual-text-poineer/llama_adapter_attention1.png" class="img-fluid" %}

<div class="caption">
In their implementation code, the gating factor is different for each layer and each attention head.
</div>


Finally, the modified attention scores are then used to perform a weighted sum over the entire values vector sequence to get the final hidden state for the last timestep.
To sum up, if the gating factor is 0, it is an ordinary attention calculation. The author initialized the gating factor as 0, which is also the reason why it's dubbed as **zero-init** attention.

The author further conducted some ablation experiments to justify the effectiveness of **zero-init** attention.

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


In addition to the analysis provided in the manuscript, we are curious how the learning factor grows throughout the training process.
Hence, we train the LLaMA-Adapter for 5 epochs on the Alplaca 52K Instruction dataset and we visualize the gating factor for each layer and each head. Notice that we only draw the absolute value of the gating factor since only the magnitude matters.
We create an interactive visulization in the following.
    

<div class="l-page">
  <iframe src="{{ 'assets/html/2022-12-01-visual-text-poineer/llama_adapter_alpha_animation.html' | relative_url }}" frameborder='0' scrolling='no' height="600px" width="100%"></iframe>
</div>

As expected, the gating factor gradually grows throughout the training process. We also observed a trend that the gating factors in the upper layers tend to have a higher value. This may be reasonable because the representations in the upper layers are more task-specific. So the role of adapters is more crucial to them.


The results for multimodal QA is in the following table. 
LLaMA-Adapter out perform previous LLM on ScienceQA.


{% include figure.html path="assets/img/2022-12-01-visual-text-poineer/llama_adapter_scienceQA.png" class="img-fluid" %}

<div class="caption">
Question Answering Accuracy (%) on ScienceQA’s [41] test set. "T" denotes the single-modal model with text-only input.
</div>


## Method Comparisons

### Quantitative Analysis-ScienceQA 

| Method |  Val Acc (%)  | Trainable Params(M) |
| -------- | -------- | -------- |
| BLIP-2(Vicuna-7B)<d-cite key="li2023blip2"></d-cite> | 77.3 | -------- |
| InstructBLIP(Vicuna-7B)<d-cite key="instructblip"></d-cite>     |  79.5   | -------- |
| LLaVA<d-cite key="liu2023llava"></d-cite>     |  90.9   | -------- |
| LLaMA-Adapter<d-cite key="zhang2023llamaadapter"></d-cite>     |  85.2   |  1.2M |

LLaVA has the best performance among all the models.
We think that it is quite expected because in the stage 2 of LLaVA, they fintune LLM + the learnable transformation layer.



### Qualitative Analysis-Different types of Image and Prompts
In this section, we perform qualitative analysis by utilizing various images and prompts to explore distinct forms of visual reasoning across three methods. Within this analysis, we categorize our visual reasoning into six distinct types, as outlined below:

1. Emotion / Atmosphere Inference:

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

<!-- Adding bed image below and adding prompt in red , and responses of different models in different colors -->
#### Images for qualitiatve analysis and Model Responses

<div class="row mt-1">
    <div class="col-sm mt-1 mt-md-0">
        {% include figure.html path="assets/img/2022-12-01-visual-text-poineer/image_prompt/bed.png" class="img-fluid rounded" %}
    </div>
    <div class="caption">
Hospital Scenario
</div>
</div>


**Prompt for picture: Describe the emotions and atmosphere depicted in the image. Explore the feelings that this setting might evoke in a person and elaborate on the ambiance of the room.**
  * **Instruct BLIP (7B) Response:** In the image, a woman is lying in a hospital bed with an intravenous (IV) drip attached to her arm. She appears to be in pain or discomfort, as she is covering her face with her hands. The room is dimly lit, which adds to the sense of unease and vulnerability. The presence of medical equipment, such as the IV drip and the bottle of pills on the nightstand, further emphasizes the seriousness of the situation. The overall atmosphere is one of sadness, fear, and concern for the woman's well-being.
  * **LLaVA-1.5 (7B) Response:** The image depicts a woman kneeling beside a hospital bed, with her hands covering her face. The emotions and atmosphere in the scene are likely to be somber, sad, or even distressed. The woman's posture and facial expression suggest that she is experiencing a range of emotions, such as grief, worry, or concern for the person lying in the bed.\n\nThe ambiance of the room is likely to be quiet, calm, and somewhat solemn. The hospital setting, with its white sheets and bed, creates a sense of tranquility and peace, but the presence of the woman's emotional distress adds a layer of complexity to the scene. The room's atmosphere might evoke feelings of empathy, support, and concern for the woman and the person in the bed, as well as a sense of vulnerability and helplessness.
  * **Llama Adapter Response:** The image depicts a woman sitting on a bed in a hospital room, with her head in her hands, looking down. This scene evokes feelings of sadness, worry, and concern. The woman's posture and facial expression suggest that she is experiencing emotional distress, possibly due to a health issue or the recent loss of a loved one. The hospital setting adds to the sense of urgency and uncertainty, as it implies that she might be in a critical or unstable situation. The dimly lit room further contributes to the atmosphere of solitude and vulnerability, as it emphasizes the woman's isolation and the importance of her emotional state. The presence of a bed and a chair in the room suggests that she might be in a recovery or recovery-related setting, which adds to the overall sense of concern and worry.


<div class="row mt-1">
 <div class="col-sm mt-1 mt-md-0">
        {% include figure.html path="assets/img/2022-12-01-visual-text-poineer/image_prompt/confusing.jpeg" class="img-fluid rounded" %}
  </div>
</div>

**Prompt for picture:  What is the unusual of this picture?**
  * **Instruct BLIP (7B) Response:** The unusual aspect of this picture is that there is a man, a woman, and two children climbing on a rock.
  * **LLaVA-1.5 (7B) Response:** The unusual aspect of this picture is that a man is squatting down to pick up a child, while another child is standing on his shoulders. This is an unconventional and potentially dangerous situation, as it is not common for a person to carry a child in such a manner. The man's posture and the children's positioning make the scene quite unique and intriguing.
  * **Llama Adapter Response:** The unusual aspect of this picture is that a man is holding three children on his back, with one child sitting on each of his shoulders. This is an unconventional way of carrying children, as it is not common to see adults carrying children in such a manner.




<div class="row mt-1">
   <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2022-12-01-visual-text-poineer/image_prompt/couple.jpeg" class="img-fluid rounded" %}
  </div> 
</div>

**Prompt for picture:  What is the relationship of these two people?**
  * **Instruct BLIP (7B) Response:** The two people in the image are a man and a woman, and they appear to be in a romantic relationship.
  * **LLaVA-1.5 (7B) Response:** The relationship between the two people in the image is not explicitly clear, but they appear to be a couple or a close pair, as they are lying down together on a blanket and looking at something.
  * **Llama Adapter Response:** The relationship of these two people is that of a couple, as they are lying together on the grass and looking at each other.


<div class="row mt-1">
   <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2022-12-01-visual-text-poineer/image_prompt/fight.jpeg" class="img-fluid rounded" %}
  </div> 
</div>

**Prompt for picture:  What might happen next?**
  * **Instruct BLIP (7B) Response:** In the image, two men are engaged in a sword-fighting match, with one man holding a sword above his head while the other man holds a sword in his hand. It is uncertain what might happen next, as the scene is ongoing and the outcome of the fight is unknown. However, it is likely that the two men will continue to engage in the sword-fighting match, exchanging blows and trying to gain an advantage over their opponent. Depending on the level of skill and experience between the two fighters, the outcome of the match could range from a draw to a decisive victory for one of the participants.
  * **LLaVA-1.5 (7B) Response:** In the image, a man is holding a sword and appears to be about to strike another man who is also holding a sword. The scene suggests that they might be engaged in a sword fight or a confrontation. It is possible that the outcome of this interaction could be one of several scenarios, such as the two men continuing their fight, one of them backing off or surrendering, or even a friendly sparring match. The exact outcome depends on the context of the situation and the intentions of the individuals involved.
  * **Llama Adapter Response:** In the image, a man is holding a sword and is about to strike another man who is holding a sword. The scene is set in a battlefield, and the two men are engaged in a sword fight. The outcome of the fight is uncertain, but it could lead to a variety of outcomes, such as one of the men being injured or killed, or the fight ending in a draw. The outcome depends on the skill level, strategy, and determination of the two combatants.
  

<div class="row mt-1">
   <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2022-12-01-visual-text-poineer/image_prompt/haunted_house.jpeg" class="img-fluid rounded" %}
  </div> 
</div>

**Prompt for picture:  Invent a detailed backstory for the abandoned old house.**
  * **Instruct BLIP (7B) Response:** The abandoned old house in the image is situated in the middle of a dense forest, surrounded by trees and shrubs. It appears to be a spooky and mysterious place, adding to the eerie atmosphere of the scene. The house has a large balcony on the second floor, which further enhances its creepy appearance. The backstory for this abandoned house could involve a tragic event that led to its abandonment, such as a natural disaster, a family tragedy, or a supernatural occurrence. Alternatively, the house could have been left uninhabited due to financial or social reasons, leaving it to decay and become a haunting presence in the forest.
  * **LLaVA-1.5 (7B) Response:** The abandoned old house, with its white exterior and black shutters, sits on a hillside surrounded by trees. It was once a grand and stately home, built in the early 20th century by a wealthy family who had a deep love for nature and the tranquility of the forest. The house was designed with a blend of architectural styles, featuring a mix of Victorian and Gothic elements.\n\nOver the years, the family's descendants continued to maintain the house, preserving its beauty and charm. However, as time passed, the family's fortunes began to decline, and they were forced to sell the house to pay off debts. The new owners, who were not as attentive to the house's upkeep, allowed it to deteriorate.\n\nThe house's current state, with its peeling paint and broken shutters, reflects the passage of time and the neglect it has suffered. The surrounding trees have grown taller, and the house has become a relic of the past, a testament to the family's once-great legacy.
  * **Llama Adapter Response:** The abandoned old house has a rich and intriguing history. It was built in the late 1800s by a wealthy family who wanted to create a grand estate in the heart of the forest. The family was known for their philanthropy and generosity, and they wanted to provide a comfortable and luxurious home for themselves and their guests.The house was designed with a mix of Victorian and Gothic architectural styles, featuring tall, narrow windows, elaborate woodwork, and a grand entrance. The family lived in the house for several decades, hosting lavish parties and entertaining guests from all walks of life. However, as time passed, the family's fortunes changed, and they were forced to sell the estate.The new owners were not as wealthy as the previous family, and they struggled to maintain the house. Over time, the house fell into disrepair, and it became a symbol of the family's decline. The house was eventually abandoned, and it has remained empty and uninhabited ever since.The house still stands tall and proud, a reminder of the family's legacy and the beauty of the bygone era. The surrounding forest has grown over
  

<div class="row mt-1">
   <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2022-12-01-visual-text-poineer/image_prompt/lemon.jpeg" class="img-fluid rounded" %}
  </div> 
</div>

**Prompt for picture:  What is this picture about? What's the feeling of the people and why they have such feelings?**
  * **Instruct BLIP (7B) Response:** The image depicts two young children, a boy and a girl, sitting on a couch while eating slices of lemon. They seem to be enjoying their snack, as they are smiling and laughing together. It is likely that they are experiencing a positive emotional state, such as happiness or contentment, due to their shared enjoyment of the lemon slices. This scene captures a moment of joy and camaraderie between the two children, emphasizing the importance of spending quality time together and savoring simple pleasures in life.
  * **LLaVA-1.5 (7B) Response:** The image features two children, a boy and a girl, sitting next to each other and eating lemon wedges. They both have a sad or unhappy expression on their faces. The reason for their sadness could be due to the taste of the lemon wedges, which are known to be sour and can be unpleasant for some people. Alternatively, their sadness might be unrelated to the lemon wedges and could be due to other factors, such as personal issues or emotions. The children's expressions and actions suggest that they are not enjoying the experience of eating the lemon wedges.
  * **Llama Adapter Response:** The image depicts a young boy and a young girl sitting on a bench, each holding a yellow banana. They appear to be enjoying their time together, possibly sharing a snack or a meal. The girl is smiling, which suggests that she is happy and content, while the boy is making a face, which could indicate that he is amused, surprised, or simply expressing his emotions. The combination of the girl's smile and the boy's facial expression creates a playful and lighthearted atmosphere, suggesting that they are having a fun and carefree moment together.

<div class="row mt-1">
   <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2022-12-01-visual-text-poineer/image_prompt/map-magnifying_glass.jpeg" class="img-fluid rounded" %}
  </div> 
</div>

**Prompt for picture:  Explain the relationship between the magnifying glass and the antique map.**
  * **Instruct BLIP (7B) Response:** The relationship between the magnifying glass and the antique map is that the magnifying glass is placed on top of the map, allowing the viewer to examine the details of the map more closely. By using the magnifying glass, the viewer can gain a better understanding of the map's features, such as geographical locations, landmarks, and other details that might be difficult to see with the naked eye. The magnifying glass serves as a tool to enhance the viewing experience of the antique map, making it more informative and engaging for the viewer.
  * **LLaVA-1.5 (7B) Response:** The magnifying glass is placed over the antique map, which is a vintage, hand-drawn map of the world. The magnifying glass is used to examine the details of the map more closely, providing a better understanding of the geographical features and information displayed on the map. The magnifying glass enhances the visual experience by allowing the viewer to see the intricate details and nuances of the antique map, which might not be easily discernible without the aid of the magnifying glass. This combination of the magnifying glass and the antique map creates a visually engaging and educational experience for the viewer.
  
  * **Llama Adapter Response:** The magnifying glass is used to enhance the view of the antique map, allowing the viewer to see more details and possibly identify specific locations or features on the map more easily. The magnifying glass can be helpful in studying the map's content, such as historical events, geographical features, or cultural aspects, as it can provide a closer and clearer view of the map's details.



### Qualitative Analysis-Robustness
To test the robustness of a multi-modal model, we provide a prompt that is completely unrelated to the image. This will evaluate the model's ability to focus on the provided textual input, ignoring the irrelevant visual context. There are two cases, with or without the hint that the image is unrelated to the picture. 

<div class="row mt-1">
   <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2022-12-01-visual-text-poineer/unrelated/rain.jpeg" class="img-fluid rounded" %}
  </div> 
</div>

**Prompt for picture:  Where is the man with the red hat?**
  * **Instruct BLIP (7B) Response:** The man with the red hat is standing under an umbrella in the rain.
  * **LLaVA-1.5 (7B) Response:** 
  
  * **Llama Adapter Response:** 

**Prompt for picture:  Where is the man with the red hat? Note that the image might be unrelated to this question.**
  * **Instruct BLIP (7B) Response:** behind umbrella
  * **LLaVA-1.5 (7B) Response:** 
  
  * **Llama Adapter Response:** 


<div class="row mt-1">
   <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2022-12-01-visual-text-poineer/unrelated/desert.jpeg" class="img-fluid rounded" %}
  </div> 
</div>

**Prompt for picture:  Why there are so many people on the ocean?**
  * **Instruct BLIP (7B) Response:** There are no people on the ocean in the image.
  * **LLaVA-1.5 (7B) Response:** 
  
  * **Llama Adapter Response:** 

**Prompt for picture:  Why there are so many people on the ocean?**
  * **Instruct BLIP (7B) Response:** no
  * **LLaVA-1.5 (7B) Response:** 
  
  * **Llama Adapter Response:** 