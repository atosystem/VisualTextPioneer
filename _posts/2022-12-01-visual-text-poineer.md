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
    url: "https://www.cs.utexas.edu/~mmuaz/"
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
  # - name: Method Comparisons
  #   subsections:
  #   - name: Quantitative Analysis-ScienceQA
  #   - name: Qualitative Analysis-Different types of Image and Prompts 
  #   - name: Qualitative Analysis-Robustness
  - name: Qualitative Analysis-Different types of Image and Prompts 
  - name: Qualitative Analysis-Robustness
  - name: Qualitative Analysis-Embedding Visualizations
  - name: Quantitative Analysis-Image-to-Text Retrieval
  - name: Quantitative Analysis-Unified Framework

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

## TL;DR

We did a blog post to introduce 3 existing type of integrating vision modality into LLM. Additionally, we also conduct both qualitative and quantitative analysis between three methods. 

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

In this blog post, we first give detailed introductions to each representative work individually.
Then we show some qualitative and quantitative analysis done by ourselves and provied some conclusion from those results.

## Method Comparisons

| Categories | Query-based  | Projection-based | Parameter-Efficient Tuning |
| Selected model | InstructBLIP<d-cite key="instructblip"></d-cite>  | LLaVA<d-cite key="liu2023llava"></d-cite> | LLaMA-Adapter<d-cite key="zhang2023llamaadapter"></d-cite>
| -------- | -------- | -------- | -------- |
| Is the extracted image features conditioned on text? | ✔    | ❌ | ❌ |
| Where are the two modalities concatenated?  | At the input of LLM    | At the input of LLM | At each adapters in multiple layers of LLM |
| Is the model pretrained on image-caption dataset?  |  ✔  | ✔ | ❌ |
| Trainable Params |  188M | 7B | 1.2M |
| ScienceQA Results (Acc %) | 79.5 | 90.9 | 85.2 |



<div class="caption">
Comparison for the three selected models. (ScienceQA results are reported from each individal paper.)
</div>

The upper two rows(Is the extracted image features conditioned on text?,Where are the two modalities concatenated?) of the table are the main concept to distinguish between the three categories.

In the following subsection, we will introduce the details about each selected model.

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

#### Bootstrap Vision-Language Representation Learning from a Frozen Image Encoder

> Extract visual representation that is most informative of the text.
 
In the representation learning stage, Q-Former is connected to a frozen image encoder and perform pre-training using image-text pairs. The goal is to enable queries to extract visual representation that is most relevent to the text. There are three learning objectives, each one employs different attention masking strategy between queries and text.

{% include figure.html path="assets/img/2022-12-01-visual-text-poineer/BLIP2_Q-Former_stage1.png" class="img-fluid" %}
<div class="caption">
Figure from <d-cite key="li2023blip2"></d-cite>. (Left) Model architecture of Q-Former and BLIP-2’s first-stage vision-language representation learning objectives. (Right) The self-attention masking strategy for each objective to control query-text interaction.
</div>

* Image-Text Contrastive Learning (ITC) 
  > Contrastive learning on the output of image and text transformer.

  It learns to align image and text representations by contrasting the image-text similarity of a positive pairs against negative pairs. Specifically, it aligns the output query representation Z and output [CLS] token in text transformer. Then, compute pair-wise similarity between each query output and [CLS] and select the highest one as the image-text similarity.

* Image-grounded Text Generation (ITG)
  > Given image as condition, generate the text. 

  It learns to generate the text from image features extracted by the queries. So the queries are forced to extract visual features that capture all the information about the text. Here it employs causual self-attention mask where the queries cannot attend to the text tokens while the text tokens can attend to all queries and previuos text tokens.

* Image-Text Matching (ITM)
  > Binary classification whether an image-text pair is matched.

  It learns the fine-grained alignment betwen image and text. It uses a bi-directional attention where all queries and text tokens can attend to each other and the output query embeddings with multimodal information will be fed into linear classifier to predict whether the image-text pair is matched(positive) or unmatched(negative). 

#### Bootstrap Vision-to-Language Generative Learning from a Frozen LLM

In the generative pre-training stage, we connect QFormer(with the frozen image encoder attached) to a frozen LLM. The output query embeddings are projected to the same dimension as the text embeddings of the LLM. Then, the projected query embeddings are prepended to the input text embeddings as 'soft prompt'. 
There are two types of LLMs: decoder-based and encoder-decoder based as the below figure shows. 

{% include figure.html path="assets/img/2022-12-01-visual-text-poineer/BLIP2_Q-Former_stage2.png" class="img-fluid" %}
<div class="caption">
Figure from <d-cite key="li2023blip2"></d-cite>. BLIP-2’s second-stage vision-to-language generative pre-training.
(Top) Bootstrapping a decoder-based LLM (e.g. OPT). (Bottom) Bootstrapping an encoder-decoder-based LLM (e.g. FlanT5).
</div>

#### InstructBLIP
In this paper, the authors conduct a systematic and comprehensive study on vision-language instruction tuning based on the pretrained BLIP-2 models. 
1. **Instruction-tuning.** They use 13 held-in datasets for instruction tuning and 13 held-out datasets for zero-shot evaluation. 
2. **Instruction-aware Visual Feature Extraction.** They proposes an instruction-aware Q-former module, which extend the Q-Former in BLIP-2 to take in the instruction text tokens as additional input. The instruction interacts with the query embeddings through self-attention layers of the Q-Former, and encourages the extraction of task-relevant image features.
3. **Balanced Data Sampling.** Due to the significant differences in the size of each dataset, mixing them uniformly could cause the model to overfit smaller datasets and underfit larger datasets. As a result, they propose to sample datasets with probabilities proportional to the square root of the numbers of training samples. 

#### Experiment

{% include figure.html path="assets/img/2022-12-01-visual-text-poineer/InstructBLIP_zero_shot.png" class="img-fluid" %}
<div class="caption">
    Table from <d-cite key="li2023blip2"></d-cite>.
</div>
*  Zero-shot Evaluation
    * Achieve new zero-shot SOTA results.
    * Demonstrate the effectiveness of vision-language instruction tuning.

{% include figure.html path="assets/img/2022-12-01-visual-text-poineer/BLIP2_ablation.png" class="img-fluid" %}
<div class="caption">
    Table from <d-cite key="li2023blip2"></d-cite>.
</div>

* **BLIP-2.** Effect of Vision-Language Representation Learning
The author shows that without representation learning, which means solely relying on vision-to-language generative learning to bridge the modality gap gives substantially lower performance on zero-shot VQA. 
 
{% include figure.html path="assets/img/2022-12-01-visual-text-poineer/InstructBLIP_ablation.png" class="img-fluid" %}
<div class="caption">
    Table from <d-cite key="instructblip"></d-cite>.
</div>
* **InstructBLIP.** Effect of Instruction-aware Visual Feature Extraction and Balanced Data Sampling strategy
    * Removing instruction awareness from visual features significantly degrades performance across all datasets.This decline is particularly notable in datasets involving spatial (e.g., ScienceQA) or temporal (e.g., iVQA) visual reasoning, where instructions guide attention to informative image regions.
    * The absence of the data balancing strategy leads to unstable and uneven training.


### Projection-based
In this section, we chose the works  LLaVA <d-cite key="liu2023llava"></d-cite> and Llava-1.5 <d-cite key="liu2023improved"></d-cite> as the representative papers. In LLaVA <d-cite key="liu2023llava"></d-cite>, the authors proposed an end-to-end multimodal model that bridges the gap between an off-the-shelf vision encoder and Large Language Model (LLM) by mapping the representations of the vision model to the same higher dimensional textual embedding space. The authors also curated a Multimodal instruction following data by utilizing language-only GPT-4 model. In Llava-1.5 <d-cite key="liu2023improved"></d-cite> the authors claim that due to the fact that LLaVA has not been pre-trained on large-scale data, as other similar approaches like query-based methods do (add citation here), the performance of Llava model is undermined. Moreover,  LLaVA model suffers from balancing between short- and long-form VQA tasks. To address these issues, the authors used prompt engineering by formatting the response formats and also, increased the complexity of projector layers.

#### Contributions of LLaVA <d-cite key="liu2023llava"></d-cite>
* The paper highlights that the amount of multimodal data (text, image) pairs have increased over time. However, the amount of instruction following multimodal data is still limited because the process is time-consuming and not well structured for human crowd source data collection. Therefore, the paper utilized language only GPT-4 model to generate three different type of data (conversation styled, detailed description, compelx reasoning) by only prompting the model with image description and the bounding box of the object present in the image.
* Secondly, the authors presented an end-to-end model using off-the-shelf image encoder (Clip Visual encoder ViT-L/14) and LLM (Llama) by mapping the representations of the vision model to the same higher dimensional textual embedding space using a linear projection layer.

#### GPT-4 Assisted Visual Instruction Data Generation
The authors utilized language GPT-4 model to generate visual instruction tuned data for (image, text) pairs by prompting the GPT-4 model with features of image such as description of scenario and objects in the image and bounding box data of the objects in the image. Then, the model is seeded with a couple of manually curated data. Three type of instruction-following data is collected:

* _Conversation_
  > A conversation between assistant and a person asking questions about a given photo. The answers are in a tone as if the assistant is seeing the image and answering the questions. Questions asked include object types, counting the objects, object actions, relative positions between objects.
* _Detailed Description_
  > A list of questions is curated that prompt GPT-4 to answer in a more detail way about the image. The questions include : 
  > * Describe the following image in detail
  > * Provide a detialed description of the given image
  > * Analyze the image in a comprehensive and detailed manner
* _Complex Reasoning_
 > This type of questions require answers that require a step-by-step reasoning process by following rigorous logic.

#### LLaVA Model Architecture
{% include figure.html path="assets/img/2022-12-01-visual-text-poineer/llava_model_architecture.png" class="img-fluid" %}
 <div class="caption">
    Model image taken from <d-cite key="liu2023llava"></d-cite>
    </div>

As LLaVA model relies on off-the-shelf pretrained vision and language models and these models maps their input to a separate high dimensional space. However, to effectively leverage their capabilites (i.e., jointly use the information captured by vision and language embeddings), the embeddings have to be mapped closer in the same higher dimensional space. For an input image  $$X_v$$, a pre-trained CLIP visual encoder ViT/L-14 is used to extract the visual features $$Z_v = g(X_v)$$. The model uses grid features before and after the last Transformer layer for experiments. To convert, the extracted visual features and to provide conditioning to the text, the features are mapped to the space of text token embeddings via a single linear layer. Specifically, $$ Z_v $$ is converted to $$ H_v $$ via the learnable matrix $$ W $$ i.e., $$ H_v = W * Z_v$$. 

#### Training of LLaVA Model

The data to instruct tune the model is generated as: For each image $$ X_v $$, multi-turn conversation data is generated as a sequence $$ ( X_q^1, X_a^1 , \dots, X_q^T, X_a^T )$$ where $$ T $$ represents the total number of turns. At iteration $$t$$, the model is given $$X_{instruct}^t$$ which is defined as : <br>

$$  
X_{instruct}^t = \begin{cases} 
          \text{Random choice } [X_v, X_q^1] \text{ or } [X_q^1, X_v] & t=1 \\
          X_q^t & t > 1 \\
       \end{cases}
  $$  
  <br>
Instruction tuning is performed on LLM on the prediction tokens via the original auto-regressive training objective. Specifically, model is fine tuned for the loss signal that is generated on predicted **answer tokens**. The model training is done in two steps:

* **Feature alignment Phase**: In order to instruction tune the model, the first thing is to be able to learn the mapping between image encoder features to text embeddings. This is done by using filtered data from CC3M dataset to get $595$ K image-text pairs. These pairs can be converted to a single turn instruction-following data by associating a _**conversation**_ style question. Both the visual encoder and LLM weights are frozen, and the likelihood of answer is maximized given the image and instruction (question) by training the linear layer parameter i.e., $\theta = W$. **This step essentially then aligns the image features with pre-trained LLM word embeddings.**
* **Fine-tuning End-to-End**: This step only keeps the weights of visual encoder frozen, and continue to update both the projection layer weights and the pre-trained LLM weights. Two specific use case scenarios are considered:
  * _Multimodal Chatbot_: $158$ K unique language-image conversation styled collected multi-turn data is used for training. Uniform sampling is done from this dataset.
  * _Science QA_: Science QA dataset is used to train the model for generating the reasoning process in natural language and then selecting the answer from the multiple choices.

#### Experiments
{% include figure.html path="assets/img/2022-12-01-visual-text-poineer/llava_base_exp.png" class="img-fluid" %}
 <div class="caption">
    Table from <d-cite key="liu2023llava"></d-cite>
    </div>

The experimental analysis showed that LLaVA achieved SOTA performance on mean accuracy on ScienceQA compared to other methods such as Chain-Of-Thought (COT) and LLaMA-Adapter.

#### LLaVA-1.5 <d-cite key="liu2023improved"></d-cite> 

**Contributions of LLaVA-1.5**

The base architecture of LLaVA is kept intact but the following modifications are made:
* CLIP-ViT-L-336px with an MLP projector is used instead of previously used visual encoder and a single linear projection layer.
* Academic-task-oriented VQA data with prompt engineering 
  
The authors claim that LLaVA model was falling short on academic benchmakrs that typically require short-form answers. They attribute this to the fact that LLaVA has not bben pretrained on large-scale data as other approaches do. The following image studies the scaling effect of data, model and image resolution on a selection of three datasets given in the following table.
{% include figure.html path="assets/img/2022-12-01-visual-text-poineer/llava_improved_scaling.png" class="img-fluid" %}
 <div class="caption">
    Table from <d-cite key="liu2023improved"></d-cite>
    </div>


Moreover, the authors proposed that to control the length of prompted answer of LLaVA model, they explicitly state that information in the prompt during its fine-training stage which can help the model to learn to control the length of output response. Apart from this, the model is fine-tuned on academic-task-oriented VQA datasets such as open-knowledge VQA, OCR VQA, and region level VQA. Moreover, a two linear layer MLP architecture is used for projecting visual features to text token embedding space. Furthermore, image size is scaled up to $336$ px and LLM model size is scaled to $13$ B.

#### Experiments
{% include figure.html path="assets/img/2022-12-01-visual-text-poineer/llava_improved_results.png" class="img-fluid" %}
 <div class="caption">
    Table from <d-cite key="liu2023improved"></d-cite>
    </div>

Based on the previous additions to the model, the performance on a total of $12$ benchmarks of academic VQA benchmarks specifically proposed for instruction following LMMs showed that LLaVA-1.5 achieved SOTA performance across $11$ out of $12$ benchmarks.

#### Limitations
* LLaVA-1.5 is not capable of processing multiple images due to lack of such instruction-following data and limit of context lengths.
* LLaVA-1.5 problem-solving capabilities can still be limited in certain domains
* LLaVA-1.5 can suffer from hallucinations and occasionally disseminating misinformation, so, it should be used with caution in critical applications like **medical applications**

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
To this end, in the LLaMA adapter, the authors introduce a gating factor $$\mathbf{g}_l$$ on the adapter part of the attention score before multiplying the values in the attention layer.
Their ablation studies further substantiate the advantage of this proposed method.

The contributions of LLaMA can be summarized as the following:
* 1.2M tunable params required to achieve comparable abilities of 7B Alpaca(Fully fintuned on LLaMA 7B)
* Train less than 1 hour on 8 A100 GPU 3x faster than Alpaca
* More flexible to switch adapters than train the whole model 
* Enable to perform multimodal tasks: ScienceQA and COCO Captions
* Zero-initialized Attention can mitigate early-stage unstability and can be generalized to other traditional vision and language models


{% include figure.html path="assets/img/2022-12-01-visual-text-poineer/llama_adapter_overview.png" class="img-fluid" %}
<div class="caption">
The overview of LLaMA-Adapter Architecture. (Figure from <d-cite key="zhang2023llamaadapter"></d-cite>)
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

    <div class="caption">
    Table from <d-cite key="zhang2023llamaadapter"></d-cite>
    </div>

  

2. Robustness to overfit

    As the model trains more epochs on ScienceQA, we see that it hardly shows overfitting.

    | Epoch | Train Loss |  Val Loss   | Val Acc (%)  |
    | -------- | -------- | --- | -------- |
    | 15     | 0.022     | 0.136    | 82.08    |
    | 30     | 0.004     | 0.241    | 83.85    |
    | 60     | 0.001     | 0.282    | **83.94**    |

     <div class="caption">
    Table from <d-cite key="zhang2023llamaadapter"></d-cite>
    </div>


3. Convergence

    The loss curve is also shown in the paper. Zero-init attention not only converge faster but also lower.



{% include figure.html path="assets/img/2022-12-01-visual-text-poineer/llama_adapter_loss_curve.png" class="img-fluid" %}
 <div class="caption">
    Training curves w/ and w/o zero-init attention. (Figure from <d-cite key="zhang2023llamaadapter"></d-cite>)
</div>


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
Question Answering Accuracy (%) on ScienceQA’s test set. "T" denotes the single-modal model with text-only input. (Table from <d-cite key="zhang2023llamaadapter"></d-cite>)
</div>


<!-- ## Method Comparisons -->

<!-- ### Quantitative Analysis-ScienceQA 

| Method |  Val Acc (%)  | Trainable Params(M) |
| -------- | -------- | -------- |
| BLIP-2(Vicuna-7B)<d-cite key="li2023blip2"></d-cite> | 77.3 | 188M |
| InstructBLIP(Vicuna-7B)<d-cite key="instructblip"></d-cite>     |  79.5   | 188M |
| LLaVA<d-cite key="liu2023llava"></d-cite>     |  90.9   | -------- |
| LLaMA-Adapter<d-cite key="zhang2023llamaadapter"></d-cite>     |  85.2   |  1.2M |

LLaVA has the best performance among all the models.
We think that it is quite expected because in the stage 2 of LLaVA, they fintune LLM + the learnable transformation layer.

 -->

## Qualitative Analysis-Different types of Image and Prompts
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
        {% include figure.html path="assets/img/2022-12-01-visual-text-poineer/image_prompt/bed.jpg" class="img-fluid rounded" %}
    </div>
</div>


**<mark>Emotion/Atmosphere Inference</mark>**<br>
**Prompt for picture: Describe the emotions and atmosphere depicted in the image. Explore the feelings that this setting might evoke in a person and elaborate on the ambiance of the room.**
  * **Instruct BLIP (7B) Response:** In the image, <span style="color:red">a woman is lying in a hospital bed with an intravenous (IV) drip attached to her arm. She appears to be in pain or discomfort, as she is covering her face with her hands</span>. The room is dimly lit, which adds to the sense of unease and vulnerability. The presence of medical equipment, such as the IV drip and the bottle of pills on the nightstand, further emphasizes the seriousness of the situation. The overall atmosphere is one of sadness, fear, and concern for the woman's well-being.
  * **LLaVA-1.5 (7B) Response:** The image depicts a woman kneeling beside a hospital bed, with her hands covering her face. <span style="color:green">The emotions and atmosphere in the scene are likely to be somber, sad, or even distressed. The woman's posture and facial expression suggest that she is experiencing a range of emotions, such as grief, worry, or concern for the person lying in the bed</span>.\n\nThe ambiance of the room is likely to be quiet, calm, and somewhat solemn. The hospital setting, with its white sheets and bed, creates a sense of tranquility and peace, but the presence of the woman's emotional distress adds a layer of complexity to the scene. The room's atmosphere might evoke feelings of empathy, support, and concern for the woman and the person in the bed, as well as a sense of vulnerability and helplessness.
  * **Llama Adapter Response:** The image depicts a woman sitting on a bed in a hospital room, with her head in her hands, looking down.<span style="color:green"> This scene evokes feelings of sadness, worry, and concern. The woman's posture and facial expression suggest that she is experiencing emotional distress, possibly due to a health issue or the recent loss of a loved one</span>. The hospital setting adds to the sense of urgency and uncertainty, as it implies that she might be in a critical or unstable situation. The dimly lit room further contributes to the atmosphere of solitude and vulnerability, as it emphasizes the woman's isolation and the importance of her emotional state. The presence of a bed and a chair in the room suggests that she might be in a recovery or recovery-related setting, which adds to the overall sense of concern and worry.


<div class="row mt-1">
 <div class="col-sm mt-1 mt-md-0">
        {% include figure.html path="assets/img/2022-12-01-visual-text-poineer/image_prompt/confusing.jpeg" class="img-fluid rounded" %}
  </div>
</div>

**<mark>Confusing Image</mark>**<br>
**Prompt for picture:  What is the unusual of this picture?**
  * **Instruct BLIP (7B) Response:** <span style="color:red">The unusual aspect of this picture is that there is a man, a woman, and two children climbing on a rock.</span>
  * **LLaVA-1.5 (7B) Response:** <span style="color:green">The unusual aspect of this picture is that</span> <span style="color:red"> a man</span> <span style="color:green">is squatting down to pick up a child, while another child is standing on his shoulders</span>. This is an unconventional and potentially dangerous situation, as it is not common for a person to carry a child in such a manner. The man's posture and the children's positioning make the scene quite unique and intriguing.
  * **Llama Adapter Response:** <span style="color:red"> The unusual aspect of this picture is that a man is holding three children on his back, with one child sitting on each of his shoulders.</span>This is an unconventional way of carrying children, as it is not common to see adults carrying children in such a manner.




<div class="row mt-1">
   <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2022-12-01-visual-text-poineer/image_prompt/couple.jpeg" class="img-fluid rounded" %}
  </div> 
</div>

**<mark>Explain Human-Human Relationship</mark>**<br>
**Prompt for picture:  What is the relationship of these two people?**
  * **Instruct BLIP (7B) Response:** <span style="color:red">The two people in the image are a man and a woman, and they appear to be in a romantic relationship.</span>
  * **LLaVA-1.5 (7B) Response:** The relationship between the two people in the image is not explicitly clear, <span style="color:green">but they appear to be a couple or a close pair,</span> as they are lying down together on a blanket and looking at something.
  * **Llama Adapter Response:** <span style="color:green">The relationship of these two people is that of a couple</span>, as they are lying together on the grass and looking at each other.


<div class="row mt-1">
   <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2022-12-01-visual-text-poineer/image_prompt/fight.jpeg" class="img-fluid rounded" %}
  </div> 
</div>

**<mark>Predict Future</mark>**<br>
**Prompt for picture:  What might happen next?**
  * **Instruct BLIP (7B) Response:** In the image, two men are engaged in a sword-fighting match, with one man holding a sword above his head while the other man holds a sword in his hand.<span style="color:green"> It is uncertain what might happen next, as the scene is ongoing and the outcome of the fight is unknown. However, it is likely that the two men will continue to engage in the sword-fighting match, exchanging blows and trying to gain an advantage over their opponent.</span> Depending on the level of skill and experience between the two fighters, the outcome of the match could range from a draw to a decisive victory for one of the participants.
  * **LLaVA-1.5 (7B) Response:** In the image, a man is holding a sword and appears to be about to strike another man who is also holding a sword. The scene suggests that they might be engaged in a sword fight or a confrontation. <span style="color:green">It is possible that the outcome of this interaction could be one of several scenarios, such as the two men continuing their fight, one of them backing off or surrendering, or even a friendly sparring match. The exact outcome depends on the context of the situation and the intentions of the individuals involved.</span>
  * **Llama Adapter Response:** In the image, a man is holding a sword and is about to strike another man who is holding a sword. The scene is set in a battlefield, and the two men are engaged in a sword fight. <span style="color:green">The outcome of the fight is uncertain, but it could lead to a variety of outcomes, such as one of the men being injured or killed, or the fight ending in a draw. The outcome depends on the skill level, strategy, and determination of the two combatants.</span>
  

<div class="row mt-1">
   <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2022-12-01-visual-text-poineer/image_prompt/haunted_house.jpeg" class="img-fluid rounded" %}
  </div> 
</div>

**<mark>Create a Backstory</mark>**<br>
**Prompt for picture:  Invent a detailed backstory for the abandoned old house.**
  * **Instruct BLIP (7B) Response:** The abandoned old house in the image is situated in the middle of a dense forest, surrounded by trees and shrubs. It appears to be a spooky and mysterious place, adding to the eerie atmosphere of the scene. The house has a large balcony on the second floor, which further enhances its creepy appearance. The backstory for this abandoned house could involve a tragic event that led to its abandonment, such as a natural disaster, a family tragedy, or a supernatural occurrence. Alternatively, the house could have been left uninhabited due to financial or social reasons, leaving it to decay and become a haunting presence in the forest.
  * **LLaVA-1.5 (7B) Response:** The abandoned old house, with its white exterior and black shutters, sits on a hillside surrounded by trees. It was once a grand and stately home, built in the early 20th century by a wealthy family who had a deep love for nature and the tranquility of the forest. The house was designed with a blend of architectural styles, featuring a mix of Victorian and Gothic elements.\n\nOver the years, the family's descendants continued to maintain the house, preserving its beauty and charm. However, as time passed, the family's fortunes began to decline, and they were forced to sell the house to pay off debts. The new owners, who were not as attentive to the house's upkeep, allowed it to deteriorate.\n\nThe house's current state, with its peeling paint and broken shutters, reflects the passage of time and the neglect it has suffered. The surrounding trees have grown taller, and the house has become a relic of the past, a testament to the family's once-great legacy.
  * **Llama Adapter Response:** The abandoned old house has a rich and intriguing history. It was built in the late 1800s by a wealthy family who wanted to create a grand estate in the heart of the forest. The family was known for their philanthropy and generosity, and they wanted to provide a comfortable and luxurious home for themselves and their guests.The house was designed with a mix of Victorian and Gothic architectural styles, featuring tall, narrow windows, elaborate woodwork, and a grand entrance. The family lived in the house for several decades, hosting lavish parties and entertaining guests from all walks of life. However, as time passed, the family's fortunes changed, and they were forced to sell the estate.The new owners were not as wealthy as the previous family, and they struggled to maintain the house. Over time, the house fell into disrepair, and it became a symbol of the family's decline. The house was eventually abandoned, and it has remained empty and uninhabited ever since.The house still stands tall and proud, a reminder of the family's legacy and the beauty of the bygone era. The surrounding forest has grown over
  

<div class="row mt-1">
   <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2022-12-01-visual-text-poineer/image_prompt/lemon.jpeg" class="img-fluid rounded" %}
  </div> 
</div>

**<mark>Explain Human-Object Relationship</mark>**<br>
**Prompt for picture:  What is this picture about? What's the feeling of the people and why they have such feelings?**
  * **Instruct BLIP (7B) Response:** The image depicts two young children, a boy and a girl, sitting on a couch while eating slices of lemon.<span style="color:red"> They seem to be enjoying their snack, as they are smiling and laughing together. It is likely that they are experiencing a positive emotional state, such as happiness or contentment, due to their shared enjoyment of the lemon slices.</span> This scene captures a moment of joy and camaraderie between the two children, emphasizing the importance of spending quality time together and savoring simple pleasures in life.
  * **LLaVA-1.5 (7B) Response:** <span style="color:green">The image features two children, a boy and a girl, sitting next to each other and eating lemon wedges. They both have a sad or unhappy expression on their faces. The reason for their sadness could be due to the taste of the lemon wedges, which are known to be sour and can be unpleasant for some people.</span> Alternatively, their sadness might be unrelated to the lemon wedges and could be due to other factors, such as personal issues or emotions. The children's expressions and actions suggest that they are not enjoying the experience of eating the lemon wedges.
  * **Llama Adapter Response:** <span style="color:red"> The image depicts a young boy and a young girl sitting on a bench, each holding a yellow banana. They appear to be enjoying their time together, possibly sharing a snack or a meal.</span> The girl is smiling, which suggests that she is happy and content, while the boy is making a face, which could indicate that he is amused, surprised, or simply expressing his emotions. The combination of the girl's smile and the boy's facial expression creates a playful and lighthearted atmosphere, suggesting that they are having a fun and carefree moment together.

<div class="row mt-1">
   <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2022-12-01-visual-text-poineer/image_prompt/map-magnifying_glass.jpeg" class="img-fluid rounded" %}
  </div> 
</div>

**<mark>Explain Object-Object Relationship</mark>**<br>
**Prompt for picture:  Explain the relationship between the magnifying glass and the antique map.**
  * **Instruct BLIP (7B) Response:** The relationship between the magnifying glass and the antique map is that the magnifying glass is placed on top of the map, allowing the viewer to examine the details of the map more closely.<span style="color:green"> By using the magnifying glass, the viewer can gain a better understanding of the map's features, such as geographical locations, landmarks, and other details that might be difficult to see with the naked eye.</span> The magnifying glass serves as a tool to enhance the viewing experience of the antique map, making it more informative and engaging for the viewer.
  * **LLaVA-1.5 (7B) Response:** The magnifying glass is placed over the antique map, which is a vintage, hand-drawn map of the world. <span style="color:green">The magnifying glass is used to examine the details of the map more closely, providing a better understanding of the geographical features and information displayed on the map. The magnifying glass enhances the visual experience by allowing the viewer to see the intricate details and nuances of the antique map,</span> which might not be easily discernible without the aid of the magnifying glass. This combination of the magnifying glass and the antique map creates a visually engaging and educational experience for the viewer.
  
  * **Llama Adapter Response:** <span style="color:green">The magnifying glass is used to enhance the view of the antique map, allowing the viewer to see more details and possibly identify specific locations or features on the map more easily.</span> The magnifying glass can be helpful in studying the map's content, such as historical events, geographical features, or cultural aspects, as it can provide a closer and clearer view of the map's details.

<div class="row mt-1">
   <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2022-12-01-visual-text-poineer/image_prompt/bed.png" class="img-fluid rounded" %}
  </div> 
</div>

**<mark>Implicit Relationship Inference</mark>**<br>
**Prompt for picture:  What is this picture about?**
  * **Instruct BLIP (7B) Response:** The image depicts a woman lying in a hospital bed, with her head resting on the pillow. She appears to be experiencing some discomfort or pain, as she is covering her face with her hands. There are several bottles scattered around the room, likely containing medication or other medical supplies.  In addition to the woman and the hospital bed, there are two chairs present in the scene. One chair is placed near the bed, while the other is positioned closer to the left side of the room. A cup can be seen on one of the chairs, suggesting that it may have been used by the woman or someone else during their stay in the hospital.<span style="color:red"> Overall, the scene conveys a sense of concern and care for the woman's well-being.</span>
  * **LLaVA-1.5 (7B) Response:** The image depicts a woman kneeling beside a hospital bed, holding her hands to her face. She appears to be in a hospital room, possibly in a state of distress or concern. The bed is positioned in the center of the room, and there are several bottles scattered around the area. \n\nIn addition to the bottles, there are two cups placed near the bed, and a vase can be seen on the left side of the room. <span style="color:green">The woman's posture and the presence of the hospital bed suggest that she might be a patient or a caregiver in the hospital.</span>
  * **Llama Adapter Response:** The image depicts a woman sitting on a bed in a hospital room, looking down and covering her face with her hands. She appears to be in a state of distress or sadness. The room is well-lit, with a window visible in the background. There are several chairs in the room, with one chair placed near the woman and another chair situated closer to the window. A clock is also visible on the wall, providing a sense of time passing. <span style="color:red">(No general conclusion is reached by the model)</span>


#### Conclusion
The following table summarizes the performance of the models on different prompts (ideas/concepts). If the model correctly covers what the question asks then, we evaluate it as a success and if the model starts hallucinating or even gets stuck or gives an irrelevant answer then, we mark that as a failure.

| Model     | Emotion / Atmosphere Inference | Create a Backstory | Predict Future | Explain Object-Object Relationship | Explain Human-Object Relationship | Explain Human-Human Relationship | Confusing Image |Implicit Relationship Inference |
|-----------|----------|----------|----------|----------|----------|----------|----------|
| InstructBLIP   | ❌                |   ❌ (Lacks creativity)     |  ✔       |   ✔      |    ❌     | ✔        | ❌        |  ❌  |
| LLaVA-1.5   | ✔                |   ✔      |  ✔       |   ✔      |     ✔     | ✔        | ✔        |   ✔ |
| LLAMA-Adapter   | ✔        |   ✔      |     ✔    |     ✔    |  ❌       | ✔        | ❌        |     ❌  |




## Qualitative Analysis-Robustness
To test the robustness of a multi-modal model, we provide a prompt that is completely unrelated to the image. This will evaluate the model's ability to focus on the provided textual input, ignoring the irrelevant visual context. There are two cases, with or without the hint that the image is unrelated to the picture. 

<div class="row mt-1">
   <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2022-12-01-visual-text-poineer/unrelated/rain.jpeg" class="img-fluid rounded" %}
  </div> 
</div>

**Prompt for picture:  Where is the man with the red hat?**
  * **Instruct BLIP (7B) Response:** The man with the red hat is standing under an umbrella in the rain.
  * **LLaVA-1.5 (7B) Response:**The man with the red hat is not visible in the image. The focus is on the open umbrella and the raindrops falling around it. 
  
  * **Llama Adapter Response:** The man with the red hat is standing in the rain, holding the umbrella to protect himself from the rain.

**Prompt for picture:  Where is the man with the red hat? Note that the image might be unrelated to this question.**
  * **Instruct BLIP (7B) Response:** behind umbrella
  * **LLaVA-1.5 (7B) Response:** The man with the red hat is not visible in the image. The image features a close-up of a black umbrella in the rain, with water droplets falling on it. 
  
  * **Llama Adapter Response:** The man with the red hat is not visible in the image. The image only shows the umbrella and the raindrops on the umbrella.


<div class="row mt-1">
   <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2022-12-01-visual-text-poineer/unrelated/desert.jpeg" class="img-fluid rounded" %}
  </div> 
</div>

**Prompt for picture:  Why there are so many people on the ocean?**
  * **Instruct BLIP (7B) Response:** There are no people on the ocean in the image.
  * **LLaVA-1.5 (7B) Response:** There are no people on the ocean in the image. The scene features a sandy beach with a tree and a dune, and the sky is orange, suggesting a sunset. The absence of people on the ocean could be due to various reasons, such as the time of day, weather conditions, or the location being a remote or less popular beach. It is also possible that the image was taken during a time when the beach is not crowded, such as early morning or late evening, when fewer people are visiting the beach.
  
  * **Llama Adapter Response:** There are so many people on the ocean because they are enjoying a beach day, spending time together, and participating in various activities such as swimming, sunbathing, or simply relaxing. The beach is a popular destination for people to unwind, socialize, and engage in recreational activities. The presence of many people on the beach indicates that it is a popular spot for leisure and relaxation.

**Prompt for picture:  Why there are so many people on the ocean? Note that the image might be unrelated to this question.**
  * **Instruct BLIP (7B) Response:** no
  * **LLaVA-1.5 (7B) Response:** The image shows a sandy desert landscape with a lone tree and a dune. There are no people visible in the image, so it is not possible to determine why there are so many people on the ocean. The image is focused on the desert environment and the lone tree, and the ocean is not a part of the scene 
  
  * **Llama Adapter Response:** There could be several reasons for the presence of so many people on the ocean. Some possible explanations include:1. Recreational activities: The ocean might be a popular destination for beachgoers, surfers, swimmers, or sunbathers, who enjoy spending time in the water and on the sand.2. Tourism: The ocean could be part of a popular tourist destination, attracting visitors from all over the world.3. Events: The ocean might be the location of a special event, such as a beach festival, a concert, or a sports competition, drawing a large crowd of spectators.4. Emergency: The ocean could be the scene of an emergency, such as a natural disaster, a search and rescue operation, or a mass evacuation.In any case, the image does not provide enough information to determine the exact reason for the large number of people on the ocean.


## Qualitative Analysis-Embedding Visualizations

In this section, we wanted to explore how image and text embeddings align in different models and whether that align is beneficial for the model. In that aspect, we generated the image and text embeddings of **VQA-v2** validation dataset, which contains **MS COCO** images with corresponding Questions and Answers. We sampled $$200$$ image text pairs from the former dataset and generated the corresponding visual and textual embeddings using the pre-trained models (Instruct BLIP, LLaVA-1.5 and LLaMA Adapter). These models generate both image and text embeddings of dimensions $$4096$$, respectively. 

In order to visualize the embeddings, we used **PCA** and **t-SNE** methods to reduce the dimensions to $$2$$ and $$3$$, respectively. Before computing the principal components and t-SNE, we concatenated the image and text embeddings (of initial dimensions $$\mathbf{R}^{D \times 4096}$$ where **D** corresponds to number of samples which in our case are $$200$$) across the dim-$$0$$ to obtain a matrix of dimensions $$\mathbf{R}^{2 D \times 4096}$$. We then computed the principal components and t-SNE on the concatenated matrix to obtain the reduced dimensions of $$2$$ and $$3$$, respectively. 

### PCA

The following figures show the interactive visualizations of PCA for the three models. The figures are in the following order (**Instruct BLIP, LLaVA-1.5 and LLaMA Adapter**).

**< Add InstructBLIP PCA figure here >**

<div class="l-page">
  <iframe src="{{ 'assets/html/2022-12-01-visual-text-poineer/instructblip_pca_2d.html' | relative_url }}"  frameborder='0' scrolling='no' height="700px" width="100%"></iframe>
</div>

<div class="l-page">
  <iframe src="{{ 'assets/html/2022-12-01-visual-text-poineer/llava_pca_2d.html' | relative_url }}"  frameborder='0' scrolling='no' height="700px" width="100%"></iframe>
</div>

<!-- **< Add LLaMA-Adapter PCA figure here >** -->

<div class="l-page">
  <iframe src="{{ 'assets/html/2022-12-01-visual-text-poineer/adapter_pca_2d.html' | relative_url }}"  frameborder='0' scrolling='no' height="700px" width="100%"></iframe>
</div>

### t-SNE

The following figures show the interactive visualizations of t-SNE for the three models. The figures are in the same order (**Instruct BLIP, LLaVA-1.5 and LLaMA Adapter**).

**< Add InstructBLIP t-SNE figure here >**

<div class="l-page">
  <iframe src="{{ 'assets/html/2022-12-01-visual-text-poineer/instructblip_tsne_3D.html' | relative_url }}"  frameborder='0' scrolling='no' height="700px" width="100%"></iframe>
</div>

<div class="l-page">
  <iframe src="{{ 'assets/html/2022-12-01-visual-text-poineer/llava_tsne_3D.html' | relative_url }}"  frameborder='0' scrolling='no' height="700px" width="100%"></iframe>
</div>

<!-- **< Add LLaMA-Adapter t-SNE figure here >** -->

<div class="l-page">
  <iframe src="{{ 'assets/html/2022-12-01-visual-text-poineer/adapter_tsne_3D.html' | relative_url }}"  frameborder='0' scrolling='no' height="700px" width="100%"></iframe>
</div>

As, it can seen from the above plots of PCA and t-SNE that for Instruct-BLIP the embeddings are clustered together indicating the text-conditioned training of the model. However, since LLaVA and LLaMA-Adapter does not use text-conditioning while training the visual embedding extractor the embeddings are well separated. 

## Quantitative Analysis-Image-to-Text Retrieval
In this section, we conducted a study to analyze if we can extract the corresponding text embeddings from the image embeddings in the original mapped space of $$\mathbf{R}^{4096}$$. For that, we used the same setup as explained in the previous section and used the $$200$$ image-text pairs from the validation split of **VQA-v2** dataset. We then computed the cosine similarity for each image embedding with all the text embeddings and extracted the top $$k$$ text embeddings with the highest cosine similarity. We then computed the accuracy of the model by checking if the corresponding text embedding is present in the top $$k$$ text embeddings. The results are shown in the following figure. From the figure, it can be seen that Instruct Blip has the perfect retrieval accuracy which indicates that text conditioning based image embedding extraction is beneficial for the model. However, LLaVA and LLaMA-Adapter have a retrieval accuracy which is at par with the random retrieval accuracy.

<div class="row mt-1">
   <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2022-12-01-visual-text-poineer/retrieval-acc.png" class="img-fluid rounded" %}
  </div>
</div>



## Quantitative Analysis-Unified Framework

In the comparison among the three methods, their individual use of diverse settings and datasets for training makes direct comparisons challenging. To address this, we introduced a unified framework that focuses on evaluating the impact of different architectural designs by incorporating two variable factors.

{% include figure.html path="assets/img/2022-12-01-visual-text-poineer/unified_framework.png" class="img-fluid" %}
<div class="caption">
The model architecture of our unified framework.
</div>

As shown in the framework overview, our framework consists of an image encoder, a Q-Former to bridge the extract image features, and a LLM. We used CLIP-ViT for the image encoder and Vicuna-7B for the LLM. Vicuna is a decoder-only Transformer instruction-tuned from LLaMA. Both the image encoder and the LLM are frozen during the training. We used VQAv2 for training and validation.

There are two changing factors in our ablation study:
1. Text-conditioning on the Q-Former.
2. Integrating an adapter into multiple LLM layers or merely concatenating image embeddings with text embeddings in the LLM input.

The results are shown below. For the VQAv2 dataset, we've recorded the top-1 accuracy percentages for different models and conditions. When utilizing an adapter, we observed a 61.83% accuracy with conditional text, slightly higher than 61.47% without it. However, removing the conditional text resulted in a decrease to 58.59% without the adapter and an increase to 60.55% with it.

|             | VQAv2 - top-1 accuracy (%) |
|-------------|---------------------------|
|             | W/ Adapter | W/O Adapter   |
| W/ Conditional Text | 61.83       | 61.47         |
| W/O Conditional Text | 58.59     | 60.55         |
| LLaVA-1.5 (7B) | --      | 78.5*          |

From these observations, two conclusions emerge:

1. Text-conditioning marginally improves performance.
2. The impact of the adapter is contingent on text-conditioning.

The initial conclusion suggests that conditioning on text bolsters the extraction of image features, aligning them more closely with the text for enhanced question-answering capabilities. Regarding the latter, it's plausible that since the adapter adjusts image features across multiple layers, the efficacy is amplified when conditioned on text, ensuring the quality of extracted image embeddings.







