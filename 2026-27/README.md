### Prof. Alfio Ferrara

# Natural Language Processing

#### Master degree in Computer Science & Master degree in Data Science for Economics and Health

###### Università degli Studi di Milano

## Part 0 — Course Introduction

This opening lesson sets the stage before any technical content: what this course is about, why natural language is a hard object to compute with, what students will be able to build and understand by the end, and how the project and assessment work. It also previews the narrative arc of the course, from raw text to tokens, from tokens to numbers, from numbers to meaning, and finally to modern large language models

**Lessons:**

1. **Course introduction: goals, narrative arc, project and assessment overview**

---

## Part I — From Text to Numbers

Before a machine can do anything with language, text has to be broken down into units it can digest, and immediately we face a trade-off between character-level and sentence-level granularity, which naturally leads to the notion of the *token*. Once we accept tokens as the basic unit, we discover two competing philosophies for producing them: linguistically motivated rules versus statistically learned subword units. Finally, tokens alone are not enough because they must be turned into numbers, which raises questions of feature selection, weighting, and the geometry of the resulting vector space.

**Lessons:**

2. **From text to tokens: character/word/sentence trade-offs and the notion of "token"**
3. **Tokenization approaches: linguistic (rule-based) vs. statistical (WordPiece/BPE)**
4. **Representing text numerically: feature selection, weighting, and the vector space (TF-IDF)**
5. **From vectors to classification: neural networks as classifiers, and how we evaluate them (precision, recall, F1, a first look at perplexity)**

---

## Part II — Learned Representations

Once a neural network is classifying text, we can look inside it: hidden layers turn out to hold a dense, learned representation of what matters for the task, quite different from the sparse, orthogonal vectors of TF-IDF. This observation is the seed of a much bigger idea: if networks can learn dense representations as a by-product of classification, we can design classification tasks specifically to obtain good representations of words themselves. This is exactly the intuition behind Word2Vec and word embeddings.

**Lessons:**

6. **What's inside the network: hidden layers as learned dense representations**
7. **Beyond orthogonal tokens: Word2Vec and word embeddings**

---

## Part III — Language Modeling

We can reuse the classification machinery in a creative way: instead of predicting a label, we predict the next word. This reframing is language modeling. Historically, the first successful approach was purely statistical  and lead to Markov language models, which estimate probabilities from co-occurrence counts. But Markov models hit a wall quickly: they cannot capture long-range dependencies or generalize beyond seen n-grams, which motivates the shift toward neural approximations of language modeling.

**Lessons:**

8. **Classification in disguise: predicting the next word**
9. **Statistical language models: Markov LMs**
10. **Where Markov breaks down: the need to approximate language with neural methods**

---

## Part IV — Sequences to Transformers

Treating text as a sequence and processing it step by step (RNNs) seems like the natural next move, but it introduces its own problems like vanishing gradients, poor parallelization, and difficulty modeling long-range dependencies. This motivates a different idea: process the whole sequence at once, using token embeddings enriched with positional information. But a Transformer without attention is still missing something essential, some mechanism to let each token gather information from the others. We introduce that mechanism, attention, together with the residual stream that lets information flow through the network, and finally assemble the full encoder-decoder Transformer architecture.

**Lessons:**

11. **Text as sequence: why naive sequential processing is hard (RNNs)**
12. **Rethinking the input: token + positional embeddings for whole-sequence processing**
13. **A Transformer without attention: what's still missing?**
14. **The attention mechanism and the residual stream**
15. **Assembling the Transformer: encoder-decoder architecture**

---

## Part V — Modern Architectures

With the Transformer building block in hand, we can look at the two architectural families that shaped the field: BERT, built around bidirectional encoding and masked pretraining, and GPT, built around autoregressive, decoder-only generation. Understanding both, and why they were pretrained the way they were, is essential to understanding everything that follows.

**Lessons:**

16. **BERT: architecture and pretraining objectives**
17. **GPT: architecture and autoregressive pretraining**

---

## Part VI — From Pretrained Model to Usable System

A pretrained language model on its own is not yet a useful assistant or tool. This gap motivates the whole modern toolkit for adapting and steering models: fine-tuning and parameter-efficient methods like LoRA, prompt engineering, instruction tuning and RLHF (with a brief mention of retrieval-augmented generation), agentic use of models, and the proprietary APIs that make these models accessible in practice. This block also extends the story beyond text into multimodal models, showing how the same core ideas generalize to other modalities.

**Lessons:**

18. **From pretraining to a usable model: fine-tuning & parameter-efficient fine-tuning (LoRA)**
19. **Prompt engineering, instruction tuning & RLHF (brief mention of RAG)**
20. **Agents and proprietary model APIs**
21. **Multimodal models: extending beyond text**

---

## Part VII — Problems and Practice

No model is neutral or fully transparent, and no course is complete without confronting that directly: what can we actually explain about a model's behavior (XAI, mechanistic interpretability), and what cultural and statistical biases do these models encode and amplify? The course closes on practical ground, surveying the frameworks used to deploy LLMs in the real world and kicking off the final project.

**Lessons:**

22. **Interpretability: XAI & mechanistic interpretability**
23. **Bias & stereotypes: cultural and statistical bias in language models**
24. **Frameworks for deployment (llama.cpp, vLLM, MLX LM) & project kickoff**


# Essential Bibliography

## Part I — From Text to Numbers

- Salton, G., & Buckley, C. (1988). *Term-weighting approaches in automatic text retrieval.* Information Processing & Management, 24(5), 513–523.
  [https://doi.org/10.1016/0306-4573(88)90021-0](https://doi.org/10.1016/0306-4573(88)90021-0)
  — The classic reference for TF-IDF weighting.

- Sennrich, R., Haddow, B., & Birch, A. (2016). *Neural Machine Translation of Rare Words with Subword Units.* ACL.
  [https://arxiv.org/abs/1508.07909](https://arxiv.org/abs/1508.07909)
  — Introduces Byte-Pair Encoding (BPE) for subword tokenization.

---

## Part II — Learned Representations

- Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). *Efficient Estimation of Word Representations in Vector Space.* arXiv:1301.3781.
  [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781)
  — The original Word2Vec paper (CBOW and Skip-gram).

- Pennington, J., Socher, R., & Manning, C. D. (2014). *GloVe: Global Vectors for Word Representation.* EMNLP.
  [https://aclanthology.org/D14-1162.pdf](https://aclanthology.org/D14-1162.pdf)
  — Word embeddings from global co-occurrence statistics, complementary to Word2Vec.

---

## Part III — Language Modeling

- Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). *A Neural Probabilistic Language Model.* Journal of Machine Learning Research, 3, 1137–1155.
  [https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
  — The first neural language model, bridging statistical and neural approaches.

---

## Part IV — Sequences to Transformers

- Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation, 9(8), 1735–1780.
  [https://doi.org/10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  — Introduces LSTM, addressing the limitations of vanilla RNNs.

- Sutskever, I., Vinyals, O., & Le, Q. V. (2014). *Sequence to Sequence Learning with Neural Networks.* NeurIPS.
  [https://arxiv.org/abs/1409.3215](https://arxiv.org/abs/1409.3215)
  — Encoder-decoder framework for sequence generation.

- Bahdanau, D., Cho, K., & Bengio, Y. (2015). *Neural Machine Translation by Jointly Learning to Align and Translate.* ICLR.
  [https://arxiv.org/abs/1409.0473](https://arxiv.org/abs/1409.0473)
  — Introduces attention as a fix for the fixed-length bottleneck in seq2seq.

- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). *Attention Is All You Need.* NeurIPS.
  [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
  — The Transformer architecture.

---

## Part V — Modern Architectures

- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL.
  [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)

- Radford, A., Wu, J., Child, R., et al. (2019). *Language Models are Unsupervised Multitask Learners.* OpenAI (GPT-2 technical report).
  [https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

- Brown, T. B., Mann, B., Ryder, N., et al. (2020). *Language Models are Few-Shot Learners.* NeurIPS.
  [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)
  — GPT-3 and the emergence of in-context learning.

---

## Part VI — From Pretrained Model to Usable System

- Hu, E. J., Shen, Y., Wallis, P., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685.
  [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)

- Ouyang, L., Wu, J., Jiang, X., et al. (2022). *Training Language Models to Follow Instructions with Human Feedback.* NeurIPS.
  [https://arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)
  — InstructGPT; the paper behind instruction tuning and RLHF.

- Wei, J., Wang, X., Schuurmans, D., et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS.
  [https://arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903)

- Lewis, P., Perez, E., Piktus, A., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS.
  [https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)
  — Reference for the brief mention of RAG.

- Radford, A., Kim, J. W., Hallacy, C., et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision.* ICML.
  [https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)
  — CLIP; entry point for the multimodal lesson.

---

## Part VII — Problems and Practice

- Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS.
  [https://arxiv.org/abs/1705.07874](https://arxiv.org/abs/1705.07874)
  — SHAP, a unifying framework for feature-attribution explanations.

- Bolukbasi, T., Chang, K.-W., Zou, J., Saligrama, V., & Kalai, A. (2016). *Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings.* NeurIPS.
  [https://arxiv.org/abs/1607.06520](https://arxiv.org/abs/1607.06520)

- Caliskan, A., Bryson, J. J., & Narayanan, A. (2017). *Semantics Derived Automatically from Language Corpora Contain Human-like Biases.* Science, 356(6334), 183–186.
  [https://arxiv.org/abs/1608.07187](https://arxiv.org/abs/1608.07187)

- Olah, C., Cammarata, N., Schubert, L., et al. (2020). *Zoom In: An Introduction to Circuits.* Distill.
  [https://distill.pub/2020/circuits/zoom-in/](https://distill.pub/2020/circuits/zoom-in/)
  — Foundational text for mechanistic interpretability.