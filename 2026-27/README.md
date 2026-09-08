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
