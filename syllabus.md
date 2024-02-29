###### Master Degree on Computer Science

# Natural Language Processing

#### Instructor: Prof. Alfio Ferrara

##### Assistant Instructors: Sergio Picascia, Davide Riva, Elisabetta Rocchetti, Darya Shlyk   

The course provides an in-depth introduction to the main research topics in the field of Deep Learning applied to Natural Language Processing. In addition to the lectures, the course includes a final project through which students will acquire the necessary skills to design, implement and understand the main neural network models for natural language, using Python and Pytorch.

The materials and code are available on the GitHub [nlp](https://github.com/afflint/nlp) repository

##### Main background about NLP

> Jurafsky, D., & Martin, J. H. *Speech and Language Processing: An Introduction to Natural Language Processing*, Computational Linguistics, and Speech Recognition. [online edition](https://web.stanford.edu/~jurafsky/slp3/)

##### Monographic focus on Deep Learning for NLP, Language Models and Transformers

> Specific references will be provided by the instructors for all the topics discussed in class

### Course topics

##### Introduction to Natural Language Processing $\sim 3\ hrs$

Goals and applications. Bag of Words, Vector Space Model and TfIdf.

> Schütze, H., Manning, C. D., & Raghavan, P. (2008). *Introduction to information retrieval* (Vol. 39, pp. 234-265). Cambridge: Cambridge University Press. [Chapters 1-3; 6-7].

##### Tools for NLP and text processing $\sim 6\ hrs$

Tokenization and normalization methods. Spelling Correction. Logical models of meaning. Description Logics and knowledge bases. Word senses and WordNet. Phonetics.

`[TOOLS]: NLTK; SpaCy; TextBlob.`

##### NLP for information retrieval, text classification, and sentiment analysis $\sim 6\ hrs$

> Schütze, H., Manning, C. D., & Raghavan, P. (2008). *Introduction to information retrieval* (Vol. 39, pp. 234-265). Cambridge: Cambridge University Press. [Chapters 8-9]
>
> Jurafsky, D., & Martin, J. H. *Speech and Language Processing: An Introduction to Natural Language Processing*, Computational Linguistics, and Speech Recognition. [Chapters 4-5]

##### Hidden Markov Models, Conditional Random Fields and Expectation Maximization $\sim 3\ hrs$ 

> Wallach, Hanna M. "Conditional random fields: An introduction." Technical Reports (CIS) (2004): 22.
>
> Jurafsky, D., & Martin, J. H. *Speech and Language Processing: An Introduction to Natural Language Processing*, Computational Linguistics, and Speech Recognition. [Appendix A]

##### Latent Semantic Indexing and Latent Dirichlet Allocation $\sim 3\ hrs$

> Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent dirichlet allocation. *Journal of machine Learning research*, *3*(Jan), 993-1022.

##### N-gram and Markov Language Models $\sim 3\ hrs$

> Jurafsky, D., & Martin, J. H. *Speech and Language Processing: An Introduction to Natural Language Processing*, Computational Linguistics, and Speech Recognition. [Chapters 3]

##### Introduction to Neural Networks and introduction to PyTorch functional $\sim 3\ hrs$

> Aggrawal, C. C. (2018). *Neural networks and deep learning: A textbook*. Springer Berlin/Heidelberg
>
> "Natural Language Processing with PyTorch: Build Intelligent Language Applications Using Deep Learning" (2019), Delip Rao and Brian McMahan, Chapter 1, 3.

##### Word embeddings and Word2Vec $\sim 3\ hrs$

> Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. *Advances in neural information processing systems*, *26*.

##### Sequence to sequence modeling and Neural Language Models $\sim 9\ hrs$

The module addresses the problem of sequence modeling, and presents the basic neural models excelling in the task: **Recurrent Neural Networks**. The goal of this module is to explain the working mechanism of **RNNs**, and to illustrate their power on sequence classification and sequence prediction tasks (language modeling). Introduction to `seq2seq` modeling paradigm with an architecture based on **encoder-decoder** through a practical use case of entailment generation.

> "Natural Language Processing with PyTorch:Build Intelligent Language Applications Using Deep Learning" (2019), Delip Rao and Brian McMahan, Chapter 6, 7, 8
>
> "Machine Learning with PyTorch and Scikit-Learn" (2022) Sebastian Raschka, Chapter 15
>
> "Speech and Language Processing" (2022), Dan Jurafsky and James H.Martin, Chapter 9

##### Attention, Transformers and BERT $\sim 6\ hrs$

This module introduces the Transformer model, which is a neural language model capable of processing textual sequences. An empirical application is discussed; in particular, this example focuses on BERT, which is a Transformer model based on Encoder-Decoder architecture.

> "Speech and Language Processing" (2022), Dan Jurafsky and James H.Martin, Chapters 9, 10, 11

##### Text2Images $\sim 3\ hrs$ 

This module gives a brief overview of the main architectures for analyzing images, CNNs and ViTs. It then focuses on zero-shot transfer exploiting natural language supervision.

> O'Shea, K., & Nash, R. (2015). An introduction to convolutional neural networks. arXiv preprint arXiv:1511.08458.

> Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.

> Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021, July). Learning transferable visual models from natural language supervision. In International conference on machine learning (pp. 8748-8763). PMLR.
