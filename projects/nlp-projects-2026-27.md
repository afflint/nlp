**Master Degree in Computer Science**

**Master Degree in Data Science for Economics and Health**

# Natural Language Processing

#### **Prof. Alfio Ferrara**

**Dott. Sergio Picascia, Dott.ssa Elisabetta Rocchetti**

*Department of Computer Science, Università degli Studi di Milano
Room 7012 via Celoria 18, 20133 Milano, Italia <a href="mailto:alfio.ferrara@unimi.it">alfio.ferrara@unimi.it</a>*

# Final projects 2026–27: themes and guidelines

## The exam at a glance

The exam consists of **a single oral interview** made of two parts:

1. **Theory questions** on the topics of the course.
2. **Discussion of your final project** (research paper, code repository and presentation).

> **IMPORTANT: the theory questions are mandatory and indispensable to pass the exam.**
> A satisfactory answer to the theory questions is a **necessary condition for a passing grade**.
> An excellent project does not compensate for insufficient knowledge of the course contents, and, conversely, a good theory preparation does not replace the project. Both parts are required, and the final grade takes both into account.
> The theory questions cover the whole programme of the course (lectures, slides and the material indicated by the instructors), not only the topics related to your project. Prepare for them independently of the project work.

The project part of the interview is assessed not only on its output, but also on your **ability to explain and justify every choice you made**, including the content of your reference paper and the reasons why your work goes beyond it.

## Instructions

The final project consists in the preparation of a short study on one of the topics of the course, identifying a precise research question and measurable objectives. The project will propose a methodology for solving the research question and provide an experimental verification of the results obtained according to results evaluation metrics. The emphasis is not on obtaining high performance but rather on the critical discussion of the results obtained, in order to understand the potential effectiveness of the proposed methodology.

**What is new this year.** We do not provide ready-made project descriptions. Instead, we propose **ten broad themes (tracks)**. You choose one track and **you design your own project within it**:

1. Choose a track from those listed below.
2. Choose a **reference paper** from the closed list of venues (see [Reference paper](#reference-paper)). Study it carefully: it is your starting point and source of inspiration.
3. Formulate your **own research question**, with measurable objectives, going beyond the reference paper. You are not expected to reproduce the reference paper.
4. Design the methodology and the experiments, and carry them out within the resources you have (see [Resources and feasibility](#resources-and-feasibility)).
5. Write the paper, prepare the GitHub repository and the presentation.

The tracks describe a scope and the requirements that a good project must meet. They are **not** assignments: the research question, the data, the models, the metrics and the experimental design are yours. Projects that are very similar to each other, or that merely apply a generic recipe to a topic, will be harder to defend at the interview.

The results must be documented in a short article of not less than 4 pages and no more than 8, written in $\LaTeX$. You also have to provide access to a [GitHub](https://github.com/) repository containing the code and reproducible experimental results.

Finally, the project will be discussed after a **10 minutes presentation in English with slides**. The presentation must cover your reference paper as well as your own work.

## Procedure

Exam dates are just for the registration of the final grade. The interview (theory questions and project discussion) will be set by appointment, according to the following procedure:

1. Subscribe to any available date.
2. Contact Prof. Ferrara as soon as
   1. The project is finished and ready to be discussed.
   2. After the date of your subscription is expired.
3. Setup an appointment and discuss your work.

**IMPORTANT**: **how to contact Prof. Ferrara and submit the project**:

The procedure is reported [here on the Ariel web site](https://myariel.unimi.it/mod/forum/discuss.php?d=98767).

There is no preliminary approval of the project: you choose a track and proceed on your own.

## Reference paper

Every project must be anchored to **at least one reference paper**, chosen by you from the closed list of venues below. The paper is a source of inspiration: you are not expected to reproduce it, but you must be able to present it, explain its methodology and results, and discuss its limitations. Your own research question and experiments must go beyond it, and the report must make clear what is yours and what comes from the reference.

**Accepted venues**

- **NLP and computational linguistics:** ACL, EMNLP, NAACL, EACL, AACL-IJCNLP, COLING, LREC-COLING, CoNLL, \*SEM; TACL and Computational Linguistics. Papers in the Findings of ACL, EMNLP, NAACL and EACL are also accepted.
- **Machine learning and AI:** NeurIPS (including the Datasets & Benchmarks track), ICML, ICLR, COLM, AAAI, IJCAI; TMLR, JAIR, JMLR.

**Further sources.** Beyond the reference paper, you are free to draw on any other source that is useful for your project. Cite everything you use.

## Structure of the paper

1. **Introduction**

   Provides an overview of the project, a short discussion of the pertinent literature, and the presentation of the reference paper and of how your work relates to it.

2. **Research question and methodology**

   Provides a clear statement on the goals of the project, an overview of the proposed approach, and a formal definition of the problem. It must make explicit what is original with respect to the reference paper.

3. **Experimental results**

   Provides an overview of the dataset used for experiments, the metrics used for evaluating performances, and the experimental methodology. Presents experimental results as plots and/or tables.

4. **Concluding remarks**

   Provides a critical discussion on the experimental results, their limits, and some ideas for future work.

## Resources and feasibility

Projects must be feasible with **limited computational resources** (a laptop, free-tier notebooks, small open models, subsets of datasets, inexpensive API usage). Each track indicates where the cost grows. You decide how ambitious to be, and you are expected to **justify your choices** (for example, why a given model size or sample size is sufficient to answer your research question). A small, well-designed and carefully analysed experiment is preferable to a large, poorly controlled one.

## AI Usage Disclaimer

Parts of this document have been developed with the assistance of **Anthropic's Claude**. The AI was used to support the **development of the structure of the tracks, the drafting of descriptive texts**, and the **formulation of project requirements**. All content produced with AI assistance has been **carefully reviewed, edited, and validated** by the instructor. The instructor takes full responsibility for the final content and its accuracy, relevance, and academic integrity.

## Using AI (for students)

Generative AI tools (such as ChatGPT, Claude, Mistral, or similar models) **may be used in this project**, both as an object of investigation and as a tool to support the development process. You are encouraged to explore how these models function, interact with them creatively, and leverage them as **inspiration or assistance in ideation, drafting, or experimentation**.

However, **AI should not be used as a substitute for original work**. The responsibility for the structure, reasoning, and understanding of the project remains entirely with you.

If generative AI has been used at any stage of the project, it is **mandatory to include a disclaimer** clearly specifying:

- **Which models** have been used (e.g., GPT-4, Claude 3, etc.)
- **For what purposes** (e.g., drafting text, summarizing ideas, generating code or examples)
- **To what extent** the outputs were modified, verified, or integrated into the final submission

The project will be assessed not only based on its output, but also on your **ability to explain and justify all choices made**. The final interview will evaluate the **depth of understanding**, and any lack of clarity or over-reliance on AI-generated material without proper insight may negatively affect the evaluation. Remember that the theory questions of the interview cannot be prepared with AI assistance during the exam: they test what you have learned.

Generative AI should be seen as a **creativity support tool**, not as a replacement for critical thinking, problem solving, or technical development.

## Instructions on coding

All project code should be written with clarity, modularity, and reusability in mind. The implementation should **not consist of a single large notebook**, but rather follow a structured and maintainable design. The recommended practice is to organize the logic into **Python modules and packages**, using **object-oriented programming (OOP)** principles where appropriate (e.g., defining classes for models, datasets, or evaluation pipelines).

Jupyter notebooks should be used primarily for **demonstration, experimentation, and visualization**, not for hosting the full application logic. Each notebook should import and showcase components from the main Python modules, illustrating how they work in practice. Code should include meaningful docstrings, comments, and clear function signatures.

Where possible, you are encouraged to separate concerns into layers (e.g., data loading and preprocessing, model interface, evaluation, and visualization) to facilitate testing and future reuse. The resulting repository should be **clean, reproducible, and extensible**, allowing others to replicate results or build upon the developed framework.

## How to read the tracks

Each track follows the same structure:

- **Overview**: what the track is about and what kind of research question it invites.
- **Possible perspectives**: examples that illustrate the breadth of the track. They are not assignments; you are free to propose something different within the scope of the track.
- **What we expect from a project**: the requirements that distinguish a good project in this track from a generic one.
- **Common pitfalls**: typical weaknesses to recognise and avoid.
- **Search keywords**: terms to help you find your reference paper on your own. No specific papers are suggested.
- **Feasibility and resources**: what can be done with limited resources and where the cost grows.

---

# Tracks

## Track 1. Strategy and Negotiation: Game Theory and Emergent Behaviour in LLMs

*What strategies emerge when language models play formally defined games, and how do they compare with theory and with human behaviour?*

### Overview

Language models are increasingly deployed as agents that bargain, cooperate, compete and coordinate. This track studies them as **players of formally defined games**, using game theory as background: payoffs, information structure, equilibria, repeated interaction. The goal is not to make models "talk to each other until one is convinced", but to **characterise the strategies that emerge**, how stable they are, and how they depend on controlled variables.

### Possible perspectives

- **Repeated games.** Do models cooperate, retaliate, forgive? Do they converge to known strategies (e.g., tit-for-tat) and how stable are they across rounds?
- **Bargaining and fairness.** How do models behave in ultimatum, dictator or public-goods games compared with human experimental data?
- **Information and deception.** What changes with private information, asymmetric incentives or cheap talk? When does communication help or hurt?
- **Sensitivity to the setting.** How do framing, assigned persona, number of rounds, order of play or model size change the strategy?
- **Adaptation.** Do models adapt in context to the opponent's strategy, or do they follow a fixed disposition?
- **Populations and tournaments.** What happens when models compete against scripted strategies or against each other in round-robin settings?

### What we expect from a project

- **A formalised game.** Define players, actions, information and payoffs. Without a formal structure the project does not belong to this track.
- **A point of comparison.** Compare the observed behaviour with a theoretical prediction (equilibrium, optimal play, known strategies) or with human behavioural data.
- **Analysis of strategies at the centre.** The contribution is the characterisation of what emerges (classification of strategies, stability, sensitivity to variables), not the fact that models were made to play.
- **Experimental rigour.** Use many repetitions, multiple seeds or samplings, and statistical tests. A single dialogue is an anecdote, not evidence.

### Common pitfalls

- Two or three chat models trying to persuade one another, with no payoff, no formal game and no benchmark.
- Reporting only win rates or agreement rates, without describing the strategies behind them.
- Labelling strategies by impression from a few transcripts instead of with a defined, checkable criterion.

### Search keywords

repeated games, iterated prisoner's dilemma, ultimatum game, bargaining, negotiation, Nash equilibrium, cheap talk, strategic reasoning, behavioural game theory, theory of mind, LLM agents, multi-agent systems, cooperation.

### Feasibility and resources

Games with discrete actions and short horizons are cheap and work with small open models or low-cost APIs. The cost grows with the number of rounds, agents and repetitions, so decide on these consciously and justify the sample size.

---

## Track 2. Fact-Checking and Retrieval: Lexical vs Semantic Search with LLMs

*How far can we trust what is retrieved, and what changes between lexical and embedding-based retrieval when the goal is to verify a claim?*

### Overview

Given a claim, a fact-checking system must retrieve evidence from a corpus and decide whether it supports, refutes or does not settle the claim. This track asks you to study the **reliability of retrieval** in this setting, comparing traditional lexical methods with retrieval based on embeddings produced by language models, and to analyse **when and why** each works or fails. Retrieval-augmented generation is one of the ways evidence can be used, but the track is not about building a new RAG architecture.

### Possible perspectives

- **Lexical vs dense vs hybrid retrieval.** In which kinds of claims does each approach win (named entities, numbers, rare terms, paraphrases, synonymy)?
- **Where does the error come from?** Retrieval failure or verification failure? How does the quality of the retrieved evidence propagate to the final verdict?
- **Robustness.** What happens with contradictory, outdated or deliberately poisoned evidence, with a change of domain, or with a language other than English?
- **Evidence vs model knowledge.** What does the model do when the retrieved evidence conflicts with what it "knows"?
- **Hard claim types.** Negations, numerical claims, temporal claims, multi-hop claims, claims with insufficient evidence.
- **Domain settings.** Health, climate, economic or scientific claims, where the cost of errors is higher.

### What we expect from a project

- **Retrieval evaluated separately from verification.** Report retrieval metrics (e.g., recall@k, nDCG) and the final verdict metrics separately. End-to-end accuracy alone is not enough.
- **A serious lexical baseline.** A well-configured lexical retriever, not a strawman.
- **Error analysis at the centre.** Categorise the failures and relate them to claim and evidence properties.
- **At least one stress condition.** For example poisoned or contradictory corpora, domain or language shift, temporal claims, or conflict with parametric knowledge.

### Common pitfalls

- Measuring only end-to-end accuracy and concluding that "dense retrieval is better" (or worse) without explaining why.
- Comparing a carefully tuned dense model with an untuned lexical baseline.
- Using an LLM as the only judge of evidence quality without checking its judgements on a sample.

### Search keywords

fact-checking, claim verification, evidence retrieval, dense retrieval, BM25, hybrid retrieval, retrieval-augmented generation, hallucination, knowledge conflicts, corpus poisoning, out-of-domain retrieval, multi-hop verification, misinformation.

### Feasibility and resources

Lexical indexes and small sentence encoders run on a laptop, and fact-checking datasets offer small subsets and sample sizes that are easy to handle. The cost grows when large language models are used as verifiers: use a small model or a sample, and say so.

---

## Track 3. Interventions and Causality: Measuring What Changes in LLM Behaviour

*What is the causal effect of an intervention (a prompt, a tuning step, a model change) on a measurable aspect of a model's behaviour?*

### Overview

Many claims about language models are statements about effects: "instruction tuning makes models more sycophantic", "asking the model to adopt a persona changes its answers", "quantisation hurts calibration". This track asks you to treat an **intervention as an experiment** and to estimate its effect on a **behavioural outcome** with an explicit design, controlling for confounders and quantifying the uncertainty.

### Possible perspectives

- **Prompt-level interventions.** Persona, language, framing, instruction wording, order of examples, reasoning prompts.
- **Training-level interventions.** Base vs instruction-tuned versions, light fine-tuning, preference tuning, model size.
- **Inference-level interventions.** Quantisation, decoding parameters, context length.
- **Outcomes.** Sycophancy, accuracy, calibration, refusals, consistency across paraphrases, linguistic complexity, stylistic features, expressed bias.
- **Heterogeneous effects.** For which kinds of inputs, domains or languages is the effect larger or reversed?
- **Automatic prompt optimisation as an intervention.** What is the effect of an optimised prompt, and does it generalise beyond the data on which it was optimised?

### What we expect from a project

- **A formalised design.** Define treatment, control, outcome, unit of analysis and the quantity to estimate (e.g., the average effect) before collecting data.
- **One change at a time.** Use paired prompts across conditions, with the semantic content controlled. For base vs instruction-tuned comparisons, discuss honestly what can and cannot be attributed to the tuning.
- **A validly measured outcome.** Justify the metric and validate it, for example on a hand-annotated sample. An LLM judgement that is never checked is not sufficient as the only measure.
- **Quantified uncertainty.** Confidence intervals or bootstrap, tests suited to paired data, multiple seeds or samplings, effect sizes and not only significance.
- **Threats to validity.** Discuss generalisation to other models, datasets and languages, and the confounders that remain.

### Common pitfalls

- Trying a few prompt variants and reporting that one "works better", without a design, controls or uncertainty.
- Comparing conditions that differ in more than one respect (e.g., different content and different framing).
- Reading a causal effect on the model's *behaviour* as an explanation of its *internal mechanism* (for the latter, see Track 6).

### Search keywords

causal inference in NLP, treatment effect, average treatment effect, instruction tuning, base vs instruct models, prompt sensitivity, persona prompting, sycophancy, calibration, quantisation effects, paired experiments, bootstrap confidence intervals, behavioural evaluation.

### Feasibility and resources

Base and instruction-tuned pairs of small open models and paired prompts on medium-size datasets run with little compute. The cost grows with the number of models and repetitions, which you can scale to your resources.

---

## Track 4. Culture and Language: Exploring the Cultural Sensitivity of LLMs

*How do language models represent, recognise and reproduce cultural differences, and how faithful is that representation?*

### Overview

Language models are used by people all over the world, but they are trained on data that represent cultures very unevenly. This track is an **exploratory investigation** of the cultural sensitivity of models. The scope is defined by the object of study (how models handle cultural differences), not by a method: **you choose the experimental perspective**.

### Possible perspectives

- **Values and norms.** Do the models reproduce the values of some populations rather than others? Compare their answers with survey data on values and attitudes.
- **Factual representation.** Does what a model says about a country or group (demographics, economy, health, occupations) match statistical reality? Statistical sources on the world, such as Worldometer, can serve as a quantitative reference, always keeping in mind that such sources are often aggregations of other sources, so it is good practice to trace and cite the primary ones (e.g., UN agencies, World Bank, WHO, national statistical offices).
- **Stereotypes and distorted representation.** Which associations, omissions or imbalances appear, and for which groups?
- **Everyday knowledge and practices.** Food, customs, festivities, etiquette, proverbs and symbols across cultures.
- **Language and pragmatics.** Register, politeness, formality, metaphor and communicative style across languages.
- **Language of the prompt vs content.** Does the language in which a question is asked change the cultural perspective of the answer?
- **Variation within a culture.** Regional, generational or dialectal differences.

### What we expect from a project

- **A precise exploratory question.** Exploratory does not mean vague: state what you want to find out and what result would contradict your expectation.
- **An external reference for the culture.** Human data (surveys, authentic corpora, native annotators, official statistics) are the point of comparison. Without one you only show that the model says different things, not that they are accurate or representative. Be clear about which dimension you measure: statistics measure facts, not norms and values.
- **At least two cultures or languages, chosen with a motivation.** Be aware that culture is not the same as country or language, and declare the limits of your choice.
- **Sensitivity to how the model is asked.** Test your conclusions with several formulations, prompt languages and samplings, otherwise you may be describing the artefact of one prompt.
- **Structured qualitative analysis.** Systematic coding of the responses (with a codebook and, if possible, agreement between annotators, or a hand-checked sample), not examples picked for convenience.

### Common pitfalls

- Generalising from a handful of striking answers.
- Treating a country or a language as a homogeneous culture.
- Using an LLM both to produce and to judge the cultural content without any human check.

### Search keywords

cultural alignment, cultural bias, cross-cultural values, World Values Survey, Hofstede dimensions, Moral Foundations Theory, stereotypes in language models, multilingual evaluation, culturally-aware NLP, everyday knowledge benchmarks, representation and erasure, low-resource languages.

### Feasibility and resources

Prompts, existing benchmarks and public survey data work with small models, but you need **reasonably good multilingual models**, and quality in under-represented languages is itself a result. Knowing languages other than English is an advantage, and you are encouraged to use it.

---

## Track 5. Literature and Language: LLMs as Readers and as Writers

*What do language models understand when they read literary texts, and what do they do when they write them?*

### Overview

Literature is a demanding test for language models: it requires sensitivity to tone, style, narrative structure, irony, point of view and intertextuality. This track invites you to look at models from **two complementary perspectives**: as **readers** (interpreting literary texts) and as **writers** (producing them). We deliberately leave the track open: we are interested in the ideas you bring when you confront models with literary production.

### Possible perspectives

- **Models as readers.** Can they track characters, plot, tone or irony? How do they handle long texts, historical or non-standard varieties of a language, or intertextual references? How do they compare with human readers or with expert annotations?
- **Memorisation vs reading.** Do models understand a text or recall what they have seen during training? Compare well-known and unknown works.
- **Models as writers.** How do their texts differ from human ones in diversity, style, narrative structure and originality?
- **Imitation.** Can models imitate an author, a genre or a period, and how can we tell?
- **Constraints and control.** How do prompts, constraints and tuning change the output?
- **Evaluation.** How should literary quality or creativity be measured in a defensible way, and do automatic metrics agree with human judgement?

### What we expect from a project

- **A human point of comparison.** Texts by human authors, human readers, expert annotations or human judgements. Without one you cannot tell whether what you observe is a fact about the model or about the text.
- **A justified and validated measure.** Justify the metric of style, diversity, quality or comprehension, and check it against human judgement on a sample. An unverified LLM judgement is not sufficient as the only measure.
- **Control of memorisation and contamination.** For known works, discuss whether the model has seen them. Recent, little-known or unpublished texts are one solution.
- **Structured qualitative analysis.** Read a sample systematically, with explicit criteria, rather than selecting examples for convenience.
- **Experimental rigour.** Use multiple samples and repetitions. One generated text proves nothing.

### Common pitfalls

- Extracting a few stylometric features and training a classifier with no research question beyond "can we tell them apart?".
- Evaluating "creativity" with a single metric that has not been validated.
- Testing comprehension on canonical works that the model has almost certainly memorised.

### Search keywords

computational literary studies, narrative understanding, story generation, stylometry, authorship attribution, creativity evaluation, text diversity, long-context comprehension, literary text analysis, narrative structure, character tracking, memorisation, machine-generated text.

### Feasibility and resources

Public-domain texts and short generations are cheap to process, even with small models. For long texts, mind the context window and work on excerpts if needed. **Copyright:** use public-domain works or short excerpts for research, and cite your sources.

---

## Track 6. Inside the Model: Mechanistic Interpretability of a Phenomenon

*How does a model implement a specific behaviour, and can we test our explanation with interventions on its internals?*

### Overview

In this track you apply the techniques of mechanistic interpretability seen in the course to **a phenomenon of your choice**. The objective is not to build a tool, but to **explain a behaviour from the inside**: formulate a hypothesis about where and how the model realises it, and put the hypothesis to the test. The research question concerns the model, and the techniques are instruments.

### Possible perspectives

- **Linguistic phenomena.** Subject-verb agreement, coreference resolution, negation, tense, number.
- **Factual recall and relations.** Where and how does the model retrieve a fact or follow a relation?
- **Pattern copying and induction.** In-context learning mechanisms, repetition, copying.
- **Simple reasoning or arithmetic.** Which components carry the computation?
- **Behavioural tendencies.** Sycophancy, refusal, sentiment or other behaviours, studied from the inside.
- **Representations.** How is a concept encoded (directions, subspaces, features), and can the encoding be manipulated?

### What we expect from a project

- **A circumscribed and measurable phenomenon.** A task with controlled inputs and a quantifiable outcome (e.g., the logit difference between the correct and the wrong answer), not a vague behaviour.
- **A falsifiable mechanistic hypothesis.** State in advance where you believe the behaviour is implemented (layers, heads, components) and which result would refute it.
- **Causal evidence, not just correlation.** Probing and attribution show that information is present, not that the model uses it. At least one intervention (ablation, activation patching, steering) with proper controls is needed.
- **Controls and baselines.** Compare with random components, carefully chosen corrupted inputs, and multiple prompts and examples, to exclude that the effect is an artefact of a single input.
- **Declared limits.** Discuss how far the result generalises to other prompts, models and sizes, and how complete the explanation is. A partial and honest explanation is worth more than a complete story that has not been verified.

### Common pitfalls

- Presenting attention maps or probe accuracies as an explanation of what the model does.
- Choosing a phenomenon that cannot be measured with a clear outcome.
- Verifying the hypothesis on one single prompt template.

### Search keywords

mechanistic interpretability, circuits, activation patching, causal tracing, logit lens, probing, attribution, induction heads, sparse autoencoders, superposition, steering vectors, representation engineering, faithfulness of explanations.

### Feasibility and resources

Small models (GPT-2 sized, Pythia, models of a few hundred million parameters) and libraries such as TransformerLens are sufficient for a solid project. The cost grows quickly with larger models, so choose consciously. **Distinction from Track 3:** Track 3 estimates causal effects on the model's behaviour (input and output); this track studies causality **inside** the model (its components).

---

## Track 7. Rules, States and Moves: LLMs and Board Games

*Can language models understand, track and reason about a game described in natural language?*

### Overview

A board game is a small formal system, described in natural language and equipped with a **verifiable ground truth**: a simulator or a rule engine can tell whether a move is legal and whether an inference is correct. This makes it an excellent testbed for the understanding of rules, the tracking of state and the quality of play. The focus of this track is on **the rules and the state of the game**; for interactive strategic behaviour between players, see Track 1.

### Possible perspectives

- **Rulebook comprehension.** Can models answer questions on the rules, find ambiguities or inconsistencies, and explain a game in simplified form?
- **State tracking.** Can they keep track of the game state over a whole match?
- **Legal moves and quality of play.** How well do they play against reference engines or agents?
- **Formalisation.** Can they translate a rulebook into a formal or executable description (a game description language, code)?
- **Modified rules.** What happens when a well-known game is changed with counterfactual rules?
- **Game design.** Can they design new games, and can playtesting in simulation estimate balance and playability?

### What we expect from a project

- **An executable ground truth.** Use a simulator or a rule validator. Assessments by eye are not sufficient.
- **Control of memorisation.** Very famous games are too well known. Use lesser-known games or **variants with modified rules**, in which memory does not help.
- **Separate metrics.** Rule understanding, correctness of the state and strategic quality are different things and must be measured separately.
- **Experimental rigour.** Use many games or states, and classify errors by type (a forgotten rule, a corrupted state, hallucinated components).

### Common pitfalls

- Testing only famous games and mistaking memorisation for understanding.
- Judging legality or game quality by reading a few transcripts.
- Evaluating "fun" or "balance" of generated games without any simulation or human assessment.

### Search keywords

game rule understanding, state tracking, board game benchmarks, general game playing, game description language, Ludii, procedural content generation, game design with LLMs, counterfactual rules, world models, Othello, game simulators.

### Feasibility and resources

Game states are short texts and the simulator runs locally, so small models are sufficient. The cost depends on the number of games and moves. Check the licence and the availability of rulebooks and data.

---

## Track 8. Beyond Text: Multimodal Models

*What does a non-textual modality add to a model, and does the model actually use it?*

### Overview

This track gives you the possibility of working with **multimodal models**: images, audio and speech, video, documents and charts. The modality is free, but the project must say **what the modality contributes**, avoiding the generic pattern "I gave images to a vision-language model and measured accuracy".

### Possible perspectives

- **Does the model really use the image or the sound?** Or does it rely on linguistic shortcuts?
- **Conflicts between modalities.** What happens when text and image disagree?
- **Sensitivity to the input.** Resolution, cropping, noise, position or order of the content.
- **Cultural and social representation.** How do models depict or describe people, places and practices across cultures (see Track 4)?
- **Multimodal fact-checking.** Claims with images, charts or tables (see Track 2).
- **Document and chart understanding.** Reading tables, figures and layouts.
- **Speech and audio.** Transcription errors across accents, languages or noise, and their effect on downstream tasks.
- **Joint representations.** How do image and text embeddings align, and where does the alignment fail?

### What we expect from a project

- **Isolating the contribution of the modality.** Use a text-only baseline and ablations (input removed, degraded, replaced, or substituted with a caption) to show whether the model uses the input.
- **A question about the modality, not only about the benchmark.** The research question should concern what the modality changes.
- **Validated evaluation.** An unverified LLM-based judgement is not sufficient as the only measure.
- **Declared data and licences.** Images and audio have stricter constraints than text: declare the sources, the licences and any privacy concerns.

### Common pitfalls

- Reporting benchmark accuracy of an off-the-shelf model with no hypothesis.
- Failing to distinguish visual understanding from language priors.
- Using images of people or personal data without considering licences and privacy.

### Search keywords

vision-language models, multimodal LLMs, CLIP, image captioning, visual question answering, multimodal hallucination, modality conflict, language priors, document understanding, speech recognition, audio-language models, cross-modal alignment, multimodal robustness.

### Feasibility and resources

This is the **most computationally demanding** track. Small models (CLIP or SigLIP, Whisper, vision-language models of a few billion parameters) and subsets of the data can be run on free-tier GPUs or modest hardware, but the cost increases quickly with model size and data volume. Choose consciously.

---

## Track 9. Trust and Provenance: Prompt Injection, Watermarking and Detection

*How robust are the mechanisms that protect language models and identify what they generate?*

### Overview

This track brings together three families of problems: **attacks and defences at the prompt level**, **provenance and detection of generated text**, and **privacy and memorisation**. The perspective is that of measurement and defence: you study how robust a mechanism is, under an explicit threat model, and what it costs.

### Possible perspectives

- **Prompt injection.** Direct and indirect injection (in retrieved documents, tools and agents) and the robustness of defences.
- **Watermarking.** Detectability, impact on text quality, and robustness to paraphrase, translation or editing.
- **Detection of machine-generated text.** Reliability, failure modes, behaviour across languages and across human writers, including non-native speakers.
- **Memorisation and privacy.** Membership inference and extraction of training data.
- **Poisoning and backdoors.** What happens when the training data or a retrieval corpus is tampered with (see also Track 2)?
- **Safety and utility trade-offs.** What does a defence cost in terms of the quality of the legitimate task?

### What we expect from a project

- **An explicit threat model.** Who attacks, what they know, what they can do. Without it there is no project.
- **A safety and utility trade-off.** Evaluate every defence both on the attack success rate and on the loss in quality on the legitimate task.
- **Correct metrics for detection.** Report the true positive rate at a fixed, low false positive rate (e.g., 1%), not only accuracy. False positives on human-written text are the critical case.
- **An adaptive adversary.** Test a defence against an attacker who knows it, not only against standard attacks.
- **A controlled environment.** Use public benchmarks and sandboxes, never real third-party systems. The focus is on measurement and defence, not on producing operational attacks.

### Common pitfalls

- Evaluating a defence only against attacks that were not designed against it.
- Reporting detection accuracy on a balanced set, without considering false positives on human text.
- Presenting a defence as robust without measuring the loss in utility.

### Search keywords

prompt injection, indirect prompt injection, jailbreaks, LLM agents security, watermarking language models, machine-generated text detection, paraphrase attacks, membership inference, training data extraction, data poisoning, backdoors, adaptive attacks, AI-generated text detection fairness.

### Feasibility and resources

Watermarking with small open models and injection experiments on public benchmarks are accessible with limited resources. The robustness of the defences of small models is itself a result. **Ethics:** work only in sandboxed environments or on public benchmarks, do not test real systems of third parties, and do not publish operational attack tools.

---

## Track 10. Training on Generated Text: Synthetic Data and Model Collapse

*What happens when models learn from text produced by other models, and when does it help or harm?*

### Overview

As generated text spreads on the web and is used for tuning, models increasingly learn from the output of other models. This track studies **when the use of synthetic data degrades a model** (loss of diversity, drift, disappearance of the tails of the distribution) **and when it improves it** (augmentation, distillation, self-generated instructions).

### Possible perspectives

- **Recursive training.** Fine-tuning or training over several generations.
- **Mixing real and synthetic data.** Which proportions? Replacement vs accumulation of data?
- **What degrades.** Diversity, calibration, style, bias, and the representation of minorities and less frequent languages.
- **Prevention.** Filtering and selection of synthetic data, and their effect on the degradation.
- **Contaminated corpora.** What happens when a web corpus contains generated text?
- **Distillation and self-generated data.** When does the use of a stronger model as a data source help a smaller one?

### What we expect from a project

- **A controlled and declared loop.** Number of generations, quantity of data, scheme (replacement or accumulation), with justified choices.
- **A baseline with real data at the same budget.** Without it you cannot separate the effect of synthetic data from that of the quantity or quality of data.
- **Metrics beyond perplexity.** Lexical and semantic diversity, coverage of the tails, divergence from the real distribution, and at least one measure of task utility.
- **Systematic variation.** Multiple proportions and seeds, with quantified uncertainty.
- **Discussion of realism.** Collapse often appears in extreme settings (total replacement of data). Discuss how closely your scenario resembles what happens in practice.

### Common pitfalls

- Showing the degradation in an unrealistic setting without discussing it.
- Reporting only perplexity.
- Comparing synthetic-data training with a baseline that uses a different amount of data.

### Search keywords

model collapse, synthetic data, recursive training, self-consuming generative models, data contamination, distillation, self-instruct, diversity of generated text, tail distribution, data mixing, training on AI-generated data.

### Feasibility and resources

Small models (e.g., GPT-2 sized or distilled models, or small models trained from scratch) and reduced datasets are enough. The cost grows with the number of generations, so scale it to your resources and justify it.

