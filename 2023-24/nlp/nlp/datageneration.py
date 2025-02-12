"""This script exploits GPT-2 to generate a dataset with minimal variety in text
    We use data.EnglishNames to randomly select names and a list of English
    verbs from Sketch Engine to compose sentences input like:
    I am [name] and today I [verb]
"""

from tqdm import tqdm
from transformers import pipeline, set_seed
from nlp.data import EnglishNames, SketchList


def generate_corpus(
    n_docs: int,
    names: EnglishNames,
    words: SketchList,
    docs_per_seed: int = 20,
    template: str = "I am {} and today I {}",
    max_len: int = 30
) -> list:
    """Generate the corpus

    Args:
        n_docs (int): max number of docs
        names (EnglishNames): name selector
        words (SketchList): words selector
        docs_per_seed (int, optional): docs generated. Defaults to 20.
        template (str, optional): template. Defaults to "I am {} and today I {}".
        max_len (int, optional): max len for generation. Defaults to 30.

    Returns:
        list: corpus
    """
    generator = pipeline("text-generation", model="gpt2")
    set_seed(42)
    corpus = []
    pbar = tqdm(total=n_docs)
    names.stats()
    n_str = names.generate_names(n=n_docs * docs_per_seed)
    w_str = words.generate_words(n=n_docs * docs_per_seed)
    run = list(range(n_docs // docs_per_seed))
    for c in tqdm(run):
        n, w = n_str[c], w_str[c]
        seed = template.format(n, w)
        generated = generator(
            seed,
            max_length=max_len,
            num_return_sequences=docs_per_seed,
            truncation=True,
            pad_token_id=50256
        )
        pbar.update(len(generated))
        for doc in generated:
            text = doc["generated_text"].replace("\n", " ").strip()
            try:
                stop = text.rindex(".")
                corpus.append(text[:stop + 1])
            except ValueError:
                pass
    return corpus
