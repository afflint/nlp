# A toy world and a generator of synthetic text for the LLM part of the course (L1.x onward).
#
# The design has two layers. The MEANING layer is a small world: people with a gender and four attributes
# (drink, job, city, favourite colour). A fact is a triple (person, relation, value). The SURFACE layer is a
# Language: templates that turn a fact into words. The same facts can be written in the English-like language
# "A" or in a second language "B" with other words and another word order (this gives translation), and
# the same world gives language modelling, reading from a context, and (later) questions and images.
#
# Everything is controlled by a WorldConfig, so a student can build a different world with other sizes,
# other values and a biased distribution of jobs, and generate the data again.
from dataclasses import dataclass

import numpy as np

SPECIALS = ["<pad>", "<s>", "</s>", "<unk>", "<mask>", "<sep>"]
PAD, BOS, EOS, UNK, MASK, SEP = range(6)

# the first four names are those of L0.9; female and male names are then balanced
GENDER = {"anna": "f", "marco": "m", "luca": "m", "sara": "f", "giulia": "f", "paolo": "m", "elena": "f", "diego": "m",
          "marta": "f", "mario": "m", "chiara": "f", "carlo": "m", "sofia": "f", "franco": "m", "laura": "f", "enzo": "m",
          "irene": "f", "dario": "m", "paola": "f", "gino": "m", "rosa": "f", "remo": "m", "vera": "f", "nico": "m"}
NAMES = list(GENDER)

DRINKS = ["coffee", "tea", "milk", "juice", "water", "cocoa"]
CITIES = ["rome", "milan", "turin", "naples", "genoa", "bari"]
COLORS = ["red", "blue", "green", "yellow", "pink", "brown"]
# each job has a "lean": the gender that the world (and the stereotype) associates with it, or None
JOBS = {"nurse": "f", "teacher": "f", "dancer": "f", "pilot": "m", "mechanic": "m", "farmer": "m",
        "baker": None, "clerk": None}
RELATIONS = ["drink", "job", "city", "color"]

# surface forms. {s} is the subject, {v} the value; the pronoun replaces {s} when it is the same subject
LANGUAGE_A = {
    "name": "A",
    "templates": {"drink": "{s} drinks {v}", "job": "{s} is a {v}", "city": "{s} lives in {v}", "color": "{s} likes {v}"},
    "pronouns": {"m": "he", "f": "she"},
    "question_word": {"drink": "drinks", "job": "is", "city": "lives", "color": "likes"},
}


@dataclass
class WorldConfig:
    n_people: int = 12
    n_drinks: int = 6
    n_cities: int = 6
    n_colors: int = 6
    relations: tuple = tuple(RELATIONS)
    job_bias: float = 0.0              # probability that a person gets a job that leans to their own gender
    seed: int = 0


class Vocabulary:
    # words <-> indices, with the special tokens first; unknown words go to <unk>
    def __init__(self, words):
        self.words = SPECIALS + [w for w in words if w not in SPECIALS]
        self.index = {w: i for i, w in enumerate(self.words)}

    def __len__(self):
        return len(self.words)

    def encode(self, tokens):
        return [self.index.get(t, UNK) for t in tokens]

    def decode(self, ids, skip_special=False):
        return [self.words[i] for i in ids if not (skip_special and i < len(SPECIALS))]

    def batch(self, sequences, length=None):
        # a rectangular array of indices, padded on the right with <pad>
        length = length or max(len(s) for s in sequences)
        out = np.full((len(sequences), length), PAD, dtype=np.int64)
        for i, s in enumerate(sequences):
            ids = self.encode(s)[:length]
            out[i, :len(ids)] = ids
        return out


def make_language_b(seed=0):
    # a second language with other words (pseudo-words, fixed by the seed) and another word order.
    # Names are proper nouns: they are copied. The order varies with the relation, so that the alignment
    # between the two sentences is not a copy of the order of the words.
    rng = np.random.default_rng(seed)
    onsets, vowels, codas = list("bdfgklmnprstvz"), list("aeiou"), ["", "n", "r", "s", "l"]
    used, lexicon = set(NAMES) | set(DRINKS + CITIES + COLORS + list(JOBS)), {}     # a pseudo-word never equals a word of A

    def new_word():
        while True:
            w = "".join(rng.choice(onsets) + rng.choice(vowels) + rng.choice(codas) for _ in range(2))
            if w not in used:
                used.add(w)
                return w

    for w in DRINKS + CITIES + COLORS + list(JOBS) + ["he", "she"]:
        lexicon[w] = new_word()
    verbs = {r: new_word() for r in RELATIONS}
    return {
        "name": "B",
        "lexicon": lexicon,
        "templates": {"drink": "{s} {v} " + verbs["drink"], "job": "{v} {s} " + verbs["job"],
                      "city": "{s} " + verbs["city"] + " {v}", "color": "{v} " + verbs["color"] + " {s}"},
        "pronouns": {"m": lexicon["he"], "f": lexicon["she"]},
        "question_word": verbs,
    }


class ToyWorld:
    def __init__(self, config=None, language_b=None):
        self.config = config or WorldConfig()
        c = self.config
        rng = np.random.default_rng(c.seed)
        self.people = NAMES[:c.n_people]
        self.gender = {p: GENDER[p] for p in self.people}
        self.values = {"drink": DRINKS[:c.n_drinks], "job": list(JOBS), "city": CITIES[:c.n_cities],
                       "color": COLORS[:c.n_colors]}
        self.language = {"A": LANGUAGE_A, "B": language_b or make_language_b(c.seed)}
        self.attributes = {p: {} for p in self.people}
        for p in self.people:
            for r in c.relations:
                if r == "job":
                    leaning = [j for j, g in JOBS.items() if g == self.gender[p]]
                    pool = leaning if rng.random() < c.job_bias else self.values["job"]
                    self.attributes[p][r] = str(rng.choice(pool))
                else:
                    self.attributes[p][r] = str(rng.choice(self.values[r]))

    # ---- the meaning layer
    def fact(self, person, relation):
        return (person, relation, self.attributes[person][relation])

    def facts_of(self, person):
        return [self.fact(person, r) for r in self.config.relations]

    def table(self):
        return [{"person": p, "gender": self.gender[p], **self.attributes[p]} for p in self.people]

    # ---- the surface layer
    def say(self, fact, language="A", pronoun=False):
        person, relation, value = fact
        lang = self.language[language]
        subject = lang["pronouns"][self.gender[person]] if pronoun else person
        if language != "A":
            value = lang["lexicon"].get(value, value)
        return lang["templates"][relation].format(s=subject, v=value).split() + ["."]

    def vocabulary(self, languages=("A",), extra=()):
        words = []
        for name in languages:
            lang = self.language[name]
            for r in self.config.relations:
                for v in self.values[r]:
                    words.append(lang.get("lexicon", {}).get(v, v))
                for g in "mf":
                    words.append(lang["pronouns"][g])
                for w in lang["templates"][r].split():
                    if not w.startswith("{"):
                        words.append(w)
        words += self.people + ["."] + list(extra)
        return Vocabulary(sorted(set(words)))

    # ---- the data: all the generators take a numpy Generator (or a seed) as first argument
    def _document_facts(self, rng, n_people, n_facts, interleave):
        who = [str(p) for p in rng.choice(self.people, size=n_people, replace=False)]
        groups = []
        for p in who:
            relations = rng.choice(self.config.relations, size=min(n_facts, len(self.config.relations)), replace=False)
            groups.append([self.fact(p, str(r)) for r in relations])
        facts = [f for g in groups for f in g]
        if interleave:
            facts = [facts[i] for i in rng.permutation(len(facts))]
        return facts

    def document(self, rng, n_people=2, n_facts=3, interleave=False, pronoun_rate=0.7, language="A"):
        # facts of the world about n_people people, one sentence per fact. When two consecutive sentences are
        # about the same person, the second can use a pronoun (to resolve it the model has to look back).
        rng = np.random.default_rng(rng)
        facts = self._document_facts(rng, n_people, n_facts, interleave)
        tokens = ["<s>"]
        for i, f in enumerate(facts):
            use_pronoun = i > 0 and facts[i - 1][0] == f[0] and rng.random() < pronoun_rate
            tokens += self.say(f, language, pronoun=use_pronoun)
        return tokens + ["</s>"]

    def documents(self, rng, n, **options):
        rng = np.random.default_rng(rng)
        return [self.document(rng, **options) for _ in range(n)]

    def recall(self, rng, n, items=3, relation="drink"):
        # the context says who has which value (random in each document, not the facts of the world), then asks
        # about one person: `<s> anna tea marco milk sara coffee . marco drinks milk </s>`, as in L0.9.
        # The answer is the token before </s>.
        rng = np.random.default_rng(rng)
        lang = self.language["A"]
        out = []
        for _ in range(n):
            who = [str(p) for p in rng.choice(self.people, size=items, replace=False)]
            what = [str(v) for v in rng.choice(self.values[relation], size=items)]
            asked = rng.integers(items)
            tokens = ["<s>"] + [t for p, v in zip(who, what) for t in (p, v)]
            out.append(tokens + [".", who[asked], lang["question_word"][relation], what[asked], "</s>"])
        return out

    def translation(self, rng, n, **options):
        # pairs (source in A, target in B) of the same document: same facts, same pronouns, other words and order
        rng = np.random.default_rng(rng)
        pairs = []
        for _ in range(n):
            seed = int(rng.integers(2**31))
            pairs.append((self.document(seed, language="A", **options), self.document(seed, language="B", **options)))
        return pairs
