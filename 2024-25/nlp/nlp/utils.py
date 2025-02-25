def messy_text(text: str):
    """Generates a messy version of a text
    """
    vocabulary = sorted(list(set(text)))
    substitutions = reversed(vocabulary)
    charmap = dict(zip(vocabulary, substitutions))
    return "".join([charmap[x] for x in text])