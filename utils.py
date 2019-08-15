def get_label(x):
    # 1 = anonymize
    # 0 = don't
    if x == "O":
        return 0
    else:
        return 1


def get_words_and_tags(sentence):
    # extracts tags and words from CONLL formatted entry
    tags = list(map(lambda word: get_label(word[1]), sentence))
    words = list(map(lambda word: word[0][0].strip().lower(), sentence))
    return words, tags
