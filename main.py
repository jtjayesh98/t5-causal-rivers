from collections import Counter

SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]

counter = Counter()


pairs = [
    ("i love dogs", "j aime les chiens"),
    ("i love cats", "j aime les chats"),
    ("he likes dogs", "il aime les chiens"),
    ("she likes cats", "elle aime les chats"),
    ("he eats pizza", "il mange pizza"),
    ("she eats food", "elle mange nourriture"),
]

for src, tgt in pairs:
    counter.update(src.split())
    counter.update(tgt.split())

vocab = SPECIAL_TOKENS + sorted(counter.keys())

word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}

def encode(sentence):
    tokens = sentence.split()
    ids = [word2idx.get(token, word2idx["<UNK>"]) for token in tokens]
    ids.append(word2idx["<EOS>"])
    return ids



def decode(indices):
    words = []
    for idx in indices:
        word = idx2word[idx]

        if word == "<EOS>":
            break

        words.append(word)

    return " ".join(words)

def create_decoder_sequences(target):
    encoded = encode(target)

    decoder_input = [word2idx["<SOS>"]] + encoded[:-1]
    decoder_target = encoded

    return decoder_input, decoder_target

# for src, tgt in pairs:
#     src_ids = encode(src)
#     tgt_ids = encode(tgt)
#     print(f"Source: {src} -> Encoded: {src_ids}")
#     print(f"Target: {tgt} -> Encoded: {tgt_ids}")
#     print(f"Decoder Input: {create_decoder_sequences(tgt)[0]}")
#     print(f"Decoder Target: {create_decoder_sequences(tgt)[1]}")    


