"""How much does context help a tiny language model? Character n-gram models, add-one smoothing,
trained and tested on text you provide. Pure Python. Run: python3 ngram.py corpus.txt"""
import math, sys, collections

def train(text, n):
    counts = collections.Counter(); ctx = collections.Counter()
    for i in range(len(text) - n + 1):
        g = text[i:i+n]; counts[g] += 1; ctx[g[:-1]] += 1
    return counts, ctx

def bits_per_char(text, n, counts, ctx, vocab):
    total = 0.0
    for i in range(len(text) - n + 1):
        g = text[i:i+n]
        p = (counts[g] + 1) / (ctx[g[:-1]] + vocab)
        total -= math.log2(p)
    return total / (len(text) - n + 1)

def run(path):
    text = open(path, encoding="utf-8").read()
    cut = int(len(text) * .8); tr, te = text[:cut], text[cut:]
    vocab = len(set(text))
    print(f"{len(text)} chars, vocabulary {vocab}, train {len(tr)}, test {len(te)}")
    results = []
    for n in range(1, 8):
        c, x = train(tr, n); b = bits_per_char(te, n, c, x, vocab)
        results.append((n, b)); print(f"n={n}  bits/char on held-out text: {b:.3f}")
    return results

if __name__ == "__main__":
    run(sys.argv[1])
