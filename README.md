# Does more context always help?

A small experiment with character n-gram models: predict the next character from the previous n-1 characters, with add-one smoothing. Train on 80% of a text, measure bits per character on the held-out 20%. Lower is better.

```sh
python3 ngram.py corpus.txt
```

## Result (this run, saved in `results.txt`)
| n | bits per char |
| --- | --- |
| 1 | 4.818 |
| 2 | 3.977 |
| 3 | 3.406 |
| 4 | 3.412 |
| 5 | 3.847 |
| 6 | 4.269 |
| 7 | 4.604 |

More context helps up to n=3 or 4, then gets worse. Longer contexts are rarely seen in the training text, and add-one smoothing hands too much probability to combinations never seen.

## What this does not show
- One corpus (`corpus.txt`, about 320,000 characters of Python standard-library documentation text), one split, no repeats. The turning point may move with other text.
- Add-one smoothing is crude. Better smoothing (for example interpolation) would very likely move the curve. I did not try it.
- This is a toy, not a claim about large language models.

Python 3, standard library only. The run is deterministic.
