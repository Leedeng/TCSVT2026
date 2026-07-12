"""Class-name masking for the grounding control (rebuttal R1-2).

Removes the target class-name string from each of its GRPO-refined descriptions,
so the text supervision can no longer leak the label token. The JSON keys (class
labels) are unchanged, so the contrastive target and classifier labels are intact;
only the description *text* is masked. Retraining on the masked descriptions tests
whether the gain comes from the rich visual-semantic content or from the label string.

Usage:
  python mask_class_names.py --desc_file descriptions/iMiGUE_grpo_descriptions.json \
      --mode phrase --output descriptions/iMiGUE_grpo_maskphrase.json
Modes:
  phrase : remove the full class-name phrase (mild; keeps body-part/motion words).
  word   : remove every content word of the class name (aggressive upper bound).
"""
import argparse
import json
import re

STOP = {"and", "or", "the", "a", "an", "of", "with", "to", "in", "on"}


def mask_phrase(desc, name):
    # remove the class-name phrase (with optional surrounding quotes), case-insensitive.
    # Deleting rather than substituting keeps the sentence clean, since the label is
    # usually adjacent to a generic word like "micro-gesture".
    pat = re.compile(r"['\"`]?\s*" + re.escape(name) + r"\s*['\"`]?", re.IGNORECASE)
    return pat.sub(" ", desc)


def mask_words(desc, name):
    out = desc
    for w in re.findall(r"[A-Za-z]+", name):
        if w.lower() in STOP or len(w) <= 2:
            continue
        out = re.sub(r"['\"`]?\b" + re.escape(w) + r"\b['\"`]?", " ", out, flags=re.IGNORECASE)
    return out


def cleanup(s):
    s = re.sub(r"\s+", " ", s)
    # collapse adjacent duplicate words left behind (e.g. "gesture gesture")
    s = re.sub(r"\b(\w[\w-]*)(\s+\1\b)+", r"\1", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+([,.;:])", r"\1", s)
    s = re.sub(r"([,;:])(\s*\1)+", r"\1", s)
    return s.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--desc_file", required=True)
    ap.add_argument("--mode", choices=["phrase", "word"], default="phrase")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    data = json.load(open(args.desc_file))
    out = {}
    n_hit = 0
    for cls, descs in data.items():
        masked = []
        for desc in descs:
            m = mask_phrase(desc, cls) if args.mode == "phrase" else mask_words(desc, cls)
            if m != desc:
                n_hit += 1
            masked.append(cleanup(m))
        out[cls] = masked

    total = sum(len(v) for v in data.values())
    json.dump(out, open(args.output, "w"), indent=2, ensure_ascii=False)
    print(f"[{args.mode}] masked {n_hit}/{total} descriptions -> {args.output}")


if __name__ == "__main__":
    main()
