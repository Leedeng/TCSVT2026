"""Translate description strings to Chinese via OpenAI gpt-4o-mini.

Inputs (any subset):
  --orig_json     descriptions/<dataset>_descriptions.json   (class -> [descs])
  --grpo_json     descriptions/<dataset>_grpo_descriptions.json
  --vlm_json      captions/<dataset>_<model>.json            (clip -> caption)

Output:
  One JSON: {"<original_text>": "<chinese_text>"} for every unique string seen.
  Resumable: if the output exists, already-translated strings are skipped.

Usage:
  export OPENAI_API_KEY=sk-...
  python translate_descs.py --inputs descriptions/iMiGUE_descriptions.json \
      descriptions/iMiGUE_grpo_descriptions.json \
      captions/iMiGUE_qwen.json captions/iMiGUE_gemma.json \
      --out experiment/translations/iMiGUE_zh.json
"""
import argparse
import json
import os
import sys
import time

from openai import OpenAI


SYSTEM = (
    "You are a precise English-to-Chinese translator for academic descriptions "
    "of micro-gestures (subtle body movements). Translate the user's text into "
    "fluent simplified Chinese. Preserve technical terms (body parts, motions). "
    "Reply with ONLY the Chinese translation, no quotation marks, no preface."
)


def collect_strings(paths):
    seen = set()
    for path in paths:
        if not os.path.exists(path):
            print(f"[warn] missing {path}", file=sys.stderr)
            continue
        data = json.load(open(path))
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    for s in v:
                        if isinstance(s, str) and s.strip():
                            seen.add(s.strip())
                elif isinstance(v, str) and v.strip():
                    seen.add(v.strip())
        elif isinstance(data, list):
            for s in data:
                if isinstance(s, str) and s.strip():
                    seen.add(s.strip())
    return sorted(seen)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--sleep_on_429", type=float, default=2.0)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY is not set")
    client = OpenAI(api_key=api_key)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cache = {}
    if os.path.exists(args.out):
        cache = json.load(open(args.out))
        print(f"Loaded cache with {len(cache)} translations")

    strings = collect_strings(args.inputs)
    todo = [s for s in strings if s not in cache]
    print(f"Total unique strings: {len(strings)}; to translate: {len(todo)}", flush=True)

    for i, s in enumerate(todo):
        for attempt in range(args.max_retries):
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": s},
                    ],
                    temperature=0.0,
                )
                cache[s] = resp.choices[0].message.content.strip()
                break
            except Exception as e:
                msg = str(e)
                if "rate" in msg.lower() and attempt < args.max_retries - 1:
                    print(f"  rate-limited; sleeping {args.sleep_on_429}s ...", flush=True)
                    time.sleep(args.sleep_on_429)
                    continue
                print(f"  [{i+1}/{len(todo)}] FAIL: {e}", flush=True)
                cache[s] = "(translation failed)"
                break

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(todo)}] {s[:60]} -> {cache[s][:60]}", flush=True)
        if (i + 1) % 25 == 0:
            with open(args.out, "w") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)

    with open(args.out, "w") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(cache)} translations -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
