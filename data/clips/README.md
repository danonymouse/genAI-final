# Clips and labels: how to

**1. Add the clips.** One file per clip, `ko_01.txt` … `ko_04.txt`, `zh_01.txt` … `zh_04.txt`.
One utterance per line (about 20 lines), copied from the transcript. Spoken content (vlogs,
casual talk, comedy), not captions.
Do not use any text that lexicon entries were drafted from (e.g. Tina's sample): see decision L16.

**2. Fix the split before anyone labels** (decision D11). Create `data/clips/split.json`:

```json
{"dev": ["ko_01", "ko_02", "zh_01", "zh_02"], "test": ["ko_03", "ko_04", "zh_03", "zh_04"]}
```

Commit it. Tune only on `dev`; every reported number comes from `test`.

**3. Make a labelling sheet per clip:**

```
python scripts/labels.py template data/clips/ko_01.txt     # -> data/gold/ko_01.csv
```

Open it in Excel or Google Sheets. One row per line:
- nothing a learner would misread → leave `surface` empty;
- a hard span → type it exactly as written into `surface`, choose `type`
  (slang, idiom, discourse_marker, register, non_literal, wordplay), optional `note`;
- two spans on one line → duplicate the row; same phrase twice → write `phrase#2`.

**4. Convert and check:** `python scripts/labels.py convert data/gold/ko_01.csv`
(it reports any phrase it can't find in the line, or a misspelled type).

**5. Score:** `python scripts/score.py detect --split dev` compares detector settings;
`python scripts/score.py sheet --split test` makes the grading sheet for the two graders.
