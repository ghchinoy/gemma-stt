# Domain showcase: does prompting with domain context actually help?

`gemma-stt` lets you override the transcription prompt (`--prompt`). Because
Gemma 4 is a general multimodal LLM rather than a fixed ASR head, it can, in
principle, use a domain hint ("this is a medical consultation," "this is a
legal proceeding") to bias word choice toward domain-appropriate vocabulary.
This doc tests that claim empirically against real public audio in three
jargon-heavy domains, rather than just asserting it. Short answer: **it
sometimes helps, sometimes does nothing, and on the smaller model it once
made things measurably worse.** Details below.

## Test data

All clips are described in `tests/fixtures/domains/manifest.json` (full
ground truth, source citations, and download URLs) -- the audio itself
isn't committed to git; fetch it first with `make fixtures-domains` (see
[`../README.md`](../README.md#test-fixtures)).

| Domain | Source | License | Notes |
|---|---|---|---|
| Legal | [Oyez.org](https://www.oyez.org) API -- *Lindke v. Freed*, No. 22-611 (SCOTUS oral argument, Oct 31 2023) | Public domain (federal government recording), published openly by Oyez | Diarized, timestamped official transcript pulled from the same API |
| Medical | [PriMock57](https://github.com/babylonhealth/primock57) -- mock primary-care consultations, day 1, consultations 01 & 05 | **CC BY 4.0** (explicit `LICENSE.md`) | Professional TextGrid transcripts, doctor audio channel only |
| Financial | Oyez.org API -- *CFPB v. Community Financial Services Assn.*, No. 22-448 (SCOTUS oral argument, Oct 3 2023) | Public domain, same as Legal | **Caveat: this is a Supreme Court oral argument about federal financial regulatory law (the CFPB's funding mechanism), not a corporate earnings call.** See [Why not an earnings call?](#why-not-an-earnings-call) below. |

8 clips total, 8-30 seconds each, chosen specifically for jargon that's hard
to get right without context: a rare case name (*Adickes*), a legal term of
art (*state action*), a specific drug name (*Trimethoprim*), a clinical
diagnosis (*gastroenteritis*), a constitutional term (*Appropriations
Clause*), a federal agency acronym (*CFPB*), and repeated precise dollar
figures (*$600 million*).

## Methodology

`tests/run_domain_showcase.py` runs every clip through `gemma-stt` twice:

1. **Generic prompt** -- the CLI's own default (`"Transcribe the following
   audio verbatim..."`, no domain context).
2. **Domain-aware prompt** -- e.g. for medical: `"Transcribe the following
   audio verbatim. This is a recording of a doctor-patient medical
   consultation. Use correct clinical, anatomical, and pharmaceutical
   terminology, including exact drug names."`

Run against both E2B and E4B. Reproduce with:

```bash
make fixtures-domains   # fetch the audio (one-time)
.venv/bin/python tests/run_domain_showcase.py --model e4b
.venv/bin/python tests/run_domain_showcase.py --model e2b
# or: make showcase (runs fixtures-domains + the e4b comparison)
```

Full raw output: `tests/fixtures/domains/showcase_results_e4b.json` and
`showcase_results_e2b.json`.

## Results

Bolded text marks a meaningful difference between the generic and
domain-aware output relative to ground truth.

### Medical

| Clip | Ground truth (key phrase) | E4B generic | E4B domain-aware | E2B generic | E2B domain-aware |
|---|---|---|---|---|---|
| `medical_antibiotics` | "...with some **antibiotics** today" | "...with some **antibodies** today" (wrong) | "...with some **antibiotics** today" (**fixed**) | "...with some **antibodies** today" (wrong) | "...with some **antibodies** today" (still wrong) |
| `medical_trimethoprim` | "start the antibiotics today, something called **Trimethoprim**" | "started the antibiotics today something called **trimethoprim**" (right word, garbled surrounding text) | "start the antibiotics today, something called **trimethoprim**" (right word, more coherent) | "start the end of today something called **trimethoprin**" (misspelled, missing "antibiotics") | "start the **endoscopy** today, something called **trimetprine**" -- also inserted **"an erectile dysfunction"** out of nowhere (**worse -- new hallucinated clinical content**) |
| `medical_gastroenteritis` | "...called **gastroenteritis**..." | Correct in both | Correct in both | Correct in both | Correct in both |

### Legal

| Clip | Ground truth (key phrase) | E4B generic | E4B domain-aware | E2B generic | E2B domain-aware |
|---|---|---|---|---|---|
| `legal_intro` | "Case **22-611**" | "case **22611**" (no hyphen) | "case **22-611**" (**fixed formatting**) | "case 22611" | "case 22611" (no change) |
| `legal_adickes` | "the **Adickes** sense..." (x2) | "the **atticus** sense..." (wrong) | "the **attic** sense..." (still wrong, arguably worse) | "the **Atticus** sense..." (wrong) | "the **Attic's** sense..." (still wrong) |
| `legal_taylorswift` | "**state action**" | Correct in both | Correct in both | Correct in both | Correct in both |

### Financial

| Clip | Ground truth (key phrase) | E4B generic | E4B domain-aware | E2B generic | E2B domain-aware |
|---|---|---|---|---|---|
| `financial_600million` | "**$600 million**" (x3), "**CFPB**" | "600 million" (no `$`, x3); "**CFBB**" (wrong acronym) | "**$600 million**" (**`$` added, x3**); "CFBB" (still wrong) | "**$600 million**" (already correct); "**cfbb**" (wrong) | "$600 million" (correct); "**CFPB**" (**fixed acronym**) |
| `financial_appropriations` | "**Appropriations Clause**" | Correct in both | Correct in both | Correct in both | Correct in both |

## Takeaways

1. **Domain prompting helps most with common-word disambiguation, not rare
   proper nouns.** "Antibiotics" vs. "antibodies" and adding `$` to dollar
   figures are both cases where the model already "knows" the right word
   exists in its vocabulary and just needed a nudge on which homophone/
   convention to prefer. It did **not** fix "Adickes" (a genuinely obscure
   case name) in either model, or "Lindke"/"Kedem" (case name and attorney
   name) -- domain framing alone isn't enough for names the model has
   essentially no chance of having strong priors on.
2. **It's not free -- it can make a smaller model worse.** On E2B,
   `medical_trimethoprim`'s domain-aware run didn't just fail to fix the
   drug name, it fabricated **"erectile dysfunction"** and "endoscopy" out
   of nothing -- content not in the audio at all. This is a real
   hallucination-risk data point, not a hypothetical one: telling a smaller
   model "this is medical" without also grounding it in the actual audio
   can push it toward *plausible-sounding medical words* rather than
   *correct* ones. E4B did not show this failure mode on the same clip.
3. **Effects were inconsistent between model sizes.** E2B fixed "CFPB" with
   domain-aware prompting where E4B didn't (on the exact same clip and
   prompt wording); E2B already had `$` formatting correct in its generic
   run where E4B needed the domain-aware prompt to add it. Don't assume a
   prompting strategy validated on one model size transfers cleanly to the
   other.
4. **When it doesn't matter, it doesn't matter** -- `gastroenteritis`,
   "state action," and "Appropriations Clause" were transcribed correctly
   regardless of prompt, on both models. Domain prompting is not a
   universal accuracy lever; it's a targeted tool for specific known
   failure modes (jargon homophones, formatting conventions), and testing
   before relying on it matters more than assuming it helps.

**Practical recommendation**: if you know your audio is domain-specific and
you've observed a *specific* recurring error (like "antibiotics" ->
"antibodies"), a domain-aware `--prompt` is worth trying and cheap to test.
Don't apply it reflexively and assume it's strictly better, especially on
E2B -- verify against your own audio and watch for new hallucinated content,
not just fixed jargon.

## Why not an earnings call?

The original ask was medical / legal / **financial (earnings reports)**
specifically. During research, no source was found that was simultaneously:
(a) real corporate earnings-call audio, (b) directly downloadable without
extra tooling, and (c) cleanly licensed for use in an open test suite.
Specifically:

- **[SPGISpeech](https://huggingface.co/datasets/kensho/spgispeech)**
  (Kensho/S&P Global) is exactly this -- 5,000 hours of real earnings-call
  audio with professional transcripts -- but it's gated behind a
  click-through agreement that explicitly prohibits redistribution
  ("academic research purposes and internal use only... must not publish,
  display, transfer or redistribute the Content"). Usable for private,
  local experimentation if you personally accept Kensho's terms, but not
  something this repo can bundle or point `tests/fixtures/` at directly.
- **Federal Reserve FOMC press conferences** are public-domain government
  recordings with verbatim PDF transcripts, but the audio is embedded via a
  gated Brightcove player on federalreserve.gov (401 on direct access
  attempts), and an Archive.org C-SPAN mirror that looked promising
  (`CSPAN_20220728_100200_Federal_Reserve_Chair_Holds_News_Conference`)
  returned "Item not available" (403) on every download attempt during
  this research session -- possibly transient, worth retrying later if you
  want genuine Fed-presser audio for this domain.
- Public companies' investor-relations pages (checked Apple, Microsoft)
  reference earnings-call replays and post transcript PDFs, but stream the
  actual audio through third-party IR platforms (Q4/Notified) with no
  static, directly-fetchable audio file URL.

Given that, the financial slot uses a Supreme Court oral argument that is
*specifically about* financial regulatory law (the constitutionality of the
CFPB's funding mechanism) -- it isn't an earnings call, but it does exercise
real financial vocabulary, dollar figures, and a financial regulatory
acronym (CFPB), using the same reliable, cleanly-licensed Oyez
infrastructure as the legal domain. If you want true earnings-call coverage
later, SPGISpeech (private/local use) or retrying the FOMC/Archive.org path
are the two documented next steps.
