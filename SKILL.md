---
name: breeze-tts2-dialogue
description: Write multi-speaker dialogue scripts as paste-ready JSON for the ComfyUI "Breeze TTS 2 Multi-Speaker" node. Use when the user provides character names, a bit about each character, and a target duration, and wants a conversation/argument/interview formatted for TTS generation.
---

# Breeze TTS 2 dialogue writer

Produce a dialogue script the user can paste straight into the `text` widget of the
**Breeze TTS 2 Multi-Speaker** ComfyUI node.

## Gather from the user

1. **Characters** — a name and a short description for each (personality, speech style,
   running gags, how they talk to each other).
2. **Duration** — target length of the spoken audio.
3. Optional: language (default English), tone (comedy, drama, interview...), and any plot
   or topic the conversation must cover.

If the user already gave all of this, do not ask follow-up questions — just write.

## Output contract

- Output **only** a raw JSON array. **No markdown code fences, no commentary, no title,
  no stage directions.** The first character of your reply must be `[`.
- Exact shape: `[{"speaker": "Name", "text": "what they say"}, ...]`
- The `speaker` value must be a character name **exactly as the user gave it** (the node
  matches names forgivingly, but exact is safest). Every character must speak at least
  2–3 times.
- `text` is one or two short spoken sentences. No quotation marks inside it — if a
  character quotes someone, paraphrase. No asterisks, no emojis, no parenthetical stage
  directions like `(angrily)` — only the vocal event tags listed below.

## Length budget

Speech runs at roughly **150 words per minute**. Compute the total word budget as
`duration_minutes × 150` and split it across 10–18 turns per minute of audio
(short turns feel like real conversation; long monologues sound robotic).

| Target | Total words | Turns |
| --- | --- | --- |
| 1 min | ~150 | 10–14 |
| 1.3 min | ~200 | 14–18 |
| 2 min | ~300 | 18–26 |
| 5 min | ~750 | 40–70 |

## Vocal event tags (the ONLY markup allowed)

Confirmed tags — use only these:

- English: `(laugh)` `(cough)` `(clears throat)` `(sigh)`
- Chinese: `[笑]` `[咳嗽]` `[清嗓子]` `[叹气]`

Place them exactly where the sound should happen, e.g. `"(laugh) You paid WHAT?"`.
They count toward the words of that turn. Use them sparingly (a handful across the whole
script) and only where the character's personality calls for it. Close variants like
`(laughing)` or `(sighs)` usually also work, but prefer the confirmed forms above.
If a reaction isn't on this list, write it as normal words ("Oh come on") instead of
inventing a tag.

## Writing rules

- Make it a real exchange: characters react to, interrupt, and build on what was just
  said. Never four disconnected monologues.
- Give each character a distinct voice driven by their description — vocabulary, rhythm,
  catchphrases, running gags. Let the running gags recur and escalate.
- Dialogue is spoken aloud: prefer natural speech over written-style prose. Numbers and
  abbreviations should be written the way they're said ("fifty grand", not "50k").
- Match the requested language. For Chinese scripts use the Chinese tag set.
- End on a beat: a punchline, a callback, or someone having the last word — not mid-thought.

## Example

User asks: "Ada (a pirate captain who treats every problem like a naval battle) argues
with Bob (a nervous first mate who over-explains everything), 60 seconds."

Correct output:

[{"speaker": "Ada", "text": "(sigh) Who steered my ship into the harbor wall this time?"}, {"speaker": "Bob", "text": "Technically, captain, the harbor steered into us. The tide charts were very clear about that."}, {"speaker": "Ada", "text": "The tide charts. (laugh) Twenty years at sea and I am outmaneuvered by paperwork."}, {"speaker": "Bob", "text": "If I may, the paperwork did warn us. Twice. In bold letters."}, ...]
