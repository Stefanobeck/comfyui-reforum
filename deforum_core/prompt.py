"""Deforum prompts: keyframed prompts, maths inside backticks, '--neg', A1111 'AND' composition.

Port of the Deforum extension for AUTOMATIC1111 (MIT, commit a7b829e): render.py (prompt_series), prompt.py
(prepare_prompt, split_weighted_subprompts) and args.py (positive/negative prompts added to every frame).
"""
import re

from .expr import evaluate
from .keyframes import check_is_number

_BACKTICK = re.compile(r"`.*?`")


def build_prompt_series(animation_prompts, max_frames, positive="", negative="", seed=-1):
    """{"0": "...", "30": "..."} -> one prompt per frame. A keyframe holds until the next one (no blending: this is
    what Deforum's render loop did). The global positive/negative prompts are added as in args.py."""
    negative = negative.replace("--neg", "")
    prompts = {key: f"{positive} {val} {'' if '--neg' in val else '--neg'} {negative}"
               for key, val in animation_prompts.items()}
    series = [None] * max_frames
    for key, prompt in prompts.items():
        key = str(key).strip()
        if key.isdigit():
            idx = int(key)
        else:
            idx = int(evaluate(key.strip('"').strip("'"), max_f=max_frames - 1, s=seed))
        if 0 <= idx < max_frames:
            series[idx] = prompt
    last = next((p for p in series if p is not None), "")
    out = []
    for p in series:                      # ffill, then bfill (the first value goes back to frame 0)
        last = p if p is not None else last
        out.append(last)
    return out


def blend_prompt_series(animation_prompts, max_frames, positive="", negative="", seed=-1, blend_frames=0):
    """Like build_prompt_series, plus a crossfade of blend_frames frames before every keyframe (an extension of this
    port; 0 = Deforum's hard cut). In the crossfade both prompts go in with A1111's composable syntax
    'old :w1 AND new :w2', the weights moving linearly from the old to the new prompt: a morph, not a jump."""
    series = build_prompt_series(animation_prompts, max_frames, positive, negative, seed)
    blend_frames = int(blend_frames)
    if blend_frames <= 0:
        return series
    keys = sorted({i for i in range(1, max_frames) if series[i] != series[i - 1]})
    out = list(series)
    for k in keys:
        old, new = series[k - 1], series[k]
        old_pos, _, old_neg = old.partition("--neg")
        new_pos, _, new_neg = new.partition("--neg")
        start = max(k - blend_frames, 0)
        for f in range(start, k):
            if out[f] != old:                                 # an earlier crossfade already wrote here
                continue
            w = (f - start + 1) / (k - start + 1)
            neg = new_neg if w >= 0.5 else old_neg
            out[f] = f"{old_pos.strip()} :{1 - w:.3f} AND {new_pos.strip()} :{w:.3f} --neg {neg.strip()}"
    return out


def prepare_prompt(prompt, max_frames, frame_idx):
    """Evaluates the `maths` in the prompt (t = frame, max_f = last frame) and returns (positive, negative)."""
    def repl(match):
        return str(evaluate(match.group(0)[1:-1], t=frame_idx, max_f=max_frames - 1))
    parsed = _BACKTICK.sub(repl, prompt)
    positive, *after = parsed.strip().split("--neg")
    return positive.strip(), "".join(after).strip()


_AND = re.compile(r"\bAND\b")
_WEIGHT = re.compile(r"\s*:\s*([-+]?(?:\d+\.?|\d*\.\d+))\s*$")


def split_and(prompt):
    """A1111 composable diffusion: 'a cat :1.2 AND a dog :0.8' -> [('a cat', 1.2), ('a dog', 0.8)]."""
    parts = []
    for chunk in _AND.split(prompt):
        m = _WEIGHT.search(chunk)
        weight = float(m.group(1)) if m else 1.0
        text = chunk[:m.start()] if m else chunk
        parts.append((text.strip(), weight))
    return parts or [("", 1.0)]
