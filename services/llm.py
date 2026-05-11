import base64
from pathlib import Path

import anthropic

from config import LLM_MODEL

_SYSTEM = (
    "You are a quality control inspector on a bottle manufacturing line. "
    "An operator has requested analysis of a specific flagged bottle."
)

_USER_TEMPLATE = """\
Session context:
- Overall pass rate: {pass_rate}%
- Total defects this session: {rejected}
- Last 10 bottles: {recent_summary}

Examine this bottle image and provide a concise 3-4 sentence report covering:
1. What defect or marking you can see and where it is on the bottle
2. Severity — minor cosmetic, moderate, or critical
3. Whether this looks like an isolated incident or part of a pattern \
given the session context above
4. Your recommendation: reject, re-inspect, or pass\
"""


def analyse_bottle(bottle_id: int, image_path: str | Path, twin_state) -> str:
    """Call Claude with the saved bottle ROI + session context. Returns report text."""
    state = twin_state.to_dict()

    last_10 = state["bottles"][-10:]
    n = len(last_10)
    defects = sum(1 for b in last_10 if not b["passed"])
    recent_summary = f"{defects}/{n} defective" if n else "no data yet"

    prompt = _USER_TEMPLATE.format(
        pass_rate=state["pass_rate"],
        rejected=state["rejected"],
        recent_summary=recent_summary,
    )

    image_b64 = base64.standard_b64encode(Path(image_path).read_bytes()).decode()

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=LLM_MODEL,
        max_tokens=512,
        system=_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return msg.content[0].text
