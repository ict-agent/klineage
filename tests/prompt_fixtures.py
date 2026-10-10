"""Read rendered task inputs in fake Codex runs."""

import json
import re

def prompt_inputs(prompt):
    return {
        (
            "input_kernel"
            if label == "The input kernel is"
            else label.lower().replace(" ", "_")
        ): json.loads(value)
        for label, value in re.findall(
            r"^(Problem|Repository|Expert kernel|Current kernel|Memory|"
            r"Input kernel|The input kernel is|Exclude skills): (.+)$",
            prompt,
            re.MULTILINE,
        )
    }
