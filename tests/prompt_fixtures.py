"""Read rendered task inputs in fake Codex runs."""

import json
import re

from kernel_fixtures import kernel, skill


def task_contexts():
    current = kernel().to_dict()
    card = skill("tile").to_dict()
    return {
        "init": {
            "problem": "/problem.json",
            "repository": "/repo",
            "expert_kernel": "kernel.cu",
        },
        "code_gen": {"current_kernel": current, "skill": card},
        "decompose": {"input_kernel": current},
        "profile": {"kernel": current, "options": {}},
        "apply": {
            "current_kernel": current,
            "skill": card,
        },
        "retrieve": {
            "current_kernel": current,
            "skills": [],
            "exclude_skills": [],
        },
    }


def prompt_inputs(prompt):
    return {
        (
            "input_kernel"
            if label == "The input kernel is"
            else label.lower().replace(" ", "_")
        ): json.loads(value)
        for label, value in re.findall(
            r"^(Problem|Repository|Expert kernel|Current kernel|Skill|Kernel|Options|"
            r"Input kernel|The input kernel is|Skills|Exclude skills): (.+)$",
            prompt,
            re.MULTILINE,
        )
    }
