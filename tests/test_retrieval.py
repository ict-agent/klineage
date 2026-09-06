import unittest
from dataclasses import replace

from test_action import accepted, kernel, skill

from klineage.contract import ABIValue, KernelABI, ProblemSpec
from klineage.kernel import Feature, Kernel, TargetContext
from klineage.memory import Lineage, Scope, SkillAdmission, SkillCard, retrieve_paths


TILE = Feature("gemm.main", "shared_memory_tiling")
MMA = Feature("gemm.main", "mma")
PIPELINE = Feature("gemm.main", "pipeline")


def _abi(size=128, dtype="float16", layout="row_major"):
    return KernelABI(inputs=(ABIValue(
        "a", dtype=dtype, shape=(size, size), constraints={"layout": layout},
    ),))


def _lineage(size=128, abi=None):
    states = tuple(replace(
        kernel(source=f"// state {index}"),
        features=features,
        problem=ProblemSpec("gemm", "Compute matrix multiplication."),
        abi=abi or _abi(size),
        validation=accepted(4.0 / (index + 1)),
    ) for index, features in enumerate(((), (TILE,), (TILE, MMA), (TILE, MMA, PIPELINE))))
    cards = []
    for index, feature in enumerate((TILE, MMA, PIPELINE)):
        card = skill(feature.name, None)
        edge = replace(
            card.evidence[0], locus=feature.locus,
            before_fingerprint=states[index].fingerprint,
            after_fingerprint=states[index + 1].fingerprint,
        )
        cards.append(replace(
            card, evidence=(edge,), provides=(feature,),
            requires=() if index == 0 else ((TILE, MMA, PIPELINE)[index - 1],),
        ))
    return Lineage(states, tuple(card.evidence[0] for card in cards), tuple(cards))


class RetrievalTests(unittest.TestCase):
    def test_contracts_roundtrip(self):
        lineage = _lineage()
        self.assertEqual(Lineage.from_dict(lineage.to_dict()), lineage)
        raw = lineage.skills[0].to_dict()
        raw.pop("requires")
        raw.pop("provides")
        raw.pop("conflicts")
        self.assertEqual(SkillCard.from_dict(raw).provides, ())
        self.assertEqual(Kernel.from_dict(kernel().to_dict()).features, ())

    def test_edges_determine_order(self):
        lineage = _lineage()
        shuffled = replace(lineage, skills=tuple(reversed(lineage.skills)))
        plan, = retrieve_paths((shuffled,), lineage.states[0])
        self.assertEqual(plan.skills, lineage.skills)
        self.assertEqual(plan.entry_index, 0)

    def test_enters_after_mma(self):
        lineage = _lineage()
        current = replace(lineage.states[2], source="// another implementation")
        plan, = retrieve_paths((lineage,), current)
        self.assertEqual(plan.entry_index, 2)
        self.assertEqual(plan.skills, (lineage.skills[2],))

    def test_actions_are_not_state(self):
        lineage = _lineage()
        context = TargetContext("gemm", "cuda", "sm120", ("tile", "mma"))
        plan, = retrieve_paths((lineage,), context, abi=_abi())
        self.assertEqual(plan.skills[0], lineage.skills[0])

    def test_requirements_bind_locus(self):
        lineage = _lineage()
        isolated = replace(lineage, skills=(lineage.skills[-1],))
        current = replace(lineage.states[0], features=(Feature("other.loop", "mma"),))
        self.assertEqual(retrieve_paths((isolated,), current), ())

    def test_conflicts_stop_path(self):
        lineage = _lineage()
        blocked = replace(lineage.skills[1], conflicts=(TILE,))
        lineage = replace(lineage, skills=(lineage.skills[0], blocked, lineage.skills[2]))
        plan, = retrieve_paths((lineage,), lineage.states[0])
        self.assertEqual(plan.skills, (lineage.skills[0],))

    def test_abi_and_shape_rank(self):
        near, far = _lineage(128), _lineage(4096)
        target = replace(near.states[0], abi=_abi(192))
        plans = retrieve_paths((far, near), target)
        self.assertEqual(plans[0].lineage, near)
        for abi in (_abi(dtype="float32"), _abi(layout="column_major")):
            self.assertEqual(retrieve_paths((near,), replace(target, abi=abi)), ())

    def test_stride_axis_order(self):
        source_abi = KernelABI(inputs=(ABIValue(
            "a", dtype="float16", shape=(128, 128),
            constraints={"contiguous": False, "stride": [256, 2]},
        ),))
        lineage = _lineage(abi=source_abi)
        target = lineage.states[0]
        for stride in ([2, 256], [2, 2], None):
            constraints = {"contiguous": False}
            if stride is not None:
                constraints["stride"] = stride
            abi = KernelABI(inputs=(replace(source_abi.inputs[0], constraints=constraints),))
            self.assertFalse(retrieve_paths((lineage,), replace(target, abi=abi)))

        larger = KernelABI(inputs=(replace(
            source_abi.inputs[0], shape=(256, 256),
            constraints={"contiguous": False, "stride": [512, 2]},
        ),))
        self.assertTrue(retrieve_paths((lineage,), replace(target, abi=larger)))

    def test_hardware_and_case_filter(self):
        lineage = _lineage()
        for context in (TargetContext("gemm", "cuda", "sm90"),
                        TargetContext("topk", "cuda", "sm120")):
            self.assertEqual(retrieve_paths((lineage,), context), ())

    def test_new_case_needs_no_source(self):
        lineage = _lineage()
        plan, = retrieve_paths(
            (lineage,), TargetContext("gemm", "cuda", "sm120"),
            problem=ProblemSpec("new_gemm", "Compute matrix multiplication."),
            abi=_abi(256),
        )
        self.assertEqual(len(plan.skills), 3)

    def test_seeds_partial_lineage(self):
        full = _lineage()
        partial = replace(
            full, states=full.states[2:], transitions=full.transitions[2:],
            skills=full.skills[2:],
        )
        context = TargetContext("gemm", "cuda", "sm120")
        plan, = retrieve_paths((partial,), context, abi=_abi())
        self.assertEqual(plan.lineage.states[plan.entry_index].features, (TILE, MMA))
        self.assertEqual(plan.skills, partial.skills)
        self.assertEqual(retrieve_paths((partial,), full.states[0]), ())

    def test_legacy_cards_not_paths(self):
        lineage = _lineage()
        legacy = replace(lineage, skills=tuple(
            replace(card, requires=(), provides=()) for card in lineage.skills
        ))
        self.assertEqual(retrieve_paths((legacy,), lineage.states[0]), ())

    def test_binds_edge_evidence(self):
        lineage = _lineage()
        unrelated = replace(lineage.transitions[0], before_fingerprint="other")
        merged = replace(lineage.skills[0], evidence=(unrelated, lineage.transitions[0]))
        lineage = replace(lineage, skills=(merged, *lineage.skills[1:]))
        plan, = retrieve_paths((lineage,), lineage.states[0])
        self.assertEqual(plan.skills[0].evidence, (lineage.transitions[0],))

    def test_no_cross_lineage_bridge(self):
        lineage = _lineage()
        first = replace(lineage, skills=(lineage.skills[0],))
        second = replace(lineage, skills=lineage.skills[1:])
        plan, = retrieve_paths((first, second), lineage.states[0])
        self.assertEqual(plan.lineage, first)
        self.assertEqual(plan.skills, (lineage.skills[0],))

    def test_scope_and_admission_gate(self):
        lineage = _lineage()
        self.assertEqual(retrieve_paths(
            (lineage,), lineage.states[0], skill_admission=SkillAdmission.ON,
        ), ())
        wrong_scope = replace(lineage.skills[0], scope=Scope(platforms=("sm90",)))
        lineage = replace(lineage, skills=(wrong_scope, *lineage.skills[1:]))
        self.assertEqual(retrieve_paths((lineage,), lineage.states[0]), ())

    def test_explicit_capabilities(self):
        lineage = _lineage()
        hardware = Feature("hardware", "mma.sync")
        tile = replace(lineage.skills[0], requires=(hardware,))
        lineage = replace(lineage, skills=(tile, *lineage.skills[1:]))
        context = TargetContext("gemm", "cuda", "sm120")
        target = replace(lineage.states[0], context=context)
        self.assertEqual(retrieve_paths((lineage,), target), ())
        plan, = retrieve_paths(
            (lineage,), replace(target, context=replace(context, capabilities=("mma.sync",))),
        )
        self.assertEqual(len(plan.skills), 3)


if __name__ == "__main__":
    unittest.main()
