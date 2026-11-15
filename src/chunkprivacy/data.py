"""Synthetic counterfactual corpus and structural validation."""

from __future__ import annotations

from collections import Counter
import random
import re

from researchkit.io import digest, stable_seed

PUBLIC = ["owner", "reader", "team_a", "team_b"]
VOCAB = {
    "city": {
        "train": ["Berlin", "Lima", "Oslo", "Kyoto"],
        "dev": ["Dublin", "Cairo", "Seoul", "Perth"],
        "test": ["Lisbon", "Accra", "Taipei", "Quito"],
    },
    "topic": {
        "train": ["astronomy", "botany", "ceramics", "geology"],
        "dev": ["acoustics", "robotics", "linguistics", "oceanography"],
        "test": ["photography", "entomology", "cartography", "metallurgy"],
    },
    "month": {s: ["January", "April", "July", "October"] for s in ["train", "dev", "test"]},
}
# Split membership is by template, never by rendered world. Styles 0..7 train,
# 8..11 development, 12..15 test. No label-dependent rendering or identifiers.
STYLES = [
    (
        "Project {name} is a public workshop.",
        "Its registration closes on {day}.",
        "The operations team will publish further arrangements.",
    ),
    (
        "The public seminar is named {name}.",
        "Applications for it close on {day}.",
        "Further arrangements are the responsibility of the operations team.",
    ),
    (
        "{name} designates an open training session.",
        "The enrollment deadline for this session is {day}.",
        "The operations team handles the remaining preparations.",
    ),
    (
        "An open lecture series operates under the name {name}.",
        "Signups for the series end on {day}.",
        "The operations team is in charge of practical arrangements.",
    ),
    (
        "This bulletin introduces the public course {name}.",
        "Its application window ends on {day}.",
        "Preparatory notices will come from the operations team.",
    ),
    (
        "The community session carries the title {name}.",
        "Participation requests for it are due on {day}.",
        "The operations team manages its logistics.",
    ),
    (
        "{name} is the name of the public tutorial.",
        "Registrations for this tutorial must arrive by {day}.",
        "Practical notices are issued by the operations team.",
    ),
    (
        "A workshop open to visitors is called {name}.",
        "Its booking list closes on {day}.",
        "The operations team coordinates the event arrangements.",
    ),
    (
        "Visitors can enroll in the public program {name}.",
        "The last day to enroll in it is {day}.",
        "Coordination belongs to the operations team.",
    ),
    (
        "The organizer has announced an open class named {name}.",
        "The class accepts registrations until {day}.",
        "The operations team will release the practical guidance.",
    ),
    (
        "An educational meeting titled {name} welcomes the public.",
        "Its reservation period finishes on {day}.",
        "Preparations are assigned to the operations team.",
    ),
    (
        "{name} appears in the public program as a skills session.",
        "Attendance forms for it are due by {day}.",
        "The operations team will provide organizational details.",
    ),
    (
        "Members of the public are invited to the {name} workshop.",
        "The invitation requires registration by {day}.",
        "The operations team oversees the practical preparations.",
    ),
    (
        "The schedule includes an open learning event, {name}.",
        "Its admissions form stops accepting entries on {day}.",
        "The operations team has responsibility for event logistics.",
    ),
    (
        "A publicly accessible study session has been given the name {name}.",
        "Its enrollment form is due on {day}.",
        "Organizational announcements are handled by the operations team.",
    ),
    (
        "{name} labels the new public education workshop.",
        "Its attendance list will be finalized on {day}.",
        "The operations team supervises the remaining arrangements.",
    ),
]
ANAPHORA = {
    "city": "Travel arrangements for that selected location will be announced later.",
    "topic": "Reading materials about that selected subject will be announced later.",
    "month": "Practical arrangements for that selected period will be announced later.",
}
PLAIN = {
    "city": "Travel arrangements for the workshop will be announced later.",
    "topic": "Reading materials for the workshop will be announced later.",
    "month": "Practical arrangements for the workshop will be announced later.",
}
SECRET = {
    "city": "Confidential office record: the selected location is {value}.",
    "topic": "Confidential office record: the selected subject is {value}.",
    "month": "Confidential office record: the selected period is {value}.",
}
DIMENSIONS = {
    "city": ("the destination city", "location", "travel arrangements", "travel details"),
    "topic": ("the study topic", "subject", "reading arrangements", "reading list"),
    "month": ("the event month", "period", "scheduling arrangements", "schedule"),
}
HIDDEN_TEMPLATES = [
    "Private planning note: {dimension} chosen for the event is {value}.",
    "The restricted organizer record identifies {value} as {dimension}.",
    "Internal decision register: {dimension} has been set to {value}.",
    "In the confidential planning sheet, {dimension} is recorded as {value}.",
    "An owner-only memorandum names {value} as {dimension} for this event.",
    "The private selection for {dimension} remains {value}.",
    "Restricted administrative entry: the organizer selected {value} for {dimension}.",
    "The confidential event brief specifies {dimension}: {value}.",
    "The organizers' internal minutes confirm {value} as {dimension}.",
    "Access-limited planning entry: {dimension} will be {value}.",
    "The confidential decision concerning {dimension} designates {value}.",
    "In an owner-only field, the recorded value for {dimension} is {value}.",
    "The sealed scheduling memorandum gives {value} as the choice for {dimension}.",
    "The organizers privately settled on {value} when determining {dimension}.",
    "The internal approval document assigns {value} to the field for {dimension}.",
    "The confidential briefing states that {dimension} was fixed as {value}.",
]
TARGET_TEMPLATES = [
    "The {action} for that selected {reference} will be announced later.",
    "The workshop {resource} will be included in the public notice.",
    "A later announcement will cover the {action} concerning this {reference}.",
    "The operations notice will give attendees the workshop {resource}.",
    "For that choice of {reference}, the {action} are still awaiting publication.",
    "Attendees can consult the upcoming notice for the workshop {resource}.",
    "A public update about the {action} for the indicated {reference} will follow.",
    "The event bulletin will contain the workshop {resource} after registration closes.",
    "The selected {reference} determines the {action} to be described in a later circular.",
    "A forthcoming workshop circular will provide attendees with the {resource}.",
    "The {action} associated with that {reference} will appear in the final public instructions.",
    "The organizers will distribute the workshop {resource} through a public announcement.",
    "The public follow-up will explain the {action} tied to this particular {reference}.",
    "Registered attendees will receive the workshop {resource} in a later bulletin.",
    "That agreed {reference} is the basis for the {action} in the forthcoming notice.",
    "Practical guidance for the workshop will include the {resource} once published.",
]
SMOKE_VOCAB = {
    "city": {
        "train": ["Bern", "Turin", "Baku", "Dakar"],
        "dev": ["Athens", "Riga", "Busan", "Hobart"],
        "test": ["Sofia", "Doha", "Genoa", "Nagoya"],
    },
    "topic": {
        "train": ["algebra", "weaving", "zoology", "anatomy"],
        "dev": ["hydrology", "sculpture", "mechanics", "archaeology"],
        "test": ["ecology", "drama", "optics", "mycology"],
    },
    "month": {s: ["February", "March", "May", "June"] for s in ["train", "dev", "test"]},
}


def generate(seed=20261010, families_per_template=5, templates_per_type=16):
    if templates_per_type not in (3, 16):
        raise ValueError("Use 16 templates/type for the protocol or 3 for engineering smoke")
    indices = range(16) if templates_per_type == 16 else [0, 8, 12]
    smoke = templates_per_type == 3
    names = ["Willow", "Cedar", "Maple", "Juniper", "Birch"]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    if not 1 <= families_per_template <= 5:
        raise ValueError("Expected 1..5 public instances/template")
    families = []
    for kind in VOCAB:
        for index in indices:
            split = "train" if index < 8 else "dev" if index < 12 else "test"
            template = f"{'smoke-' if smoke else ''}{kind}-{index:02d}-v3"
            for instance in range(families_per_template):
                fid = f"{template}-{instance}"
                candidates = list((SMOKE_VOCAB if smoke else VOCAB)[kind][split])
                random.Random(stable_seed(seed, fid, "candidates")).shuffle(candidates)
                intro, deadline, operations = [
                    s.format(name=names[instance], day=days[instance]) for s in STYLES[index]
                ]
                dimension, reference, action, resource = DIMENSIONS[kind]
                target = TARGET_TEMPLATES[index].format(
                    reference=reference, action=action, resource=resource
                )
                if smoke:
                    intro = (
                        f"The practice event used for software testing is called {names[instance]}."
                    )
                    deadline = f"Registration for this practice event ends on {days[instance]}."
                    operations = "A practice operations team manages the event preparation."
                    target = "Further details for this private choice are awaiting a public practice notice."
                public = [
                    {"id": "p0", "text": intro, "readers": PUBLIC},
                    {"id": "p1", "text": deadline, "readers": PUBLIC},
                    {"id": "p2", "text": target, "readers": PUBLIC},
                    {"id": "p3", "text": operations, "readers": PUBLIC},
                ]
                worlds = []
                for truth, value in enumerate(candidates):
                    hidden_text = (
                        f"Internal practice field for {dimension}: {value}."
                        if smoke
                        else HIDDEN_TEMPLATES[index].format(dimension=dimension, value=value)
                    )
                    hidden = {"id": "h0", "text": hidden_text, "readers": ["owner"]}
                    worlds.append(
                        {
                            "world_id": f"w{truth}",
                            "truth": truth,
                            "chunks": public[:2] + [hidden] + public[2:],
                        }
                    )
                families.append(
                    {
                        "family_id": fid,
                        "template_id": template,
                        "split": split,
                        "type": kind,
                        "design_id": f"{'smoke-' if smoke else ''}design-{index:02d}",
                        "instance_id": str(instance),
                        "anaphoric": smoke or index % 2 == 0,
                        "candidates": candidates,
                        "target_id": "p2",
                        "public_target_text": target,
                        "principal": "reader",
                        "worlds": worlds,
                        "utility_context": {
                            "event_name": names[instance],
                            "deadline_day": days[instance],
                        },
                        "utility": [
                            {
                                "query": "Who handles the practical arrangements?",
                                "relevant": ["p3"],
                            },
                            {
                                "query": f"When does registration close for {names[instance]}?",
                                "relevant": ["p1"],
                            },
                        ],
                    }
                )
    return families


def distractors(family, count=200):
    if count <= 0 or count % 8:
        raise ValueError("Each distractor half must balance all four candidates")
    rows = []
    for i in range(count):
        candidate = family["candidates"][i % 4]
        if i < count // 2:
            text = f"Public reference card {i}: general information about {candidate} is available in the archive."
        else:
            text = f"Public reference card {i}: routine equipment maintenance uses a standard inspection checklist."
        rows.append({"id": f"d{i:04d}", "text": text, "readers": PUBLIC})
    return rows


def validate(families, expected_main=False):
    if not families:
        raise ValueError("Empty privacy corpus")
    seen = set()
    counts = Counter()
    templates = {}
    designs = {}
    design_instances = set()
    for f in families:
        for key in ["family_id", "template_id", "design_id", "instance_id"]:
            if not isinstance(f.get(key), str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", f[key]):
                raise ValueError(f"Invalid or missing safe identifier: {key}")
        if (
            f["type"] not in VOCAB
            or f["split"] not in {"train", "dev", "test"}
            or f["principal"] != "reader"
        ):
            raise ValueError("Invalid type, split or protocol principal")
        designs.setdefault(f["design_id"], set()).add(f["split"])
        instance = (f["design_id"], f["instance_id"], f["type"])
        if instance in design_instances:
            raise ValueError("Duplicate design/instance/type")
        design_instances.add(instance)
        if f["family_id"] in seen:
            raise ValueError("Duplicate family")
        seen.add(f["family_id"])
        if (
            len(f["candidates"]) != 4
            or any(not isinstance(c, str) or not c.strip() for c in f["candidates"])
            or len({c.casefold() for c in f["candidates"]}) != 4
            or sorted(w["truth"] for w in f["worlds"]) != list(range(4))
        ):
            raise ValueError("Exactly four balanced counterfactual worlds are required")
        if len({w["world_id"] for w in f["worlds"]}) != 4:
            raise ValueError("Duplicate world ID")
        templates.setdefault(f["template_id"], set()).add(f["split"])
        public_hashes = set()
        skeletons = set()
        for w in f["worlds"]:
            chunks = w["chunks"]
            if [c["id"] for c in chunks] != ["p0", "p1", "h0", "p2", "p3"]:
                raise ValueError("Main corpus chunk order/boundaries must be fixed")
            skeletons.add(digest([(c["id"], c["readers"]) for c in chunks]))
            if len({c["id"] for c in chunks}) != len(chunks):
                raise ValueError("Duplicate chunk ID")
            for c in chunks:
                if not c["text"] or c["readers"] != sorted(set(c["readers"])):
                    raise ValueError("Empty text or noncanonical ACL")
            public = [c for c in chunks if f["principal"] in c["readers"]]
            if [c["id"] for c in public] != ["p0", "p1", "p2", "p3"]:
                raise ValueError("Protocol expects the four fixed authorized chunk IDs in order")
            secret = next((c for c in chunks if c["id"] == "h0"), None)
            if secret is None or f["principal"] in secret["readers"]:
                raise ValueError("Missing forbidden secret chunk")
            present = [
                i
                for i, c in enumerate(f["candidates"])
                if c.casefold() in secret["text"].casefold()
            ]
            if present != [w["truth"]]:
                raise ValueError("Private secret text disagrees with evaluator truth")
            if f["target_id"] not in [c["id"] for c in public]:
                raise ValueError("Target must be authorized")
            if f["target_id"] != "p2" or f["public_target_text"] != next(
                c["text"] for c in public if c["id"] == "p2"
            ):
                raise ValueError("Public target text/ID disagrees with indexed target")
            public_hashes.add(digest(public))
            for candidate in f["candidates"]:
                if any(candidate.casefold() in c["text"].casefold() for c in public):
                    raise ValueError("Candidate leaked into public target document")
        if len(public_hashes) != 1 or len(skeletons) != 1:
            raise ValueError("Counterfactual worlds must have identical authorized views")
        if len(f["utility"]) != 2:
            raise ValueError("Two controlled utility judgments are required")
        for q in f["utility"]:
            if (
                not q["query"].strip()
                or not q["relevant"]
                or len(set(q["relevant"])) != len(q["relevant"])
                or not set(q["relevant"]) <= {"p0", "p1", "p2", "p3"}
            ):
                raise ValueError("Invalid authorized utility judgment")
        context = f.get("utility_context", {})
        if (
            not context.get("event_name")
            or context["event_name"] not in public[0]["text"]
            or context.get("deadline_day")
            not in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            or context["deadline_day"] not in public[1]["text"]
        ):
            raise ValueError("Utility context metadata must match public event/deadline chunks")
        counts[(f["type"], f["split"])] += 1
    if any(len(s) != 1 for s in templates.values()):
        raise ValueError("Template crosses data splits")
    if any(len(s) != 1 for s in designs.values()):
        raise ValueError("Structural design crosses data splits")
    if expected_main and (len(families) != 240 or len(templates) != 48):
        raise ValueError("Main protocol requires 48 templates and 240 families")
    if expected_main:
        if len(designs) != 16 or any(f.get("engineering_only", False) for f in families):
            raise ValueError("Main requires 16 structural designs and no engineering-only families")
        for design in designs:
            by_design = [f for f in families if f["design_id"] == design]
            if len(by_design) != 15 or {f["type"] for f in by_design} != set(VOCAB):
                raise ValueError("Each design must have five coupled instances of all three types")
            if any(
                {f["type"] for f in by_design if f["instance_id"] == i} != set(VOCAB)
                for i in {f["instance_id"] for f in by_design}
            ):
                raise ValueError("Design instances must couple all attribute types")
        for kind in VOCAB:
            for split_name, count in [("train", 40), ("dev", 20), ("test", 20)]:
                if counts[(kind, split_name)] != count:
                    raise ValueError("Incorrect type/partition family allocation")
        per_template = Counter(f["template_id"] for f in families)
        if set(per_template.values()) != {5}:
            raise ValueError("Main protocol needs exactly five families/template")
    for kind in ["city", "topic"]:
        pools = {
            s: {
                c
                for f in families
                if f["type"] == kind and f["split"] == s
                for c in f["candidates"]
            }
            for s in ["train", "dev", "test"]
        }
        if any(
            pools[a] & pools[b] for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]
        ):
            raise ValueError("City/topic candidate vocabulary crosses partitions")
    return {
        "families": len(families),
        "worlds": len(families) * 4,
        "templates": len(templates),
        "structural_designs": len(designs),
        "counts": {f"{k[0]}/{k[1]}": v for k, v in counts.items()},
    }
