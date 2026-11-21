"""Fixed RoboCasa24 evaluation schedule (DeVA split-B objects).

24 tasks x 50 rollouts per task. Each episode has a fixed environment seed and
layout/style pair; episodes 0-9, 10-19, ... use the five layout/style pairs in
order. Success is judged only by the environment's ``_check_success``.
"""

TASK_MAX_STEPS = {
    "PnPCounterToCab": 500, "PnPCabToCounter": 500,
    "PnPCounterToSink": 700, "PnPSinkToCounter": 500,
    "PnPCounterToMicrowave": 600, "PnPMicrowaveToCounter": 500,
    "PnPCounterToStove": 500, "PnPStoveToCounter": 500,
    "OpenSingleDoor": 500, "CloseSingleDoor": 500,
    "OpenDoubleDoor": 1000, "CloseDoubleDoor": 700,
    "OpenDrawer": 500, "CloseDrawer": 500,
    "TurnOnStove": 500, "TurnOffStove": 500,
    "TurnOnSinkFaucet": 500, "TurnOffSinkFaucet": 500,
    "TurnSinkSpout": 500, "CoffeeSetupMug": 600,
    "CoffeeServeMug": 600, "CoffeePressButton": 300,
    "TurnOnMicrowave": 500, "TurnOffMicrowave": 500,
}
LAYOUT_STYLES = ((1, 1), (2, 2), (4, 4), (6, 9), (7, 10))
CAMERA_NAMES = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
EPISODES_PER_TASK = 50
CHUNK = 16
REPLAN_STEPS = 16
SETTLE_STEPS = 10


def planned_cases(ranks=1):
    if ranks < 1:
        raise ValueError("ranks must be positive")
    cases = []
    for episode in range(EPISODES_PER_TASK):
        for index, (task, max_steps) in enumerate(TASK_MAX_STEPS.items()):
            cases.append({
                "case_id": f"{task}_ep{episode:03d}", "task": task,
                "task_index": index, "episode_idx": episode,
                "env_seed": 195 * episode * 256,
                "policy_seed": 195 + index * 1000 + episode,
                "layout_style": list(LAYOUT_STYLES[episode // 10]),
                "max_steps": max_steps, "rank": (index + episode) % ranks,
            })
    return cases


def validate_evaluation_manifest(manifest, *, require_official=False):
    from thinkwam_p1.evaluation_protocol import canonical_json, validate_manifest
    validate_manifest(manifest)
    protocol = manifest["protocol"]
    from thinkwam_p1.released_checkpoint import validate_released_manifest
    validate_released_manifest(manifest)
    expected = {"benchmark": "robocasa24", "action_horizon": CHUNK,
                "replan_steps": REPLAN_STEPS, "flow_steps": 4,
                "settle_steps": SETTLE_STEPS, "camera_names": list(CAMERA_NAMES),
                "object_split": "B", "success_predicate": "native env._check_success"}
    for key, value in expected.items():
        if canonical_json(protocol.get(key)) != canonical_json(value):
            raise ValueError(f"Invalid RoboCasa24 protocol field: {key}")
    full = planned_cases(protocol["world_size"])
    if require_official and not manifest["official"]:
        raise ValueError("Debug/preflight results cannot produce an official RoboCasa24 summary")
    if manifest["official"]:
        if protocol.get("checkpoint_profile", "paper") == "paper" and manifest["checkpoint"]["step"] != 40000:
            raise ValueError("Formal RoboCasa24 evaluation requires the final step-40000 checkpoint")
        if canonical_json(manifest["cases"]) != canonical_json(full):
            raise ValueError("Official RoboCasa24 requires the fixed 24 x 50 case schedule")
    else:
        expected_by_id = {case["case_id"]: case for case in full}
        for case in manifest["cases"]:
            if canonical_json(case) != canonical_json(expected_by_id.get(case["case_id"])):
                raise ValueError("Debug RoboCasa24 case is outside the fixed schedule")
    for key in ("norm_stats_sha256", "controller_sha256"):
        import re
        if not re.fullmatch(r"[0-9a-f]{64}", protocol.get(key, "")):
            raise ValueError(f"Missing RoboCasa24 input identity: {key}")
    return manifest
