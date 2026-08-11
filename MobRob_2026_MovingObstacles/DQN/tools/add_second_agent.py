#!/usr/bin/env python3
"""Add a second agent + second target to the Unity Training_arena scene.

The Unity project currently holds a single Agent (behavior "agent_navigation") and a
single Target. This script duplicates the entire Agent GameObject subtree (ML-Agents
Agent, BehaviorParameters, DecisionRequester, RayPerceptionSensor3D, Rigidbody,
colliders, the embedded TurtleBot model, ...) and the whole Target object, remapping
every scene-local fileID so the copies are fully independent objects. It then:

  * renames the copy "Agent2" / "Target2",
  * gives Agent2 its own behavior name "agent_navigation_2",
  * points Agent2's CustomAgent.targetName at "Target2",
  * offsets Agent2 and Target2 to a distinct starting spot,
  * registers both new root Transforms in the SceneRoots block.

Everything else (physics layers, tags, sensor settings, materials) is inherited from
the originals, so the second agent behaves exactly like the first.

Usage:
    python tools/add_second_agent.py

The scene file is modified in place. A backup is left next to it (Training_arena.unity.bak)
if the backup does not already exist.
"""

import os
import re
import shutil
import sys

SCENE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "turtlebot3UnityDQN", "Assets", "Scenes", "Training_arena.unity",
)
BACKUP_PATH = SCENE_PATH + ".bak"

# Behavior name for the second agent (the first keeps "agent_navigation").
AGENT2_BEHAVIOR = "agent_navigation_2"
# Distinct starting spot for the second agent and its target.
AGENT2_POSITION = {"x": 0.0, "y": 0.09, "z": 1.2}
TARGET2_POSITION = {"x": 1.4, "y": 0.0, "z": 0.0}

# CustomAgent MonoBehaviour script guid (matches Assets/Scripts/CustomAgent.cs).
CUSTOM_AGENT_GUID = "8393f484be145c64d880c4d26fb3ad63"
# BehaviorParameters MonoBehaviour script guid (ml-agents).
BEHAVIOR_PARAMETERS_GUID = "5d1c4e0b1822b495aa52bc52839ecb30"

HEADER_RE = re.compile(r"^--- !u!(?P<type>\d+) &(?P<fid>\d+)\s*$")
LOCAL_REF_RE = re.compile(r"\{fileID: (?P<fid>\d+)\}")


def read_blocks(text):
    """Split the scene YAML into {fileid: (unity_type, body_lines)}.

    Also returns the file header (the %YAML / %TAG directives that precede the
    first '---' block marker) so it can be preserved on write -- Unity needs those
    directives and treats the file as corrupt without them.
    """
    lines = text.splitlines()
    header = []
    i = 0
    while i < len(lines) and not lines[i].startswith("---"):
        header.append(lines[i])
        i += 1
    blocks = {}
    current = None
    current_body = []

    def flush(fid, unity_type, body):
        blocks[fid] = (unity_type, body)

    for line in lines[i:]:
        m = HEADER_RE.match(line)
        if m:
            if current is not None:
                flush(current[0], current[1], current_body)
            current = (int(m.group("fid")), int(m.group("type")))
            current_body = []
        elif current is not None:
            current_body.append(line)
    if current is not None:
        flush(current[0], current[1], current_body)

    if not blocks:
        raise SystemExit(f"No scene blocks parsed from {SCENE_PATH}")
    return blocks, header


def local_refs(body):
    """All scene-local fileIDs referenced from a block's body."""
    return {int(m.group("fid")) for m in LOCAL_REF_RE.finditer("\n".join(body))}


def find_root_go(blocks, name):
    """Return the fileID of the GameObject named exactly `name` that is a scene root
    (its Transform has m_Father: {fileID: 0})."""
    for fid, (unity_type, body) in blocks.items():
        if unity_type != 1:
            continue
        joined = "\n".join(body)
        if re.search(rf"^\s*m_Name: {re.escape(name)}$", joined, re.M) and _is_root_object(blocks, fid):
            return fid
    raise SystemExit(f"Root GameObject named '{name}' not found in scene")


def _is_root_object(blocks, go_fid):
    unity_type, body = blocks[go_fid]
    joined = "\n".join(body)
    for m in LOCAL_REF_RE.finditer(joined):
        ref = int(m.group("fid"))
        if ref in blocks:
            ref_type, ref_body = blocks[ref]
            if ref_type == 4:  # Transform
                ref_joined = "\n".join(ref_body)
                if f"m_GameObject: {{fileID: {go_fid}}}" in ref_joined:
                    return "m_Father: {fileID: 0}" in ref_joined
    return False


def closure(blocks, seed):
    """All blocks transitively referenced from `seed`, via scene-local fileIDs."""
    seen = set()
    stack = [seed]
    while stack:
        fid = stack.pop()
        if fid in seen:
            continue
        seen.add(fid)
        unity_type, body = blocks[fid]
        for ref in local_refs(body):
            if ref in blocks and ref not in seen:
                stack.append(ref)
    return seen


def pick_new_ids(blocks, amount, start=5_000_000_000):
    existing = set(blocks.keys())
    chosen = []
    candidate = start
    while len(chosen) < amount:
        if candidate not in existing:
            chosen.append(candidate)
        candidate += 1
    return chosen


def remap_body(body, id_map):
    """Rewrite every 'fileID: N' where N is in id_map to its replacement."""
    if not id_map:
        return list(body)
    numbers = sorted(id_map.keys(), key=lambda n: -len(str(n)))
    out = []
    for line in body:
        for old in numbers:
            if f"fileID: {old}" in line:
                line = line.replace(f"fileID: {old}", f"fileID: {id_map[old]}")
        out.append(line)
    return out


def set_field(body, field, value):
    """Replace `field: old` on its own line with `field: value` (first occurrence)."""
    out = list(body)
    pattern = re.compile(rf"^(\s*){re.escape(field)}:.*$")
    for i, line in enumerate(out):
        m = pattern.match(line)
        if m:
            out[i] = f"{m.group(1)}{field}: {value}"
            return out
    raise SystemExit(f"Field '{field}' not found in block body")


def insert_after_field(body, field, line_to_insert):
    """Insert a new serialized line right after the first `field:` occurrence."""
    out = list(body)
    pattern = re.compile(rf"^(\s*){re.escape(field)}:.*$")
    for i, line in enumerate(out):
        m = pattern.match(line)
        if m:
            out.insert(i + 1, f"{m.group(1)}{line_to_insert}")
            return out
    raise SystemExit(f"Field '{field}' not found in block body")


def main():
    if not os.path.isfile(SCENE_PATH):
        raise SystemExit(f"Scene not found: {SCENE_PATH}")

    if not os.path.exists(BACKUP_PATH):
        shutil.copy2(SCENE_PATH, BACKUP_PATH)
        print(f"Backup written to {BACKUP_PATH}")

    text = open(SCENE_PATH, "r", encoding="utf-8").read()
    blocks, yaml_header = read_blocks(text)

    # Refuse to run twice: both clones must not already exist.
    joined = text
    for name in ("Agent2", "Target2"):
        if f"m_Name: {name}" in joined:
            raise SystemExit(f"'{name}' already present in the scene, aborting")

    agent_go = find_root_go(blocks, "Agent")
    target_go = find_root_go(blocks, "Target")
    print(f"Found Agent GO {agent_go} and Target GO {target_go}")

    agent_closure = closure(blocks, agent_go)
    target_closure = closure(blocks, target_go)
    print(f"Agent subtree: {len(agent_closure)} blocks, "
          f"Target subtree: {len(target_closure)} blocks")

    # Sanity check: no block outside a subtree may reference it, and the two
    # subtrees must be disjoint. The SceneRoots block is exempt (it legitimately
    # lists the root Transform of every scene object).
    if agent_closure & target_closure:
        raise SystemExit("Agent and Target subtrees overlap, aborting")
    for fid, (unity_type, body) in blocks.items():
        if fid in agent_closure or fid in target_closure:
            continue
        if unity_type == 1660057539:  # SceneRoots
            continue
        refs = local_refs(body)
        if refs & agent_closure:
            raise SystemExit(f"Block {fid} references the Agent subtree from outside")
        if refs & target_closure:
            raise SystemExit(f"Block {fid} references the Target subtree from outside")

    # Fresh unique ids for the two cloned subtrees.
    agent_map = dict(zip(sorted(agent_closure), pick_new_ids(blocks, len(agent_closure))))
    target_map = dict(zip(sorted(target_closure), pick_new_ids(blocks, len(target_closure), start=6_000_000_000)))

    agent2_go = agent_map[agent_go]
    target2_go = target_map[target_go]

    # Locate the agent's CustomAgent + BehaviorParameters + root Transform inside the closure.
    agent_custom = None
    agent_behavior = None
    agent_root_transform = None
    for fid in agent_closure:
        unity_type, body = blocks[fid]
        joined = "\n".join(body)
        if unity_type == 114 and f"guid: {CUSTOM_AGENT_GUID}" in joined:
            agent_custom = fid
        if unity_type == 114 and f"guid: {BEHAVIOR_PARAMETERS_GUID}" in joined:
            agent_behavior = fid
        if unity_type == 4 and f"m_GameObject: {{fileID: {agent_go}}}" in joined and "m_Father: {fileID: 0}" in joined:
            agent_root_transform = fid
    if agent_custom is None or agent_behavior is None or agent_root_transform is None:
        raise SystemExit("Could not locate CustomAgent / BehaviorParameters / root Transform of the Agent")

    target_root_transform = None
    for fid in target_closure:
        unity_type, body = blocks[fid]
        joined = "\n".join(body)
        if unity_type == 4 and f"m_GameObject: {{fileID: {target_go}}}" in joined and "m_Father: {fileID: 0}" in joined:
            target_root_transform = fid
    if target_root_transform is None:
        raise SystemExit("Could not locate the root Transform of the Target")

    # Build the duplicated blocks.
    new_blocks = []

    for fid in sorted(agent_closure):
        unity_type, body = blocks[fid]
        new_fid = agent_map[fid]
        new_body = remap_body(body, agent_map)
        if fid == agent_go:
            new_body = set_field(new_body, "m_Name", "Agent2")
        if fid == agent_custom:
            # New serialized field (added to CustomAgent.cs): insert right after
            # useContinuousActions to match the C# field declaration order.
            new_body = insert_after_field(new_body, "useContinuousActions", "targetName: Target2")
        if fid == agent_behavior:
            new_body = set_field(new_body, "m_BehaviorName", AGENT2_BEHAVIOR)
        if fid == agent_root_transform:
            new_body = set_field(new_body, "m_LocalPosition",
                                 f"{{x: {AGENT2_POSITION['x']}, y: {AGENT2_POSITION['y']}, z: {AGENT2_POSITION['z']}}}")
        new_blocks.append((new_fid, unity_type, new_body))

    for fid in sorted(target_closure):
        unity_type, body = blocks[fid]
        new_fid = target_map[fid]
        new_body = remap_body(body, target_map)
        if fid == target_go:
            new_body = set_field(new_body, "m_Name", "Target2")
        if fid == target_root_transform:
            new_body = set_field(new_body, "m_LocalPosition",
                                 f"{{x: {TARGET2_POSITION['x']}, y: {TARGET2_POSITION['y']}, z: {TARGET2_POSITION['z']}}}")
        new_blocks.append((new_fid, unity_type, new_body))

    # Register the two new root Transforms in the SceneRoots block.
    scene_roots = None
    for fid, (unity_type, body) in blocks.items():
        if unity_type == 1660057539:
            scene_roots = (fid, body)
            break
    if scene_roots is None:
        raise SystemExit("SceneRoots block not found")

    roots_fid, roots_body = scene_roots
    roots_out = [ln for ln in roots_body if ln.strip() != "" or ln == ""][:]
    while roots_out and roots_out[-1].strip() == "":
        roots_out.pop()
    # Append after the last root entry (keep each entry on its own line).
    roots_out.append(f"  - {{fileID: {agent_map[agent_root_transform]}}}")
    roots_out.append(f"  - {{fileID: {target_map[target_root_transform]}}}")

    # Write the final scene: original blocks + new blocks, then the (patched) SceneRoots.
    # Preserve the %YAML/%TAG header -- Unity rejects the file as corrupt without it.
    out_lines = [ln for ln in yaml_header if ln.strip() != ""]
    for fid in sorted(blocks.keys()):
        unity_type, body = blocks[fid]
        if unity_type == 1660057539:
            continue  # handled at the end
        out_lines.append(f"--- !u!{unity_type} &{fid}")
        out_lines.extend(body)

    for new_fid, unity_type, body in sorted(new_blocks):
        out_lines.append(f"--- !u!{unity_type} &{new_fid}")
        out_lines.extend(body)

    out_lines.append(f"--- !u!1660057539 &9223372036854775807")
    out_lines.extend(roots_out)

    new_text = "\n".join(out_lines) + "\n"

    # ---- Validation -----------------------------------------------------------
    parsed, _ = read_blocks(new_text)
    known = set(parsed.keys())
    missing = []
    for fid, (unity_type, body) in parsed.items():
        for ref in local_refs(body):
            if ref != 0 and ref not in known:
                missing.append((fid, ref))
    if missing:
        for fid, ref in missing[:20]:
            print(f"  DANGLING: block {fid} -> {ref}")
        raise SystemExit("Validation failed: dangling local references in output scene")

    open(SCENE_PATH, "w", encoding="utf-8").write(new_text)
    print(f"Agent2 = GO {agent2_go}, behavior '{AGENT2_BEHAVIOR}', position {AGENT2_POSITION}")
    print(f"Target2 = GO {target2_go}, position {TARGET2_POSITION}")
    print(f"New blocks added: {len(new_blocks)}; scene now has {len(parsed)} blocks")
    print(f"Scene rewritten: {SCENE_PATH}")


if __name__ == "__main__":
    main()
