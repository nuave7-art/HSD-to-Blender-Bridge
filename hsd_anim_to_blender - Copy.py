"""
HSD animation -> Blender armature, independent of bone orientation.

Why this exists
---------------
The Maya .anim add-on assumes the file's local axes equal Blender's bone axes.
Blender's Collada importer rebuilds bones (Y along the bone), so HSD Euler values
land in the wrong axes. This script instead:

  1. reads each joint's ORIGINAL local bind matrix from the exported .dae,
  2. reads the animated local values from the exported .anim,
  3. computes each joint's world "delta" (pose world * bind world^-1),
  4. moves each Blender bone rigidly by that delta, using Blender's OWN rest matrix.

Usage (Blender, Scripting tab):
  - Select the armature (active object), set the three paths below, Run Script.
  - Keys are written every frame into a new action. Rotation modes are respected.

Assumptions you can check in the printed report:
  - HSD Euler order is Rz*Ry*Rx (same as Blender 'XYZ'). Verified visually on this model.
  - Bones are not "Connected" and inherit rotation/scale normally (Collada import default).
"""
import xml.etree.ElementTree as ET
import numpy as np

try:
    import bpy
    from mathutils import Matrix, Quaternion, Euler
except ImportError:  # lets the math below be unit-tested outside Blender
    bpy = None

# ----------------------------------------------------------------------------
# SETTINGS - edit these
# ----------------------------------------------------------------------------
DAE_PATH = r"C:\Users\[your-path]\[your-model].dae"
ANIM_PATH = r"C:\Users\[your-path]\[your-anim].anim"
ACTION_NAME = "[your action name]"
FRAME_OFFSET = 1          # Blender frame = anim frame + FRAME_OFFSET
FPS = 60                  # HSD runs at 60 (header says ntscf)
ROT_ORDER = "ZYX"         # R = Rz*Ry*Rx, i.e. Blender 'XYZ'. Use "XYZ" only if poses look wrong.

NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}


# ----------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------
def load_dae(path):
    """Returns (joints, order). joints[name] = {'local': 4x4, 'parent': name|None}."""
    root = ET.parse(path).getroot()
    scene = root.find(".//c:library_visual_scenes", NS)
    joints, order = {}, []

    def walk(node, parent):
        if node.get("type") == "JOINT":
            m = np.array(node.find("c:matrix", NS).text.split(), float).reshape(4, 4)
            name = node.get("id")
            joints[name] = {"local": m, "parent": parent}
            order.append(name)
            parent = name
        for child in node.findall("c:node", NS):
            walk(child, parent)

    for vs in scene.findall("c:visual_scene", NS):
        for n in vs.findall("c:node", NS):
            walk(n, None)
    return joints, order


def load_anim(path):
    """Returns (header, tracks). tracks[joint][channel][frame] = value."""
    lines = open(path).read().replace("\r", "").split("\n")
    header, tracks, cur, i = {}, {}, None, 0
    while i < len(lines):
        a = lines[i].strip().rstrip(";").split()
        if a and a[0] in ("timeUnit", "linearUnit", "angularUnit", "startTime", "endTime"):
            header[a[0]] = a[1]
        elif a and a[0] == "anim" and len(a) == 7:
            cur = (a[3], a[2])
            tracks.setdefault(a[3], {})[a[2]] = {}
        elif a and a[0] == "keys":
            i += 1
            while "}" not in lines[i]:
                k = lines[i].strip().rstrip(";").split()
                tracks[cur[0]][cur[1]][int(float(k[0]))] = float(k[1])
                i += 1
        i += 1
    return header, tracks


# ----------------------------------------------------------------------------
# Math core (numpy only, unit-tested outside Blender)
# ----------------------------------------------------------------------------
def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def euler_to_matrix(order, x, y, z):
    return _rz(z) @ _ry(y) @ _rx(x) if order == "ZYX" else _rx(x) @ _ry(y) @ _rz(z)


def matrix_to_euler(R, order):
    if order == "ZYX":  # R = Rz Ry Rx
        sy = -R[2, 0]
        y = np.arcsin(np.clip(sy, -1, 1))
        if abs(sy) < 0.99999:
            return np.arctan2(R[2, 1], R[2, 2]), y, np.arctan2(R[1, 0], R[0, 0])
        return np.arctan2(-R[1, 2], R[1, 1]), y, 0.0
    sy = R[0, 2]  # R = Rx Ry Rz
    y = np.arcsin(np.clip(sy, -1, 1))
    if abs(sy) < 0.99999:
        return np.arctan2(-R[1, 2], R[2, 2]), y, np.arctan2(-R[0, 1], R[0, 0])
    return np.arctan2(R[2, 1], R[1, 1]), y, 0.0


def world_matrices(joints, order, tracks, frame, rot_order=ROT_ORDER):
    """World matrix of every joint (DAE space). tracks=None gives the bind pose.
    Channels missing from the animation fall back to the DAE bind values."""
    world = {}
    for name in order:
        local = joints[name]["local"]
        pos = local[:3, 3].copy()
        rot = local[:3, :3]
        if tracks is not None and name in tracks:
            t = tracks[name]
            e = list(matrix_to_euler(rot, rot_order))
            for i, ch in enumerate(("rotateX", "rotateY", "rotateZ")):
                if ch in t:
                    e[i] = t[ch][frame]
            rot = euler_to_matrix(rot_order, *e)
            for i, ch in enumerate(("translateX", "translateY", "translateZ")):
                if ch in t:
                    pos[i] = t[ch][frame]
        m = np.eye(4)
        m[:3, :3] = rot
        m[:3, 3] = pos
        parent = joints[name]["parent"]
        world[name] = (world[parent] if parent else np.eye(4)) @ m
    return world


# Candidate "armature space <- DAE space" transforms (Blender usually keeps DAE space
# and rotates the object, but we check instead of assuming).
_YUP_TO_ZUP = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)
SPACE_CANDIDATES = {
    "identity": np.eye(4),
    "yup_to_zup": _YUP_TO_ZUP,
    "zup_to_yup": np.linalg.inv(_YUP_TO_ZUP),
}


def detect_space(bind_world, rest_heads):
    """rest_heads: {bone: xyz of bone head in armature space}. Picks the candidate with
    the smallest mean distance between bone heads and joint origins. Returns (name, G, err)."""
    best = None
    for cname, G in SPACE_CANDIDATES.items():
        errs = []
        for n, head in rest_heads.items():
            if n in bind_world:
                errs.append(np.linalg.norm((G @ bind_world[n])[:3, 3] - np.asarray(head)))
        if errs:
            err = float(np.mean(errs))
            if best is None or err < best[2]:
                best = (cname, G, err)
    return best


def pose_matrices(bind_world, world, rest, G):
    """Armature-space pose matrix per bone: rigid delta of the HSD joint applied to the
    bone's own rest matrix.   P = G * (W * W0^-1) * G^-1 * R"""
    Ginv = np.linalg.inv(G)
    out = {}
    for n, R in rest.items():
        if n not in world:
            continue
        D = world[n] @ np.linalg.inv(bind_world[n])
        out[n] = G @ D @ Ginv @ R
    return out


def basis_matrices(pose, rest, parents):
    """Blender pose-bone basis (loc/rot/scale) per bone so that
       P = P_parent * (R_parent^-1 * R) * Basis      (root: P = R * Basis)."""
    basis = {}
    for n, P in pose.items():
        par = parents.get(n)
        if par and par in pose:
            rel = np.linalg.inv(rest[par]) @ rest[n]
            basis[n] = np.linalg.inv(rel) @ np.linalg.inv(pose[par]) @ P
        else:
            basis[n] = np.linalg.inv(rest[n]) @ P
    return basis


# ----------------------------------------------------------------------------
# Blender part
# ----------------------------------------------------------------------------
REPORT = []


def log(msg):
    print(msg)
    REPORT.append(str(msg))


def run():
    arm = bpy.context.active_object
    if arm is None or arm.type != "ARMATURE":
        raise RuntimeError("Select the armature (make it the active object) first.")

    joints, order = load_dae(DAE_PATH)
    header, tracks = load_anim(ANIM_PATH)
    last = max(max(vals) for t in tracks.values() for vals in t.values() if vals)
    frames = range(0, last + 1)

    bones = arm.data.bones
    rest = {b.name: np.array(b.matrix_local) for b in bones if b.name in joints}
    parents = {b.name: (b.parent.name if b.parent else None) for b in bones}
    skipped = [b.name for b in bones if b.name not in joints]
    missing = [n for n in joints if n not in bones]
    unknown_tracks = [n for n in tracks if n not in bones]

    bind_world = world_matrices(joints, order, None, 0)
    heads = {b.name: np.array(b.head_local) for b in bones if b.name in joints}
    space_name, G, err = detect_space(bind_world, heads)

    log("---- HSD -> Blender report ----")
    log(f"bones in armature: {len(bones)}   joints in DAE: {len(joints)}")
    log(f"armature bones not in DAE (left alone): {skipped}")
    log(f"DAE joints not in armature (ignored):   {missing}")
    log(f"animated joints not in armature:        {unknown_tracks}")
    log(f"space guess: {space_name}, mean head error {err:.5f}")
    if err > 0.05:
        log("WARNING: bone heads do not line up with DAE joint origins; the pose will be wrong.")
        log("         Send this report back instead of trusting the result.")

    bpy.ops.object.mode_set(mode="POSE")
    arm.data.pose_position = "POSE"
    scene = bpy.context.scene
    scene.render.fps = FPS

    action = bpy.data.actions.new(ACTION_NAME)
    if arm.animation_data is None:
        arm.animation_data_create()
    arm.animation_data.action = action

    prev_rot = {}
    for f in frames:
        w = world_matrices(joints, order, tracks, f)
        pose = pose_matrices(bind_world, w, rest, G)
        basis = basis_matrices(pose, rest, parents)
        bf = f + FRAME_OFFSET
        for pb in arm.pose.bones:  # data order is parent-first; basis does not depend on it
            if pb.name not in basis:
                continue
            loc, quat, scl = Matrix(basis[pb.name].tolist()).decompose()
            pb.location = loc
            pb.scale = scl
            if pb.rotation_mode == "QUATERNION":
                if pb.name in prev_rot:
                    quat.make_compatible(prev_rot[pb.name])
                prev_rot[pb.name] = quat.copy()
                pb.rotation_quaternion = quat
                pb.keyframe_insert("rotation_quaternion", frame=bf)
            else:
                eul = quat.to_euler(pb.rotation_mode, prev_rot.get(pb.name, Euler((0, 0, 0), pb.rotation_mode)))
                prev_rot[pb.name] = eul.copy()
                pb.rotation_euler = eul
                pb.keyframe_insert("rotation_euler", frame=bf)
            pb.keyframe_insert("location", frame=bf)
            pb.keyframe_insert("scale", frame=bf)

    for fc in action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"

    scene.frame_start = FRAME_OFFSET
    scene.frame_end = last + FRAME_OFFSET
    scene.frame_set(FRAME_OFFSET)
    log(f"done: {len(frames)} frames keyed into action '{ACTION_NAME}'")
    txt = bpy.data.texts.get("hsd_report") or bpy.data.texts.new("hsd_report")
    txt.clear()
    txt.write("\n".join(REPORT))
    print("Report also saved as text block 'hsd_report' (Text Editor > dropdown).")


if bpy is not None:
    run()
