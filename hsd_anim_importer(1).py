"""
HSD Animation Batch Importer (Blender add-on)

Imports one or many HSDRawViewer-exported .anim files onto the active armature,
one action per file, independent of how Blender oriented the bones.

How it works
------------
The Maya .anim add-on assumes the file's local axes equal Blender's bone axes.
Blender's Collada importer rebuilds bones with its own axes, so HSD Euler values land
in the wrong frames. This add-on instead:
  1. reads each joint's ORIGINAL local bind matrix from the exported .dae,
  2. reads the animated local values from each .anim,
  3. computes each joint's world delta (pose world * bind world^-1),
  4. moves each Blender bone rigidly by that delta, using Blender's OWN rest matrix.

Install:  Edit > Preferences > Add-ons > Install... > pick this file > tick it.
Use:      select the armature, then EITHER
            a) File > Import > HSD Animations (.anim) and multi-select the .anim
               files TOGETHER WITH the model's .dae (Ctrl+click) in the same
               file window, or
            b) first set the model once: 3D Viewport > N panel > HSD tab >
               "Set Model (.dae)", then import the .anim files as usual.
          The .dae is remembered on the armature afterwards. If neither is done,
          a single .dae sitting in the same folder as the .anim files is used.
          (Blender cannot open a second file window from inside the import window,
          which is why the .dae is not picked with its own button there.)
Results:  one action per file (fake user on, so none are lost), report in the
          Text Editor block "hsd_report".
"""
bl_info = {
    "name": "HSD Animation Batch Importer",
    "author": "community",
    "version": (1, 1, 0),
    "blender": (3, 6, 0),
    "location": "File > Import > HSD Animations (.anim)",
    "description": "Import HSDRawViewer .anim files as actions, independent of bone orientation",
    "category": "Import-Export",
}

import os
import xml.etree.ElementTree as ET
import numpy as np

try:
    import bpy
    from bpy.props import StringProperty, BoolProperty, IntProperty, CollectionProperty
    from bpy_extras.io_utils import ImportHelper
    from mathutils import Matrix, Euler
except ImportError:  # lets the math below be unit-tested outside Blender
    bpy = None

ROT_ORDER = "ZYX"  # R = Rz*Ry*Rx, i.e. Blender 'XYZ'
NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
HEAD_ERROR_LIMIT = 0.05


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
    """Returns (header, tracks, last_frame, sparse_channels).
    tracks[joint][channel] = (frames ndarray, values ndarray)."""
    lines = open(path).read().replace("\r", "").split("\n")
    header, raw, cur, i = {}, {}, None, 0
    while i < len(lines):
        a = lines[i].strip().rstrip(";").split()
        if a and a[0] in ("timeUnit", "linearUnit", "angularUnit", "startTime", "endTime"):
            header[a[0]] = a[1]
        elif a and a[0] == "anim" and len(a) == 7:
            cur = (a[3], a[2])
            raw.setdefault(a[3], {})[a[2]] = ([], [])
        elif a and a[0] == "keys":
            i += 1
            while "}" not in lines[i]:
                k = lines[i].strip().rstrip(";").split()
                raw[cur[0]][cur[1]][0].append(float(k[0]))
                raw[cur[0]][cur[1]][1].append(float(k[1]))
                i += 1
        i += 1
    tracks, last, sparse = {}, 0, 0
    for jn, chans in raw.items():
        for ch, (fr, val) in chans.items():
            if not fr:
                continue
            order = np.argsort(fr)
            tracks.setdefault(jn, {})[ch] = (np.array(fr)[order], np.array(val)[order])
            last = max(last, int(max(fr)))
    for chans in tracks.values():
        for fr, _ in chans.values():
            if len(fr) != last + 1:
                sparse += 1
    return header, tracks, last, sparse


# ----------------------------------------------------------------------------
# Math core (numpy only; unit-tested outside Blender)
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
    Channels missing from the animation fall back to the DAE bind values. Frames
    without a key are linearly interpolated (np.interp also clamps at the ends)."""
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
                    e[i] = float(np.interp(frame, *t[ch]))
            rot = euler_to_matrix(rot_order, *e)
            for i, ch in enumerate(("translateX", "translateY", "translateZ")):
                if ch in t:
                    pos[i] = float(np.interp(frame, *t[ch]))
        m = np.eye(4)
        m[:3, :3] = rot
        m[:3, 3] = pos
        parent = joints[name]["parent"]
        world[name] = (world[parent] if parent else np.eye(4)) @ m
    return world


_YUP_TO_ZUP = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)
SPACE_CANDIDATES = {
    "identity": np.eye(4),
    "yup_to_zup": _YUP_TO_ZUP,
    "zup_to_yup": np.linalg.inv(_YUP_TO_ZUP),
}


def detect_space(bind_world, rest_heads):
    """Picks the 'armature space <- DAE space' transform whose joint origins best match
    the Blender bone heads. Returns (name, G, mean_error)."""
    best = None
    for cname, G in SPACE_CANDIDATES.items():
        errs = [
            np.linalg.norm((G @ bind_world[n])[:3, 3] - np.asarray(h))
            for n, h in rest_heads.items()
            if n in bind_world
        ]
        if errs:
            err = float(np.mean(errs))
            if best is None or err < best[2]:
                best = (cname, G, err)
    return best


def pose_matrices(bind_inv, world, rest, G, Ginv):
    """Armature-space pose per bone: P = G * (W * W0^-1) * G^-1 * R"""
    out = {}
    for n, R in rest.items():
        if n in world:
            out[n] = G @ (world[n] @ bind_inv[n]) @ Ginv @ R
    return out


def basis_matrices(pose, rest, parents):
    """Blender pose-bone basis per bone so that
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
# Blender glue
# ----------------------------------------------------------------------------
class ModelContext:
    """Everything derived from the .dae + armature, computed once per batch."""

    def __init__(self, arm, dae_path):
        self.joints, self.order = load_dae(dae_path)
        bones = arm.data.bones
        self.rest = {b.name: np.array(b.matrix_local) for b in bones if b.name in self.joints}
        self.parents = {b.name: (b.parent.name if b.parent else None) for b in bones}
        self.bone_names = {b.name for b in bones}
        self.skipped = [b.name for b in bones if b.name not in self.joints]
        self.missing = [n for n in self.joints if n not in self.bone_names]
        self.bind_world = world_matrices(self.joints, self.order, None, 0)
        self.bind_inv = {n: np.linalg.inv(m) for n, m in self.bind_world.items()}
        heads = {b.name: np.array(b.head_local) for b in bones if b.name in self.joints}
        self.space_name, self.G, self.head_error = detect_space(self.bind_world, heads)
        self.Ginv = np.linalg.inv(self.G)


def import_one(arm, scene, mctx, anim_path, frame_offset, fake_user, log):
    """Creates one action from one .anim. Returns (action, last_blender_frame)."""
    header, tracks, last, sparse = load_anim(anim_path)
    name = os.path.splitext(os.path.basename(anim_path))[0]
    unknown = [n for n in tracks if n not in mctx.bone_names]
    log(f"[{name}] frames 0..{last}, animated joints {len(tracks)}"
        + (f", NOT in armature: {unknown}" if unknown else ""))
    if sparse:
        log(f"[{name}] WARNING: {sparse} channels have keys that are not on every frame. "
            "Linear interpolation was used; spline tangents are ignored. "
            "Export with Bake Animation on for exact results.")

    action = bpy.data.actions.new(name)
    action.use_fake_user = fake_user
    if arm.animation_data is None:
        arm.animation_data_create()
    arm.animation_data.action = action

    prev_rot = {}
    for f in range(0, last + 1):
        w = world_matrices(mctx.joints, mctx.order, tracks, f)
        pose = pose_matrices(mctx.bind_inv, w, mctx.rest, mctx.G, mctx.Ginv)
        basis = basis_matrices(pose, mctx.rest, mctx.parents)
        bf = f + frame_offset
        for pb in arm.pose.bones:
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
                eul = quat.to_euler(pb.rotation_mode,
                                    prev_rot.get(pb.name, Euler((0, 0, 0), pb.rotation_mode)))
                prev_rot[pb.name] = eul.copy()
                pb.rotation_euler = eul
                pb.keyframe_insert("rotation_euler", frame=bf)
            pb.keyframe_insert("location", frame=bf)
            pb.keyframe_insert("scale", frame=bf)

    for fc in getattr(action, "fcurves", ()):  # absent on Blender 5.0+, where keys stay as inserted
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
    return action, last + frame_offset


def find_dae(selected, typed, arm, directory):
    """Resolve the .dae: selected in the file window > typed path > stored on the
    armature > the only .dae next to the .anim files (or one folder up)."""
    for p in selected:
        if p.lower().endswith(".dae") and os.path.isfile(p):
            return p
    for cand in (typed, arm.get("hsd_dae_path", "") if arm else ""):
        if cand:
            c = bpy.path.abspath(cand)
            if os.path.isfile(c):
                return c
    for d in (directory, os.path.dirname(os.path.normpath(directory)) if directory else ""):
        if d and os.path.isdir(d):
            daes = [f for f in os.listdir(d) if f.lower().endswith(".dae")]
            if len(daes) == 1:
                return os.path.join(d, daes[0])
    return None


def write_report(lines):
    txt = bpy.data.texts.get("hsd_report") or bpy.data.texts.new("hsd_report")
    txt.clear()
    txt.write("\n".join(lines))


if bpy is not None:

    class IMPORT_OT_hsd_anim_batch(bpy.types.Operator, ImportHelper):
        """Import HSD .anim files as actions (one per file)"""
        bl_idname = "import_anim.hsd_batch"
        bl_label = "Import HSD Animations"
        bl_options = {"UNDO", "PRESET"}

        filename_ext = ".anim"
        filter_glob: StringProperty(default="*.anim;*.dae", options={"HIDDEN"})
        files: CollectionProperty(type=bpy.types.OperatorFileListElement,
                                  options={"HIDDEN", "SKIP_SAVE"})
        directory: StringProperty(subtype="DIR_PATH")

        # Plain string on purpose: a FILE_PATH field would try to open a second file
        # browser, which Blender refuses while this one is open. Paste a path here,
        # or just select the .dae together with the .anim files.
        dae_path: StringProperty(
            name="Model (.dae)",
            description="Optional: paste the full path of the model's .dae. Leave empty to use "
                        "a .dae selected alongside the .anim files, the one remembered on the "
                        "armature, or the one in the same folder")
        frame_offset: IntProperty(
            name="Start Frame", default=1,
            description="Blender frame that animation frame 0 is placed on")
        fps: IntProperty(name="Frame Rate", default=60, min=1,
                         description="Scene frame rate (HSD runs at 60)")
        fake_user: BoolProperty(
            name="Fake User", default=True,
            description="Keep every action even when it is not assigned")
        ignore_check: BoolProperty(
            name="Import Despite Mismatch", default=False,
            description="Import even when bone positions do not match the .dae")

        def invoke(self, context, event):
            arm = context.active_object
            if arm is not None and not self.dae_path:
                self.dae_path = arm.get("hsd_dae_path", "")
            return super().invoke(context, event)

        def execute(self, context):
            arm = context.active_object
            if arm is None or arm.type != "ARMATURE":
                self.report({"ERROR"}, "Select the armature first (it must be the active object)")
                return {"CANCELLED"}

            selected = [os.path.join(self.directory, f.name) for f in self.files if f.name]
            if not selected and self.filepath:
                selected = [self.filepath]
            paths = [p for p in selected if p.lower().endswith(".anim")]

            dae = find_dae(selected, self.dae_path, arm, self.directory)
            if dae is None:
                self.report({"ERROR"}, "No model .dae found. Ctrl+click the .dae together with the "
                                       ".anim files, or use N panel > HSD > Set Model (.dae) first")
                return {"CANCELLED"}
            arm["hsd_dae_path"] = dae
            if not paths:
                self.report({"INFO" if os.path.isfile(dae) else "ERROR"},
                            "Model .dae saved on the armature; now select the .anim files")
                return {"CANCELLED"}

            lines = []

            def log(msg):
                print(msg)
                lines.append(str(msg))

            try:
                mctx = ModelContext(arm, dae)
            except Exception as e:  # unreadable or unexpected .dae
                self.report({"ERROR"}, f"Could not read the .dae: {e}")
                return {"CANCELLED"}

            log("---- HSD batch import ----")
            log(f"model: {dae}")
            log(f"bones in armature: {len(arm.data.bones)}   joints in DAE: {len(mctx.joints)}")
            log(f"armature bones not in DAE (left alone): {mctx.skipped}")
            log(f"DAE joints not in armature (ignored):   {mctx.missing}")
            log(f"space guess: {mctx.space_name}, mean head error {mctx.head_error:.5f}")
            if mctx.head_error > HEAD_ERROR_LIMIT and not self.ignore_check:
                log("STOPPED: bone heads do not line up with the .dae joint origins. "
                    "Wrong model, or a scaled/moved armature. Nothing was imported.")
                write_report(lines)
                self.report({"ERROR"}, "Armature does not match the .dae (see hsd_report)")
                return {"CANCELLED"}

            if context.object and context.object.mode != "POSE":
                bpy.ops.object.mode_set(mode="POSE")
            arm.data.pose_position = "POSE"
            context.scene.render.fps = self.fps

            made, longest, failed = [], 0, []
            for p in sorted(paths):
                try:
                    action, end = import_one(arm, context.scene, mctx, p,
                                             self.frame_offset, self.fake_user, log)
                    made.append(action)
                    longest = max(longest, end)
                except Exception as e:
                    failed.append(os.path.basename(p))
                    log(f"[{os.path.basename(p)}] FAILED: {e}")

            if made:
                arm.animation_data.action = made[0]
                context.scene.frame_start = self.frame_offset
                context.scene.frame_end = longest
                context.scene.frame_set(self.frame_offset)
            log(f"done: {len(made)} action(s) created, {len(failed)} failed")
            write_report(lines)
            self.report({"WARNING" if failed else "INFO"},
                        f"{len(made)} action(s) imported" + (f", {len(failed)} failed" if failed else ""))
            return {"FINISHED"} if made else {"CANCELLED"}

        def draw(self, context):
            layout = self.layout
            layout.prop(self, "dae_path")
            layout.prop(self, "frame_offset")
            layout.prop(self, "fps")
            layout.prop(self, "fake_user")
            layout.prop(self, "ignore_check")

    class HSD_OT_set_dae(bpy.types.Operator, ImportHelper):
        """Choose the model's .dae and remember it on the active armature"""
        bl_idname = "import_anim.hsd_set_dae"
        bl_label = "Set Model (.dae)"

        filename_ext = ".dae"
        filter_glob: StringProperty(default="*.dae", options={"HIDDEN"})

        @classmethod
        def poll(cls, context):
            return context.active_object is not None and context.active_object.type == "ARMATURE"

        def execute(self, context):
            if not os.path.isfile(self.filepath):
                self.report({"ERROR"}, "File not found")
                return {"CANCELLED"}
            context.active_object["hsd_dae_path"] = self.filepath
            self.report({"INFO"}, f"Model set: {os.path.basename(self.filepath)}")
            return {"FINISHED"}

    class HSD_PT_panel(bpy.types.Panel):
        bl_label = "HSD Animations"
        bl_idname = "HSD_PT_anim_importer"
        bl_space_type = "VIEW_3D"
        bl_region_type = "UI"
        bl_category = "HSD"

        def draw(self, context):
            layout = self.layout
            arm = context.active_object
            if arm is None or arm.type != "ARMATURE":
                layout.label(text="Select the armature", icon="INFO")
                return
            current = arm.get("hsd_dae_path", "")
            layout.label(text=os.path.basename(current) if current else "No model set",
                         icon="CHECKMARK" if current else "ERROR")
            layout.operator(HSD_OT_set_dae.bl_idname, icon="FILE_FOLDER")
            layout.operator(IMPORT_OT_hsd_anim_batch.bl_idname, icon="IMPORT")

    CLASSES = (IMPORT_OT_hsd_anim_batch, HSD_OT_set_dae, HSD_PT_panel)

    def menu_func_import(self, context):
        self.layout.operator(IMPORT_OT_hsd_anim_batch.bl_idname, text="HSD Animations (.anim)")

    def register():
        for c in CLASSES:
            bpy.utils.register_class(c)
        bpy.types.TOPBAR_MT_file_import.append(menu_func_import)

    def unregister():
        bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
        for c in reversed(CLASSES):
            bpy.utils.unregister_class(c)

    if __name__ == "__main__":
        register()
