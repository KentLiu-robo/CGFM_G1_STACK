"""Display colours per scene-graph class (BGR), shared by the scene-graph
panel, its legend and the VLM panel of the composite video. No heavy imports,
so make_cgfm_composite_video.py can use it too."""


# Fixed display colour per class (BGR), shared by the scene-graph panel and its
# legend; chosen to stay distinct from the map's own colours (light-green explored,
# gray padding, black obstacles, blue frontier rings, cyan robot).
CLASS_COLORS_BGR = {
    "chair": (40, 40, 220),        # red
    "table": (0, 140, 255),        # orange
    "desk": (30, 80, 140),         # brown
    "monitor": (180, 105, 255),    # pink
    "cabinet": (140, 0, 140),      # purple
    "door": (0, 190, 190),         # olive
    "trash can": (255, 0, 255),    # magenta
    "couch": (0, 110, 0),          # dark green
    "whiteboard": (130, 130, 0),   # teal
    "plant": (0, 200, 120),        # yellow-green
}
_EXTRA_COLORS_BGR = [(90, 90, 90), (0, 0, 128), (128, 128, 255), (255, 128, 0), (0, 255, 255), (128, 0, 64)]


def class_color(label: str):
    """Display colour for a class; classes outside CLASS_COLORS_BGR (e.g. a
    target that is not in the vocabulary) get a deterministic extra colour."""
    if label in CLASS_COLORS_BGR:
        return CLASS_COLORS_BGR[label]
    return _EXTRA_COLORS_BGR[sum(map(ord, label)) % len(_EXTRA_COLORS_BGR)]
