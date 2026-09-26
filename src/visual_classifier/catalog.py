"""Default scene questions and configurable object vocabulary."""

OBJECTS = (
    "person", "dog", "cat", "bird", "plant", "tree", "flower", "car",
    "bicycle", "motorcycle", "bus", "truck", "traffic light", "road sign",
    "building", "door", "window", "chair", "table", "desk", "sofa",
    "bed", "lamp", "television", "computer", "laptop", "phone", "book",
    "bottle", "cup", "plate", "food", "bag", "backpack", "clock", "toy",
)


def object_questions(objects):
    """Build independent presence questions; several objects can be present."""
    result = {}
    for name in objects:
        name = name.strip().lower()
        if not name:
            continue
        if name in result:
            raise ValueError(f"Duplicate object: {name}")
        result[name] = {
            "type": "binary",
            "instructions": f"Is a {name} visibly present in the image? Answer yes only if visibly supported.",
        }
    if not result:
        raise ValueError("At least one object is required")
    return result


SCENE_QUESTIONS = {
    "setting": {
        "type": "choice", "instructions": "Where is the camera scene?",
        "criteria": {"indoors": None, "outdoors": None, "unclear": None},
    },
    "light": {
        "type": "score", "instructions": "How bright is the visible scene?",
        "criteria": ["dark", "dim", "bright"],
    },
}
