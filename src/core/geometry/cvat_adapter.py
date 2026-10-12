"""Adapter for converting between internal 3D cuboids and CVAT 3D REST API schema."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from .cuboid import Cuboid3D


class CVAT3DAdapter:
    """Handles mapping between Cuboid3D list and CVAT 3D shapes/tracks schema."""

    def __init__(self, label_mapping: Optional[Dict[str, int]] = None):
        """Args:
            label_mapping: Maps class_name (e.g. 'vehicles') to CVAT label_id (e.g. 1).
        """
        self.label_mapping = label_mapping or {}

    def to_shape_dict(self, cuboid: Cuboid3D, frame: int = 0) -> Dict[str, Any]:
        """Creates a single CVAT 3D shape dictionary."""
        label_id = cuboid.label_id or self.label_mapping.get(cuboid.class_name, 1)
        return {
            "type": "cuboid",
            "frame": frame,
            "occluded": False,
            "outside": False,
            "keyframe": True,
            "z_order": 0,
            "rotation": 0.0,
            "points": cuboid.to_cvat_points(),
            "label_id": label_id,
            "attributes": [
                {"name": "confidence", "value": f"{cuboid.score:.3f}"},
                {"name": "class_name", "value": cuboid.class_name}
            ]
        }

    def to_shapes_payload(self, cuboids: List[Cuboid3D], frame: int = 0) -> Dict[str, Any]:
        """Wraps cuboid shapes into standard CVAT job patch payload."""
        shapes = [self.to_shape_dict(c, frame=frame) for c in cuboids]
        return {
            "version": 0,
            "tags": [],
            "shapes": shapes,
            "tracks": []
        }

    def to_tracks_payload(self, cuboids_by_frame: Dict[int, List[Cuboid3D]]) -> Dict[str, Any]:
        """Generates sequence tracks payload (compatible with CVAT-3D-Copilot extension)."""
        tracks = []
        track_id = 1
        for frame, cuboids in sorted(cuboids_by_frame.items()):
            for c in cuboids:
                label_id = c.label_id or self.label_mapping.get(c.class_name, 1)
                tracks.append({
                    "id": track_id,
                    "label_id": label_id,
                    "frame": frame,
                    "group": 0,
                    "source": "semi-auto",
                    "attributes": [
                        {"spec_id": 1, "value": c.class_name}
                    ],
                    "shapes": [
                        {
                            "type": "cuboid",
                            "frame": frame,
                            "occluded": False,
                            "outside": False,
                            "keyframe": True,
                            "z_order": 0,
                            "rotation": 0.0,
                            "points": c.to_cvat_points(),
                            "attributes": []
                        }
                    ]
                })
                track_id += 1

        return {
            "version": 0,
            "tags": [],
            "shapes": [],
            "tracks": tracks
        }

    @staticmethod
    def parse_cvat_annotations(data: Dict[str, Any], id_to_label: Optional[Dict[int, str]] = None) -> List[Cuboid3D]:
        """Extracts Cuboid3D instances from CVAT annotations payload."""
        id_to_label = id_to_label or {}
        cuboids: List[Cuboid3D] = []

        # From shapes
        for s in data.get("shapes", []):
            if s.get("type") == "cuboid" and len(s.get("points", [])) >= 9:
                label_id = s.get("label_id")
                class_name = id_to_label.get(label_id, "vehicles")
                cuboids.append(Cuboid3D.from_cvat_points(
                    points=s["points"],
                    class_name=class_name,
                    label_id=label_id
                ))

        # From tracks
        for t in data.get("tracks", []):
            label_id = t.get("label_id")
            class_name = id_to_label.get(label_id, "vehicles")
            for s in t.get("shapes", []):
                if s.get("type") == "cuboid" and len(s.get("points", [])) >= 9:
                    cuboids.append(Cuboid3D.from_cvat_points(
                        points=s["points"],
                        class_name=class_name,
                        label_id=label_id
                    ))

        return cuboids
