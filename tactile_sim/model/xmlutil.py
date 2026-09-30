"""Small helpers for building MJCF with xml.etree."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterable


def fmt(v) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, (int, float)):
        return f"{v:.9g}"
    return " ".join(fmt(x) for x in v)


def sub(parent: ET.Element, tag: str, **attrs) -> ET.Element:
    return ET.SubElement(parent, tag, {k: fmt(v) for k, v in attrs.items() if v is not None})


def el(tag: str, **attrs) -> ET.Element:
    return ET.Element(tag, {k: fmt(v) for k, v in attrs.items() if v is not None})


def find_body(root: ET.Element, name: str) -> ET.Element:
    for b in root.iter("body"):
        if b.get("name") == name:
            return b
    raise KeyError(f"body {name!r} not found")


def disable_collisions(elems: Iterable[ET.Element]) -> None:
    """All contacts in the scene come from explicit <pair>s; turn off automatic collision."""
    for e in elems:
        for g in e.iter("geom"):
            g.set("contype", "0")
            g.set("conaffinity", "0")
