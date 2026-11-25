"""Tiny SVG diagram kit: panels, nodes, routed edges, light and dark themes.

Used by make.py. Every diagram is laid out by hand with explicit coordinates;
this module only draws. Each diagram is written twice, NAME.svg (light) and
NAME-dark.svg, with fixed colours so GitHub's <picture> element can pick one.
"""
from html import escape

FONT = ('-apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", '
        'Helvetica, Arial, sans-serif')

THEMES = {
    "light": {
        "bg": "#ffffff", "ink": "#1f2328", "mute": "#59636e", "edge": "#818b98",
        "panel": {"pc": ("#f3f6fd", "#d5def3"), "hw": ("#f0f8f3", "#cde7d6"),
                  "grey": ("#f6f8fa", "#d8dee4"), "amber": ("#fdf8ef", "#efdcb8")},
        "node": {"build": ("#ffffff", "#4f6bd8"), "check": ("#fffaf0", "#d08a1e"),
                 "hw": ("#ffffff", "#2f9b62"), "store": ("#fffaf0", "#d08a1e"),
                 "neutral": ("#ffffff", "#8c959f"), "decide": ("#f7f3ff", "#8b5cf6"),
                 "future": ("#ffffff", "#8c959f"), "note": ("#f6f8fa", "#afb8c1")},
    },
    "dark": {
        "bg": "#0d1117", "ink": "#e6edf3", "mute": "#9198a1", "edge": "#7d8590",
        "panel": {"pc": ("#121a2c", "#2a3757"), "hw": ("#0f1f18", "#24483a"),
                  "grey": ("#151b23", "#30363d"), "amber": ("#1e1810", "#4a3a1e")},
        "node": {"build": ("#161f36", "#7d95f0"), "check": ("#2a2113", "#e9a94a"),
                 "hw": ("#122a1f", "#4cc488"), "store": ("#2a2113", "#e9a94a"),
                 "neutral": ("#161b22", "#8b949e"), "decide": ("#221a38", "#a78bfa"),
                 "future": ("#0d1117", "#8b949e"), "note": ("#151b23", "#3d444d")},
    },
}

T_LH, S_LH, PAD = 18, 16, 11   # title / subtitle line height, vertical padding


def _lines(s):
    if s is None:
        return []
    return s if isinstance(s, list) else s.split("\n")


def node_height(title, sub=None):
    nt, ns = len(_lines(title)), len(_lines(sub))
    return 2 * PAD + nt * T_LH + ns * S_LH + (3 if ns else 0)


def rounded(pts, r=8):
    """Polyline through pts with rounded corners, as an SVG path string."""
    if len(pts) < 3:
        return "M" + " L".join(f"{x:g} {y:g}" for x, y in pts)
    d = [f"M{pts[0][0]:g} {pts[0][1]:g}"]
    for (x0, y0), (x1, y1), (x2, y2) in zip(pts, pts[1:], pts[2:]):
        l1 = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** .5
        l2 = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** .5
        rr = min(r, l1 / 2, l2 / 2)
        ax, ay = x1 - (x1 - x0) / l1 * rr, y1 - (y1 - y0) / l1 * rr
        bx, by = x1 + (x2 - x1) / l2 * rr, y1 + (y2 - y1) / l2 * rr
        d.append(f"L{ax:g} {ay:g} Q{x1:g} {y1:g} {bx:g} {by:g}")
    d.append(f"L{pts[-1][0]:g} {pts[-1][1]:g}")
    return " ".join(d)


class Diagram:
    def __init__(self, name, w, h, alt):
        self.name, self.w, self.h, self.alt = name, w, h, alt
        self.items = []          # (layer, callable(theme) -> svg)
        self.nodes = {}

    # ---- geometry helpers -------------------------------------------------
    def a(self, nid, side, t=0.5):
        """Anchor point on a node side: l, r, t (top), b (bottom); t = 0..1."""
        x, y, w, h = self.nodes[nid]
        return {"l": (x, y + h * t), "r": (x + w, y + h * t),
                "t": (x + w * t, y), "b": (x + w * t, y + h)}[side]

    # ---- drawing ------------------------------------------------------------
    def panel(self, x, y, w, h, title, sub=None, kind="pc"):
        def draw(th):
            f, s = th["panel"][kind]
            out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{f}" stroke="{s}"/>',
                   f'<text x="{x + 20}" y="{y + 28}" class="pt">{escape(title)}</text>']
            if sub:
                out.append(f'<text x="{x + 20}" y="{y + 45}" class="ps">{escape(sub)}</text>')
            return "".join(out)
        self.items.append((0, draw))

    def node(self, nid, x, y, w, title, sub=None, kind="build", h=None, shape="rect"):
        tl, sl = _lines(title), _lines(sub)
        h = h or node_height(tl, sl)
        self.nodes[nid] = (x, y, w, h)

        def draw(th):
            f, s = th["node"][kind]
            rx = h / 2 if shape == "pill" else 8
            dash = ' stroke-dasharray="5 4"' if kind in ("future", "note") else ""
            out = [f'<g class="n" data-id="{nid}">'
                   f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{min(rx, 22):g}" '
                   f'fill="{f}" stroke="{s}" stroke-width="1.5"{dash}/>']
            block = len(tl) * T_LH + len(sl) * S_LH + (3 if sl else 0)
            cy = y + (h - block) / 2
            tcls = "ni" if kind == "note" else "nt"
            for i, line in enumerate(tl):
                out.append(f'<text x="{x + w / 2:g}" y="{cy + 13.5 + i * T_LH:g}" class="{tcls}">{escape(line)}</text>')
            cy += len(tl) * T_LH + 3
            for i, line in enumerate(sl):
                out.append(f'<text x="{x + w / 2:g}" y="{cy + 12 + i * S_LH:g}" class="ns">{escape(line)}</text>')
            out.append("</g>")
            return "".join(out)
        self.items.append((2, draw))
        return self

    def edge(self, pts, dashed=False, head=True, accent=None):
        def draw(th):
            col = th["node"][accent][1] if accent else th["edge"]
            sw = 2.2 if accent else 1.5
            mk = f' marker-end="url(#{self._mk(accent)})"' if head else ""
            dash = ' stroke-dasharray="5 4"' if dashed else ""
            return (f'<path d="{rounded(pts)}" fill="none" stroke="{col}" '
                    f'stroke-width="{sw}"{dash}{mk}/>')
        self.items.append((1, draw))

    def label(self, x, y, text, anchor="middle", bg="bg", accent=None, size=None):
        """Edge label; bg names the halo colour: 'bg' or a panel kind."""
        def draw(th):
            halo = th["bg"] if bg == "bg" else th["panel"][bg][0]
            col = th["node"][accent][1] if accent else th["mute"]
            fs = f' font-size="{size}"' if size else ""
            out = []
            for i, line in enumerate(_lines(text)):
                out.append(f'<text x="{x}" y="{y + i * 14.5}" text-anchor="{anchor}" class="el" '
                           f'fill="{col}" stroke="{halo}"{fs}>{escape(line)}</text>')
            return "".join(out)
        self.items.append((3, draw))

    # ---- output ---------------------------------------------------------------
    def _mk(self, accent):
        return "ah-" + (accent or "edge")

    def render(self, theme):
        th = THEMES[theme]
        markers = [("ah-edge", th["edge"])] + [
            (f"ah-{k}", v[1]) for k, v in th["node"].items()]
        defs = "".join(
            f'<marker id="{i}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" '
            f'markerHeight="9" markerUnits="userSpaceOnUse" orient="auto-start-reverse">'
            f'<path d="M0 0.5 L10 5 L0 9.5 z" fill="{c}"/></marker>' for i, c in markers)
        style = (f'text{{font-family:{FONT};fill:{th["ink"]}}}'
                 f'.pt{{font-size:14px;font-weight:700}}'
                 f'.ps{{font-size:12px;fill:{th["mute"]}}}'
                 f'.nt{{font-size:14px;font-weight:600;text-anchor:middle}}'
                 f'.ni{{font-size:13px;font-style:italic;text-anchor:middle;fill:{th["mute"]}}}'
                 f'.ns{{font-size:12px;text-anchor:middle;fill:{th["mute"]}}}'
                 f'.el{{font-size:11.5px;font-style:italic;stroke-width:4px;'
                 f'stroke-linejoin:round;paint-order:stroke}}')
        body = "".join(fn(th) for _, fn in sorted(self.items, key=lambda it: it[0]))
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
                f'width="{self.w}" height="{self.h}" role="img" aria-label="{escape(self.alt)}">'
                f'<title>{escape(self.alt)}</title><style>{style}</style><defs>{defs}</defs>'
                f'{body}</svg>\n')
