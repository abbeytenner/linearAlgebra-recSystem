"""
VECTOR LOVE: a matchmaking recommender in one file
Linear Algebra, Sections 5.1 - 5.3 (Inner Product Spaces)

Every person is a vector of attribute scores on an ordered basis. Matches come
from the inner product, norm, distance, and direction cosine between vectors.

Run:   python3 vector_love.py
Needs: numpy (pip install numpy). tkinter ships with Python; on some Linux
       installs it is a separate package: sudo apt install python3-tk
"""

import math
import random
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np


# ----------------------------------------------------------------------------
# BLOCK 1: INNER PRODUCT (Section 5.1 Theorem 5.1, Section 5.3 Definition 5.1)
# Concept: the standard inner product on R^n is the dot product. Section 5.3
# generalizes it: any function (u, v) -> real number that satisfies the four
# axioms is an inner product. A weighted dot product, sum(w_i * u_i * v_i),
# satisfies all four axioms as long as every weight is strictly positive, so
# it lets us say "attribute 2 matters twice as much" and still stay inside an
# inner product space.
# ----------------------------------------------------------------------------
def make_inner_product(weights=None):
    """Return a function (u, v) -> real number. weights=None gives the dot product."""
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if np.any(weights <= 0):
            # Axiom 1 ((u,u) = 0 only when u = 0) fails if any weight is <= 0.
            raise ValueError("All weights must be strictly positive.")

    def inner(u, v):
        u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
        w = 1.0 if weights is None else weights
        return float(np.sum(w * u * v))

    return inner


# ----------------------------------------------------------------------------
# BLOCK 2: VERIFYING THE FOUR AXIOMS (Section 5.3 Definition 5.1)
# Concept: an inner product must satisfy
#   1. (u, u) >= 0, and (u, u) = 0 if and only if u = 0   (positive definite)
#   2. (v, u) = (u, v)                                      (symmetry)
#   3. (u + v, w) = (u, w) + (v, w)                         (additivity)
#   4. (c u, v) = c (u, v)                                  (homogeneity)
# This block tests them numerically on random vectors before the system uses
# the inner product, so a bad weight choice is caught early.
# ----------------------------------------------------------------------------
def verify_inner_product_axioms(inner, dim, trials=200, tol=1e-9, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(trials):
        u, v, w = rng.integers(-5, 6, size=(3, dim)).astype(float)
        c = float(rng.integers(-5, 6))
        assert inner(u, u) >= -tol, "Axiom 1 failed: (u,u) < 0"
        assert abs(inner(v, u) - inner(u, v)) < tol, "Axiom 2 failed"
        assert abs(inner(u + v, w) - (inner(u, w) + inner(v, w))) < tol, "Axiom 3 failed"
        assert abs(inner(c * u, v) - c * inner(u, v)) < tol, "Axiom 4 failed"
    assert inner(np.zeros(dim), np.zeros(dim)) == 0, "Axiom 1 failed: (0,0) != 0"
    return True


# ----------------------------------------------------------------------------
# BLOCK 3: LENGTH / NORM (Section 5.1 "The Concept of Length")
# Concept: ||u|| = sqrt((u, u)). In the deck, Descartes has ||u|| = sqrt(8)
# and Nikolai has sqrt(2), so Descartes expresses his traits more strongly.
# Length measures how prominent the traits are, not whether two people match.
# ----------------------------------------------------------------------------
def norm(u, inner):
    return float(np.sqrt(inner(u, u)))


# ----------------------------------------------------------------------------
# BLOCK 4: DISTANCE (Section 5.1 Distance, Section 5.3 Definition 5.3)
# Concept: d(u, v) = ||u - v||. Two people with a small distance have similar
# scores on every attribute, which is a stricter kind of closeness than
# pointing the same way (Kyle and David in the deck).
# ----------------------------------------------------------------------------
def distance(u, v, inner):
    return norm(np.asarray(u, dtype=float) - np.asarray(v, dtype=float), inner)


# ----------------------------------------------------------------------------
# BLOCK 5: UNIT VECTORS (Section 5.1 Unit Vectors)
# Concept: for a nonzero x, u = x / ||x|| has length 1 and the same direction
# as x. Normalizing strips away length so that only direction remains.
# ----------------------------------------------------------------------------
def unit_vector(x, inner):
    x = np.asarray(x, dtype=float)
    n = norm(x, inner)
    if n == 0:
        raise ValueError("The zero vector has no direction.")
    return x / n


# ----------------------------------------------------------------------------
# BLOCK 6: DIRECTION AND ANGLE (Section 5.1 Direction)
# Concept: cos(theta) = (u, v) / (||u|| ||v||), with 0 <= theta <= pi.
# cos = 1 means identical direction (Descartes and Nikolai, theta = 0),
# cos = -1 means opposite directions, cos = 0 means orthogonal.
# Cauchy-Schwarz (Block 7) guarantees the ratio sits in [-1, 1], so the
# arccos call below is always defined. The clip only absorbs rounding error.
# ----------------------------------------------------------------------------
def cosine(u, v, inner):
    nu, nv = norm(u, inner), norm(v, inner)
    if nu == 0 or nv == 0:
        return 0.0  # a person with no recorded traits has no direction
    return float(np.clip(inner(u, v) / (nu * nv), -1.0, 1.0))


def angle_degrees(u, v, inner):
    return float(np.degrees(np.arccos(cosine(u, v, inner))))


# ----------------------------------------------------------------------------
# BLOCK 7: CAUCHY-SCHWARZ AND TRIANGLE INEQUALITIES (Section 5.3 Theorem 5.3)
# Concept: |(u, v)| <= ||u|| ||v||  and  ||u + v|| <= ||u|| + ||v||.
# These hold in every inner product space. The recommender checks them on the
# real profile data as a sanity test on the inner product it was handed.
# ----------------------------------------------------------------------------
def check_inequalities(profiles, inner, tol=1e-9):
    names = list(profiles)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            u, v = np.asarray(profiles[a], float), np.asarray(profiles[b], float)
            assert abs(inner(u, v)) <= norm(u, inner) * norm(v, inner) + tol, \
                f"Cauchy-Schwarz failed for {a}, {b}"
            assert norm(u + v, inner) <= norm(u, inner) + norm(v, inner) + tol, \
                f"Triangle inequality failed for {a}, {b}"
    return True


# ----------------------------------------------------------------------------
# BLOCK 8: ORTHOGONALITY (Section 5.3 Definitions 5.4 and 5.5, Theorem 5.4)
# Concept: u and v are orthogonal when (u, v) = 0. A set is orthogonal when
# every distinct pair is orthogonal. Theorem 5.4: a finite orthogonal set of
# nonzero vectors is linearly independent. In the matchmaking language of the
# deck, orthogonal pairs are on "different wavelengths": neither pulls the
# other toward or away from its own traits.
# ----------------------------------------------------------------------------
def are_orthogonal(u, v, inner, tol=1e-9):
    return abs(inner(u, v)) < tol


def is_orthogonal_set(vectors, inner, tol=1e-9):
    return all(are_orthogonal(vectors[i], vectors[j], inner, tol)
               for i in range(len(vectors)) for j in range(i + 1, len(vectors)))


def orthogonal_pairs(profiles, inner):
    names = list(profiles)
    return [(a, b) for i, a in enumerate(names) for b in names[i + 1:]
            if are_orthogonal(profiles[a], profiles[b], inner)]


def demonstrate_theorem_5_4(vectors, inner):
    """If the vectors are orthogonal and nonzero, the rank must equal their count."""
    vectors = [np.asarray(v, float) for v in vectors]
    if not (is_orthogonal_set(vectors, inner) and all(norm(v, inner) > 0 for v in vectors)):
        return None
    return np.linalg.matrix_rank(np.vstack(vectors)) == len(vectors)


# ----------------------------------------------------------------------------
# BLOCK 9: COMPATIBILITY LABELS (Section 5.1 and the Love Stereotypes activity)
# Concept: the sign of the dot product gives the verdict from the deck:
#   (u, v) > 0 -> "they vibe"
#   (u, v) < 0 -> "emotionally opposites"
#   (u, v) = 0 -> "different wavelengths"
# The cosine then says how strong the verdict is, independent of length.
# ----------------------------------------------------------------------------
def compatibility_label(dot, cos, tol=1e-9):
    if abs(dot) < tol:
        return "different wavelengths (orthogonal)"
    strength = ("perfectly" if abs(cos) > 1 - 1e-9 else
                "strongly" if abs(cos) >= 0.8 else
                "moderately" if abs(cos) >= 0.4 else "slightly")
    return f"{strength} {'they vibe' if dot > 0 else 'emotionally opposite'}"


# ----------------------------------------------------------------------------
# BLOCK 10: THE RECOMMENDER
# Concept: every ranking mode is one of the quantities built above.
#   "cosine"   : direction only. Ignores how intense the traits are.
#   "dot"      : the raw inner product. Rewards long vectors, so a person with
#                strong traits can outrank a better directional match
#                (the "Descartes is more dominant" question in the deck).
#   "distance" : closeness in space, via 1 / (1 + d(u, v)).
#   "blend"    : alpha * direction + (1 - alpha) * closeness. Direction is
#                rescaled from [-1, 1] to [0, 1] so both terms share a scale.
# ----------------------------------------------------------------------------
class MatchRecommender:
    MODES = ("cosine", "dot", "distance", "blend")

    def __init__(self, profiles, weights=None):
        self.profiles = {k: np.asarray(v, dtype=float) for k, v in profiles.items()}
        dims = {v.shape[0] for v in self.profiles.values()}
        if len(dims) != 1:
            raise ValueError("Every profile must live in the same R^n.")
        self.dim = dims.pop()
        self.inner = make_inner_product(weights)
        verify_inner_product_axioms(self.inner, self.dim)   # Block 2
        check_inequalities(self.profiles, self.inner)       # Block 7

    def score(self, a, b, mode="cosine", alpha=0.5):
        u, v = self.profiles[a], self.profiles[b]
        if mode == "cosine":
            return cosine(u, v, self.inner)
        if mode == "dot":
            return self.inner(u, v)
        if mode == "distance":
            return 1.0 / (1.0 + distance(u, v, self.inner))
        if mode == "blend":
            direction = (cosine(u, v, self.inner) + 1.0) / 2.0
            closeness = 1.0 / (1.0 + distance(u, v, self.inner))
            return alpha * direction + (1.0 - alpha) * closeness
        raise ValueError(f"mode must be one of {self.MODES}")

    def recommend(self, name, k=3, mode="cosine", alpha=0.5):
        scored = [(other, self.score(name, other, mode, alpha))
                  for other in self.profiles if other != name]
        return sorted(scored, key=lambda t: t[1], reverse=True)[:k]

    def explain(self, a, b):
        u, v = self.profiles[a], self.profiles[b]
        dot, cos = self.inner(u, v), cosine(u, v, self.inner)
        return {
            "dot": dot,
            "length_a": norm(u, self.inner),
            "length_b": norm(v, self.inner),
            "distance": distance(u, v, self.inner),
            "cosine": cos,
            "angle_deg": angle_degrees(u, v, self.inner),
            "verdict": compatibility_label(dot, cos),
        }


# ----------------------------------------------------------------------------
# BLOCK 11: DATA (Section 5.1 "Universe of CS Students" and the Love Stereotypes)
# Concept: a person is a coordinate vector relative to an ordered basis. Scores
# run from -2 (strongly disagree) to 2 (strongly agree).
#   CS basis  : (DP, O, MC) = (Debugging Philosopher, Overengineer, Midnight Compiler)
#   Love basis: (HR, CP, SS, HE, CE) = (Hopeless Romantic, Commitment Phobic,
#               Situationship Specialist, Healing Era, Clown Energy)
# ----------------------------------------------------------------------------
CS_STUDENTS = {
    "Einstein": [0, 2, 2],   "Descartes": [-2, 0, 2],
    "Turing":   [1, 1, 1],   "Nikolai":   [-1, 0, 1],
    "David":    [1, -2, 1],  "Augustin":  [-2, 1, 0],
    "Samantha": [1, 0, -2],  "Kyle":      [0, -2, 1],
    "Miguel":   [2, 0, -1],  "Dominic":   [0, 0, 2],
}

LOVE_STEREOTYPES = {
    "Hopeless Romantic":        [2, -2, -2, 0, 1],
    "Commitment Phobic":        [-1, 2, 1, 1, 0],
    "Situationship Specialist": [1, 1, 2, -1, 1],
    "Healing Era":              [0, -1, 1, 2, 0],
    "Clown Energy":             [2, -2, 1, 0, 2],
}


# ----------------------------------------------------------------------------
# BLOCK 12: UNIVERSES (Section 5.1 "Universe of CS Students", Love Stereotypes)
# Concept: a person is a coordinate vector relative to an ordered basis. Each
# basis vector gets a quiz statement; the slider value on the -2..2 scale
# becomes that coordinate. Joining the universe adds a new vector to the same
# inner product space as everyone else.
# ----------------------------------------------------------------------------
SCALE = {-2: "Strongly disagree", -1: "Disagree", 0: "Neutral", 1: "Agree", 2: "Strongly agree"}


class Universe:
    def __init__(self, title, basis, statements, profiles):
        self.title = title
        self.basis = basis              # ordered basis labels
        self.statements = statements    # one quiz statement per basis vector
        self.profiles = {k: list(v) for k, v in profiles.items()}
        self.weights = None             # None means the standard dot product
        self.rec = None
        self.rebuild()

    @property
    def dim(self):
        return len(self.basis)

    def rebuild(self):
        """Re-verifies the inner product axioms every time the space changes."""
        self.rec = MatchRecommender(self.profiles, self.weights)


UNIVERSES = {
    "1": Universe(
        "CS Students (R^3)",
        ["Debugging Philosopher", "Overengineer", "Midnight Compiler"],
        ["I stare at one bug for hours, five StackOverflow tabs deep, asking why.",
         "Given a Hello World, I would add a database, auth, and a cloud deployment.",
         "My best code appears around 2 AM with dark mode on and brightness at 5 percent."],
        CS_STUDENTS),
    "2": Universe(
        "Love Stereotypes (R^5)",
        ["Hopeless Romantic", "Commitment Phobic", "Situationship Specialist",
         "Healing Era", "Clown Energy"],
        ["I believe in the one and I make playlists for people who do not text back.",
         "I fall fast and disappear faster.",
         "'IDK what we are' is basically my love language.",
         "I am focusing on myself, with a dating app still installed.",
         "I reply in 0.2 seconds and call one compliment per month fate."],
        LOVE_STEREOTYPES),
}


# ----------------------------------------------------------------------------
# BLOCK 13: VERDICT TEXT (Section 5.1 and the Love Stereotypes activity)
# Concept: the sign of the inner product gives the verdict and the cosine
# gives its strength. (u,v) > 0 they vibe, (u,v) < 0 opposites, (u,v) = 0
# orthogonal. The one-liners below attach a quip to each range.
# ----------------------------------------------------------------------------
MODE_HELP = {
    "cosine": "direction only (angle between vectors)",
    "dot": "raw inner product (loud profiles get a boost)",
    "distance": "closeness in space, 1 / (1 + d(u,v))",
    "blend": "mix of direction and closeness",
}

LESSONS = [
    "We are all vectors in the same space. What makes us compatible is the angle between us.",
    "To understand someone is to find a basis that spans their space.",
    "The best relationships are orthogonal: independent, yet part of the same space.",
    "Love is linear. It thrives when our transformations preserve each other's space.",
]


def quip(cos, dot):
    if abs(dot) < 1e-9:
        return "Zero overlap. Two lines that never argue and never agree."
    if cos >= 0.999:
        return "Same direction, angle zero. The space matched you for free."
    if cos >= 0.8:
        return "Pointing almost the same way. Group chat energy."
    if cos >= 0.4:
        return "Real overlap with room to surprise each other."
    if cos > 0:
        return "A faint tailwind. Possible, with effort."
    if cos <= -0.999:
        return "Exact opposites. Gravity pulls both ways."
    if cos <= -0.8:
        return "Head-on collision course. Bring snacks."
    if cos <= -0.4:
        return "Friction guaranteed, drama optional."
    return "A slight headwind. Communicate."


def fmt_vec(v):
    return "[" + ", ".join(f"{x:g}" for x in v) + "]"


# ----------------------------------------------------------------------------
# BLOCK 14: THEME
# Concept: none from the chapter. One palette shared by every widget so the
# window looks like one piece.
# ----------------------------------------------------------------------------
BG, PANEL, RAISED = "#171923", "#212433", "#2c3044"
FG, MUTED = "#e6e8f2", "#8b90a8"
CYAN, MAGENTA, GREEN, RED, YELLOW, BLUE = "#7dcfff", "#bb9af7", "#9ece6a", "#f7768e", "#e0af68", "#7aa2f7"
GRID = "#363b54"


def apply_theme(root):
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", background=BG, foreground=FG, fieldbackground=PANEL, bordercolor=GRID,
                    lightcolor=PANEL, darkcolor=PANEL, troughcolor=PANEL, font=("TkDefaultFont", 10))
    style.configure("TFrame", background=BG)
    style.configure("Panel.TFrame", background=PANEL)
    style.configure("TLabel", background=BG, foreground=FG)
    style.configure("Muted.TLabel", foreground=MUTED)
    style.configure("Title.TLabel", font=("TkDefaultFont", 16, "bold"), foreground=MAGENTA)
    style.configure("Head.TLabel", font=("TkDefaultFont", 11, "bold"))
    style.configure("TButton", background=RAISED, foreground=FG, padding=(10, 5), borderwidth=0)
    style.map("TButton", background=[("active", BLUE), ("pressed", BLUE)],
              foreground=[("active", BG)])
    style.configure("Accent.TButton", background=MAGENTA, foreground=BG, font=("TkDefaultFont", 10, "bold"))
    style.map("Accent.TButton", background=[("active", CYAN)])
    style.configure("TCombobox", arrowcolor=FG, selectbackground=RAISED, selectforeground=FG)
    style.map("TCombobox", fieldbackground=[("readonly", PANEL)], foreground=[("readonly", FG)])
    style.configure("TNotebook", background=BG, borderwidth=0)
    style.configure("TNotebook.Tab", background=PANEL, foreground=MUTED, padding=(14, 7))
    style.map("TNotebook.Tab", background=[("selected", RAISED)], foreground=[("selected", FG)])
    style.configure("Treeview", background=PANEL, fieldbackground=PANEL, foreground=FG,
                    rowheight=28, borderwidth=0)
    style.configure("Treeview.Heading", background=RAISED, foreground=MUTED, relief="flat")
    style.map("Treeview", background=[("selected", BLUE)], foreground=[("selected", BG)])
    style.configure("Horizontal.TScale", background=BG, troughcolor=RAISED)
    style.map("Horizontal.TScale", background=[("disabled", BG), ("active", MAGENTA)])
    # Some Tk 8.6 builds ignore Treeview row tag colors unless this map is trimmed.
    for opt in ("foreground", "background"):
        trimmed = [e for e in style.map("Treeview", query_opt=opt) if e[:2] != ("!disabled", "!selected")]
        style.map("Treeview", **{opt: trimmed})


def verdict_color(cos, dot):
    if abs(dot) < 1e-9:
        return YELLOW
    return GREEN if cos > 0 else RED


def center_over(dialog, parent):
    dialog.update_idletasks()
    x = parent.winfo_rootx() + (parent.winfo_width() - dialog.winfo_reqwidth()) // 2
    y = parent.winfo_rooty() + (parent.winfo_height() - dialog.winfo_reqheight()) // 2
    dialog.geometry(f"+{max(x, 0)}+{max(y, 0)}")


# ----------------------------------------------------------------------------
# BLOCK 15: JOIN DIALOG (Section 5.1 vectors in R^n, Section 5.3 inner product)
# Concept: each slider sets one coordinate of your vector on the ordered basis.
# While you drag, the dialog computes the direction cosine (u,v)/(‖u‖‖v‖)
# against everyone already in the space, so you watch your best match change
# as your vector rotates.
# ----------------------------------------------------------------------------
class JoinDialog(tk.Toplevel):
    def __init__(self, app):
        super().__init__(app)
        self.app, self.uni = app, app.universe
        self.title("Join the universe")
        self.configure(bg=BG, padx=18, pady=14)
        self.resizable(False, False)
        self.transient(app)
        self.name = tk.StringVar()
        ttk.Label(self, text="Join the universe", style="Title.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(self, text="Name").grid(row=1, column=0, sticky="w", pady=(10, 2))
        entry = ttk.Entry(self, textvariable=self.name, width=28)
        entry.grid(row=1, column=1, sticky="w", pady=(10, 2))
        entry.focus_set()

        self.scales, self.captions = [], []
        row = 2
        for label, text in zip(self.uni.basis, self.uni.statements):
            ttk.Label(self, text=label, style="Head.TLabel").grid(row=row, column=0, columnspan=2, sticky="w", pady=(12, 0))
            ttk.Label(self, text=f"\u201c{text}\u201d", style="Muted.TLabel", wraplength=470).grid(
                row=row + 1, column=0, columnspan=2, sticky="w")
            scale = tk.Scale(self, from_=-2, to=2, orient="horizontal", resolution=1, length=330,
                             showvalue=False, bg=BG, fg=FG, troughcolor=PANEL, highlightthickness=0,
                             activebackground=MAGENTA, command=lambda _v: self.update_preview())
            scale.set(0)
            scale.grid(row=row + 2, column=0, sticky="w")
            caption = ttk.Label(self, text="", width=18)
            caption.grid(row=row + 2, column=1, sticky="w")
            self.scales.append(scale)
            self.captions.append(caption)
            row += 3

        self.preview = ttk.Label(self, text="", justify="left", font=("TkFixedFont", 10))
        self.preview.grid(row=row, column=0, columnspan=2, sticky="w", pady=(14, 4))
        buttons = ttk.Frame(self)
        buttons.grid(row=row + 1, column=0, columnspan=2, sticky="e", pady=(6, 0))
        ttk.Button(buttons, text="Cancel", command=self.destroy).pack(side="left", padx=6)
        ttk.Button(buttons, text="Join", style="Accent.TButton", command=self.submit).pack(side="left")
        self.bind("<Return>", lambda _e: self.submit())
        self.bind("<Escape>", lambda _e: self.destroy())
        self.update_preview()
        center_over(self, app)
        self.grab_set()

    def vector(self):
        return [float(s.get()) for s in self.scales]

    def update_preview(self):
        vec = self.vector()
        for cap, v in zip(self.captions, vec):
            cap.configure(text=f"{int(v):+d}  {SCALE[int(v)]}")
        ip = self.uni.rec.inner
        if not any(vec):
            self.preview.configure(text="All zeros is the zero vector: no direction yet.", foreground=YELLOW)
            return
        ranked = sorted(((cosine(vec, p, ip), n) for n, p in self.uni.profiles.items()), reverse=True)[:3]
        lines = [f"{i}. {n:<26} cos {c:+.2f}" for i, (c, n) in enumerate(ranked, 1)]
        self.preview.configure(text="Closest directions right now\n" + "\n".join(lines), foreground=FG)

    def submit(self):
        name, vec = self.name.get().strip(), self.vector()
        if not name:
            messagebox.showwarning("Name needed", "Type a name first.", parent=self)
            return
        if name in self.uni.profiles:
            messagebox.showwarning("Name taken", "That name already lives in this universe.", parent=self)
            return
        self.uni.profiles[name] = vec
        self.uni.rebuild()
        self.destroy()
        self.app.after_join(name)


# ----------------------------------------------------------------------------
# BLOCK 16: INNER PRODUCT DIALOG (Section 5.3 Definition 5.1)
# Concept: the weighted dot product sum(w_i u_i v_i) is an inner product as
# long as every weight is strictly positive. The Universe rebuilds the
# recommender after any change, which re-verifies all four axioms. A weight
# of zero or below breaks axiom 1 and gets rejected.
# ----------------------------------------------------------------------------
class WeightsDialog(tk.Toplevel):
    def __init__(self, app):
        super().__init__(app)
        self.app, self.uni = app, app.universe
        self.title("Inner product weights")
        self.configure(bg=BG, padx=18, pady=14)
        self.resizable(False, False)
        self.transient(app)
        ttk.Label(self, text="Weighted inner product", style="Title.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(self, text="(u,v) = \u03a3 w\u1d62 u\u1d62 v\u1d62   Weights must be positive.",
                  style="Muted.TLabel").grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 10))
        current = self.uni.weights or [1.0] * self.uni.dim
        self.vars = []
        for i, (label, w) in enumerate(zip(self.uni.basis, current)):
            ttk.Label(self, text=label).grid(row=i + 2, column=0, sticky="w", pady=3)
            var = tk.StringVar(value=f"{w:g}")
            ttk.Entry(self, textvariable=var, width=8).grid(row=i + 2, column=1, sticky="e", padx=(16, 0))
            self.vars.append(var)
        buttons = ttk.Frame(self)
        buttons.grid(row=self.uni.dim + 2, column=0, columnspan=2, sticky="e", pady=(14, 0))
        ttk.Button(buttons, text="Reset to dot product", command=self.reset).pack(side="left", padx=6)
        ttk.Button(buttons, text="Apply", style="Accent.TButton", command=self.apply).pack(side="left")
        self.bind("<Escape>", lambda _e: self.destroy())
        center_over(self, app)
        self.grab_set()

    def apply(self):
        try:
            weights = [float(v.get()) for v in self.vars]
            old, self.uni.weights = self.uni.weights, weights
            try:
                self.uni.rebuild()
            except ValueError:
                self.uni.weights = old
                raise
        except ValueError:
            messagebox.showerror("Not an inner product",
                                 "Every weight must be a number greater than zero, "
                                 "otherwise axiom 1 fails.", parent=self)
            return
        self.destroy()
        self.app.refresh_all()

    def reset(self):
        self.uni.weights = None
        self.uni.rebuild()
        self.destroy()
        self.app.refresh_all()


# ----------------------------------------------------------------------------
# BLOCK 17: MAIN WINDOW LAYOUT
# Concept: none from the chapter. Left: everyone in the space. Right: four
# tabs (Matches, Compare, Math lab, Orthogonal squad) that all read the same
# recommender object, so every tab agrees with every other tab.
# ----------------------------------------------------------------------------
class MatchGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        try:  # sharper text on high-DPI Windows screens
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
        self.title("uhhhhhh")
        self.geometry("1120x720")
        self.minsize(960, 620)
        self.configure(bg=BG)
        apply_theme(self)

        self.universes = {u.title: u for u in UNIVERSES.values()}
        self.universe = next(iter(self.universes.values()))
        self.mode = tk.StringVar(value="cosine")
        self.alpha = tk.DoubleVar(value=0.5)
        self.k = tk.IntVar(value=5)
        self.cmp_a, self.cmp_b = tk.StringVar(), tk.StringVar()

        self.build_topbar()
        self.build_body()
        self.build_statusbar()
        self.reload_universe()

    @property
    def rec(self):
        return self.universe.rec

    def build_topbar(self):
        head = ttk.Frame(self, padding=(16, 12, 16, 0))
        head.pack(fill="x")
        ttk.Label(head, text="uhhhhh", style="Title.TLabel").pack(side="left")
        ttk.Label(head, text="every person is a vector", style="Muted.TLabel").pack(side="left", padx=(10, 0), pady=(6, 0))
        ttk.Button(head, text="Inner product\u2026", command=lambda: WeightsDialog(self)).pack(side="right")

        bar = ttk.Frame(self, padding=(16, 10, 16, 6))
        bar.pack(fill="x")
        ttk.Label(bar, text="Universe").pack(side="left", padx=(0, 6))
        self.universe_box = ttk.Combobox(bar, values=list(self.universes), state="readonly", width=22)
        self.universe_box.pack(side="left", padx=(0, 20))
        self.universe_box.bind("<<ComboboxSelected>>", lambda _e: self.switch_universe())
        ttk.Label(bar, text="Rank by").pack(side="left", padx=(0, 6))
        self.mode_box = ttk.Combobox(bar, values=MatchRecommender.MODES, textvariable=self.mode,
                                     state="readonly", width=10)
        self.mode_box.pack(side="left", padx=(0, 20))
        self.mode_box.bind("<<ComboboxSelected>>", lambda _e: self.on_mode_change())
        ttk.Label(bar, text="Blend: closeness").pack(side="left", padx=(0, 6))
        self.alpha_scale = ttk.Scale(bar, from_=0, to=1, variable=self.alpha, length=130,
                                     command=lambda _v: self.refresh_matches())
        self.alpha_scale.pack(side="left")
        ttk.Label(bar, text="direction").pack(side="left", padx=(6, 0))

    def build_body(self):
        body = ttk.Frame(self, padding=(16, 4, 16, 8))
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left = ttk.Frame(body, style="Panel.TFrame", padding=10)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 12))
        ttk.Label(left, text="People", style="Head.TLabel", background=PANEL).pack(anchor="w")
        self.count_label = ttk.Label(left, text="", style="Muted.TLabel", background=PANEL)
        self.count_label.pack(anchor="w", pady=(0, 6))
        self.people = tk.Listbox(left, width=24, exportselection=False, activestyle="none", bg=PANEL, fg=FG,
                                 selectbackground=BLUE, selectforeground=BG, highlightthickness=0,
                                 borderwidth=0, font=("TkDefaultFont", 11))
        self.people.pack(fill="y", expand=True)
        self.people.bind("<<ListboxSelect>>", lambda _e: self.on_person_select())
        ttk.Button(left, text="+ Join the universe", style="Accent.TButton",
                   command=lambda: JoinDialog(self)).pack(fill="x", pady=(10, 0))

        self.tabs = ttk.Notebook(body)
        self.tabs.grid(row=0, column=1, sticky="nsew")
        self.tab_matches = ttk.Frame(self.tabs, padding=12)
        self.tab_compare = ttk.Frame(self.tabs, padding=12)
        self.tab_math = ttk.Frame(self.tabs, padding=12)
        self.tab_squad = ttk.Frame(self.tabs, padding=12)
        for tab, title in ((self.tab_matches, "Matches"), (self.tab_compare, "Compare"),
                           (self.tab_math, "Math lab"), (self.tab_squad, "Orthogonal squad")):
            self.tabs.add(tab, text=title)
        self.build_matches_tab()
        self.build_compare_tab()
        self.math_text = self.make_text(self.tab_math)
        self.squad_text = self.make_text(self.tab_squad)

    def build_statusbar(self):
        self.status = ttk.Label(self, text="", style="Muted.TLabel", padding=(16, 4, 16, 10))
        self.status.pack(fill="x")
        self.status.configure(text="Tip: double-click a match to compare. \u201c"
                                   + random.choice(LESSONS) + "\u201d")

    def make_text(self, parent):
        box = tk.Text(parent, bg=PANEL, fg=FG, wrap="word", relief="flat", padx=16, pady=14,
                      font=("TkFixedFont", 11), highlightthickness=0, state="disabled", spacing3=3)
        box.pack(fill="both", expand=True)
        box.tag_configure("h", foreground=MAGENTA, font=("TkFixedFont", 12, "bold"), spacing1=2)
        box.tag_configure("ok", foreground=GREEN)
        box.tag_configure("bad", foreground=RED)
        box.tag_configure("dim", foreground=MUTED)
        box.tag_configure("hot", foreground=YELLOW)
        return box

    def write(self, box, chunks):
        box.configure(state="normal")
        box.delete("1.0", "end")
        for text, tag in chunks:
            box.insert("end", text, tag)
        box.configure(state="disabled")


# ----------------------------------------------------------------------------
# BLOCK 18: MATCHES TAB (Sections 5.1 and 5.3)
# Concept: rank everyone against the selected person with the chosen mode
# (cosine, dot, distance, or blend). The cosine and angle columns always show
# direction, whatever the ranking mode, so you can see when modes disagree.
# ----------------------------------------------------------------------------
    def build_matches_tab(self):
        top = ttk.Frame(self.tab_matches)
        top.pack(fill="x")
        self.matches_title = ttk.Label(top, text="", style="Head.TLabel")
        self.matches_title.pack(side="left")
        ttk.Label(top, text="  show top").pack(side="left")
        ttk.Spinbox(top, from_=1, to=20, width=3, textvariable=self.k,
                    command=self.refresh_matches).pack(side="left", padx=6)
        self.mode_help = ttk.Label(top, text="", style="Muted.TLabel")
        self.mode_help.pack(side="right")

        cols = ("rank", "name", "score", "cos", "angle", "verdict")
        self.tree = ttk.Treeview(self.tab_matches, columns=cols, show="headings", selectmode="browse")
        for col, text, width, anchor in (("rank", "#", 40, "center"), ("name", "Name", 190, "w"),
                                         ("score", "Score", 80, "e"), ("cos", "cos \u03b8", 80, "e"),
                                         ("angle", "Angle", 80, "e"), ("verdict", "Verdict", 240, "w")):
            self.tree.heading(col, text=text)
            self.tree.column(col, width=width, anchor=anchor, stretch=(col == "verdict"))
        self.tree.tag_configure("pos", foreground=GREEN)
        self.tree.tag_configure("neg", foreground=RED)
        self.tree.tag_configure("zero", foreground=YELLOW)
        self.tree.pack(fill="both", expand=True, pady=(10, 8))
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self.on_match_select())
        self.tree.bind("<Double-1>", lambda _e: self.tabs.select(self.tab_compare))
        self.best_label = ttk.Label(self.tab_matches, text="", wraplength=720, justify="left")
        self.best_label.pack(anchor="w")

    def score_text(self, score):
        return f"{score:+.2f}" if self.mode.get() in ("cosine", "dot") else f"{score:.3f}"

    def refresh_matches(self):
        name = self.current_person()
        if not name:
            return
        mode = self.mode.get()
        self.mode_help.configure(text=MODE_HELP[mode])
        self.matches_title.configure(text=f"Matches for {name}  {fmt_vec(self.universe.profiles[name])}")
        self.tree.delete(*self.tree.get_children())
        rows = self.rec.recommend(name, self.k.get(), mode, self.alpha.get())
        for rank, (other, score) in enumerate(rows, 1):
            info = self.rec.explain(name, other)
            tag = "zero" if abs(info["dot"]) < 1e-9 else ("pos" if info["dot"] > 0 else "neg")
            self.tree.insert("", "end", iid=other, tags=(tag,),
                             values=(rank, other, self.score_text(score), f"{info['cosine']:+.3f}",
                                     f"{info['angle_deg']:.1f}\u00b0", info["verdict"]))
        if rows:
            best = self.rec.explain(name, rows[0][0])
            self.best_label.configure(text=f"Best pick: {rows[0][0]}. " + quip(best["cosine"], best["dot"]))


# ----------------------------------------------------------------------------
# BLOCK 19: COMPARE TAB (Section 5.1 Length, Distance, Direction)
# Concept: any two vectors in R^n span a plane, so this canvas draws that
# plane with true geometry: u lies on the horizontal axis with length ‖u‖,
# v sits at the angle theta from cos(theta) = (u,v)/(‖u‖‖v‖) with length
# ‖v‖, and the dashed segment between the tips has length d(u,v) = ‖u - v‖.
# The gauge underneath maps cos(theta) in [-1, 1] (guaranteed by
# Cauchy-Schwarz) onto a horizontal track.
# ----------------------------------------------------------------------------
    def build_compare_tab(self):
        pick = ttk.Frame(self.tab_compare)
        pick.pack(fill="x")
        ttk.Label(pick, text="u", foreground=CYAN, font=("TkDefaultFont", 12, "bold")).pack(side="left")
        self.box_a = ttk.Combobox(pick, textvariable=self.cmp_a, state="readonly", width=24)
        self.box_a.pack(side="left", padx=(6, 18))
        ttk.Label(pick, text="v", foreground=MAGENTA, font=("TkDefaultFont", 12, "bold")).pack(side="left")
        self.box_b = ttk.Combobox(pick, textvariable=self.cmp_b, state="readonly", width=24)
        self.box_b.pack(side="left", padx=(6, 0))
        for box in (self.box_a, self.box_b):
            box.bind("<<ComboboxSelected>>", lambda _e: self.refresh_pair())

        self.canvas = tk.Canvas(self.tab_compare, bg=PANEL, highlightthickness=0, height=330)
        self.canvas.pack(fill="both", expand=True, pady=(10, 8))
        self.canvas.bind("<Configure>", lambda _e: self.draw_pair())
        self.readout = ttk.Label(self.tab_compare, text="", font=("TkFixedFont", 11), justify="left")
        self.readout.pack(anchor="w")

    def draw_pair(self):
        c = self.canvas
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        a, b = self.cmp_a.get(), self.cmp_b.get()
        profiles = self.universe.profiles
        if w < 80 or a not in profiles or b not in profiles:
            return
        info = self.rec.explain(a, b)
        la, lb = info["length_a"], info["length_b"]
        theta = math.radians(info["angle_deg"])
        ox, oy = w / 2, h - 80
        longest = max(la, lb, 1e-9)
        scale = min((w / 2 - 130) / longest, (oy - 50) / longest)

        c.create_text(16, 14, anchor="nw", fill=MUTED, font=("TkDefaultFont", 9),
                      text="The plane spanned by u and v, drawn with true lengths and the true angle")
        c.create_line(24, oy, w - 24, oy, fill=GRID)
        c.create_oval(ox - 3, oy - 3, ox + 3, oy + 3, fill=MUTED, outline="")

        ua = (ox + la * scale, oy)
        vb = (ox + lb * scale * math.cos(theta), oy - lb * scale * math.sin(theta))
        c.create_line(*ua, *vb, dash=(5, 4), fill=YELLOW, width=2)
        mid = ((ua[0] + vb[0]) / 2, (ua[1] + vb[1]) / 2)
        c.create_text(mid[0] + 8, mid[1] - 10, anchor="w", fill=YELLOW, font=("TkDefaultFont", 10),
                      text=f"d = {info['distance']:.2f}")

        radius = max(20, min(58, 0.4 * min(la, lb) * scale))
        c.create_arc(ox - radius, oy - radius, ox + radius, oy + radius, start=0, extent=info["angle_deg"],
                     style="arc", outline=YELLOW, width=2)
        half = theta / 2
        c.create_text(ox + (radius + 22) * math.cos(half), oy - (radius + 22) * math.sin(half),
                      fill=YELLOW, font=("TkDefaultFont", 11, "bold"), text=f"{info['angle_deg']:.1f}\u00b0")

        c.create_line(ox, oy, *ua, arrow="last", arrowshape=(13, 15, 5), width=4, fill=CYAN)
        c.create_line(ox, oy, *vb, arrow="last", arrowshape=(13, 15, 5), width=4, fill=MAGENTA)
        c.create_text(ua[0], ua[1] + 20, fill=CYAN, font=("TkDefaultFont", 10, "bold"),
                      text=f"{a}  \u2016u\u2016 = {la:.2f}")
        c.create_text(vb[0], vb[1] - 20, fill=MAGENTA, font=("TkDefaultFont", 10, "bold"),
                      text=f"{b}  \u2016v\u2016 = {lb:.2f}")

        gx0, gx1, gy = w * 0.18, w * 0.82, h - 26
        c.create_rectangle(gx0, gy - 5, (gx0 + gx1) / 2, gy + 5, fill="#4a2a36", outline="")
        c.create_rectangle((gx0 + gx1) / 2, gy - 5, gx1, gy + 5, fill="#2f4a2f", outline="")
        c.create_line((gx0 + gx1) / 2, gy - 10, (gx0 + gx1) / 2, gy + 10, fill=MUTED)
        for x, label in ((gx0, "-1"), ((gx0 + gx1) / 2, "0"), (gx1, "+1")):
            c.create_text(x, gy + 18, fill=MUTED, text=label, font=("TkDefaultFont", 9))
        px = gx0 + (info["cosine"] + 1) / 2 * (gx1 - gx0)
        color = verdict_color(info["cosine"], info["dot"])
        c.create_oval(px - 9, gy - 9, px + 9, gy + 9, fill=color, outline=BG, width=2)
        c.create_text(gx0 - 12, gy, anchor="e", fill=FG, font=("TkDefaultFont", 10), text="cos \u03b8")
        c.create_text(gx1 + 12, gy, anchor="w", fill=color, font=("TkDefaultFont", 11, "bold"),
                      text=f"{info['cosine']:+.3f}")

    def refresh_pair(self):
        self.draw_pair()
        a, b = self.cmp_a.get(), self.cmp_b.get()
        profiles = self.universe.profiles
        if a in profiles and b in profiles:
            info = self.rec.explain(a, b)
            self.readout.configure(
                text=(f"(u,v) = {info['dot']:+.2f}     d(u,v) = {info['distance']:.3f}     "
                      f"\u03b8 = {info['angle_deg']:.1f}\u00b0\n"
                      f"{info['verdict'].capitalize()}. {quip(info['cosine'], info['dot'])}"))
        self.refresh_math()


# ----------------------------------------------------------------------------
# BLOCK 20: MATH LAB TAB (Sections 5.1 and 5.3)
# Concept: Cauchy-Schwarz |(u,v)| <= ‖u‖‖v‖, the triangle inequality
# ‖u+v‖ <= ‖u‖ + ‖v‖, and unit vectors u/‖u‖ whose inner product equals
# cos(theta), evaluated on the pair chosen in the Compare tab.
# ----------------------------------------------------------------------------
    def refresh_math(self):
        a, b = self.cmp_a.get(), self.cmp_b.get()
        profiles = self.universe.profiles
        if a not in profiles or b not in profiles:
            return
        ip = self.rec.inner
        x, y = np.array(profiles[a], float), np.array(profiles[b], float)
        lhs, rhs = abs(ip(x, y)), norm(x, ip) * norm(y, ip)
        lhs2, rhs2 = norm(x + y, ip), norm(x, ip) + norm(y, ip)
        chunks = [(f"u = {a} {fmt_vec(x)}\nv = {b} {fmt_vec(y)}\n", "dim"),
                  ("\nCauchy-Schwarz  (5.3 Theorem 5.3)\n", "h"),
                  (f"|(u,v)| = {lhs:.3f}   \u2264   \u2016u\u2016\u2016v\u2016 = {rhs:.3f}   ", ""),
                  ("holds\n" if lhs <= rhs + 1e-9 else "VIOLATED\n", "ok" if lhs <= rhs + 1e-9 else "bad")]
        if abs(lhs - rhs) < 1e-9 and rhs > 0:
            chunks.append(("Equality: the vectors are parallel, one is a scalar multiple of the other.\n", "hot"))
        chunks += [("\nTriangle inequality  (5.3 Corollary)\n", "h"),
                   (f"\u2016u+v\u2016 = {lhs2:.3f}   \u2264   \u2016u\u2016+\u2016v\u2016 = {rhs2:.3f}   ", ""),
                   ("holds\n" if lhs2 <= rhs2 + 1e-9 else "VIOLATED\n", "ok" if lhs2 <= rhs2 + 1e-9 else "bad"),
                   ("\nUnit vectors  (5.1)\n", "h")]
        if min(norm(x, ip), norm(y, ip)) == 0:
            chunks.append(("One of these is the zero vector, which has no direction.\n", "hot"))
        else:
            ux, uy = unit_vector(x, ip), unit_vector(y, ip)
            chunks += [(f"u/\u2016u\u2016 = {np.round(ux, 3)}   length {norm(ux, ip):.3f}\n"
                        f"v/\u2016v\u2016 = {np.round(uy, 3)}   length {norm(uy, ip):.3f}\n"
                        f"(u/\u2016u\u2016, v/\u2016v\u2016) = {ip(ux, uy):+.3f}   "
                        f"cos \u03b8 = {cosine(x, y, ip):+.3f}\n", ""),
                       ("Normalizing strips the length away. Direction is all that remains.\n", "dim")]
        w = self.universe.weights
        chunks.append((f"\nInner product in use: {'standard dot product' if w is None else 'weights ' + str(w)}\n", "dim"))
        self.write(self.math_text, chunks)


# ----------------------------------------------------------------------------
# BLOCK 21: ORTHOGONAL SQUAD TAB (Section 5.3 Defs 5.4, 5.5, Theorem 5.4)
# Concept: (u,v) = 0 means orthogonal. Starting from each person in turn and
# greedily adding anyone orthogonal to the whole group so far finds a large
# orthogonal set. Theorem 5.4 says a finite orthogonal set of nonzero vectors
# is linearly independent, so the squad can never outgrow the dimension of
# the space, and the matrix rank confirms it.
# ----------------------------------------------------------------------------
    def refresh_squad(self):
        profiles, ip = self.universe.profiles, self.rec.inner
        pairs = orthogonal_pairs(profiles, ip)
        chunks = [(f"Orthogonal pairs, (u,v) = 0   [{len(pairs)} found]\n", "h")]
        chunks += [(f"{a}  \u27c2  {b}\n", "") for a, b in pairs] or [("None. Nobody is on a different wavelength.\n", "dim")]
        squad = []
        for start in profiles:
            trial = []
            for name in [start] + [n for n in profiles if n != start]:
                v = np.array(profiles[name], float)
                if norm(v, ip) > 0 and all(are_orthogonal(v, profiles[m], ip) for m in trial):
                    trial.append(name)
            if len(trial) > len(squad):
                squad = trial
        chunks.append(("\nLargest orthogonal squad\n", "h"))
        if squad:
            rank = np.linalg.matrix_rank(np.vstack([profiles[m] for m in squad]))
            ok = demonstrate_theorem_5_4([profiles[m] for m in squad], ip)
            chunks += [(", ".join(squad) + "\n", "hot"),
                       (f"size {len(squad)}   rank {rank}   dimension of space {self.universe.dim}   ", ""),
                       ("linearly independent\n" if ok else "check failed\n", "ok" if ok else "bad"),
                       ("Theorem 5.4 caps any such squad at the dimension of the space.\n", "dim")]
        self.write(self.squad_text, chunks)


# ----------------------------------------------------------------------------
# BLOCK 22: EVENTS AND REFRESH
# Concept: none from the chapter. Changing the person, the mode, the weights,
# or the universe re-runs the same recommender calls and redraws every tab.
# ----------------------------------------------------------------------------
    def current_person(self):
        sel = self.people.curselection()
        return self.people.get(sel[0]) if sel else None

    def reload_universe(self):
        self.universe_box.set(self.universe.title)
        self.fill_people()
        self.people.selection_set(0)
        self.on_person_select()

    def fill_people(self):
        self.people.delete(0, "end")
        for name in self.universe.profiles:
            self.people.insert("end", name)
        self.count_label.configure(text=f"{len(self.universe.profiles)} vectors in R{self.universe.dim}")
        names = list(self.universe.profiles)
        self.box_a.configure(values=names)
        self.box_b.configure(values=names)
        self.on_mode_change(refresh=False)

    def switch_universe(self):
        self.universe = self.universes[self.universe_box.get()]
        self.reload_universe()

    def on_mode_change(self, refresh=True):
        self.alpha_scale.state(["!disabled"] if self.mode.get() == "blend" else ["disabled"])
        if refresh:
            self.refresh_matches()

    def on_person_select(self):
        name = self.current_person()
        if not name:
            return
        rows = self.rec.recommend(name, 1, self.mode.get(), self.alpha.get())
        self.cmp_a.set(name)
        self.cmp_b.set(rows[0][0] if rows else name)
        self.refresh_all()

    def on_match_select(self):
        sel = self.tree.selection()
        if sel:
            self.cmp_a.set(self.current_person())
            self.cmp_b.set(sel[0])
            self.refresh_pair()

    def after_join(self, name):
        self.fill_people()
        index = list(self.universe.profiles).index(name)
        self.people.selection_clear(0, "end")
        self.people.selection_set(index)
        self.people.see(index)
        self.on_person_select()
        self.tabs.select(self.tab_matches)

    def refresh_all(self):
        self.refresh_matches()
        self.refresh_pair()
        self.refresh_squad()


if __name__ == "__main__":
    MatchGUI().mainloop()