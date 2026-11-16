"""
CINE VECTORS: a movie recommendation system built on inner product spaces
Linear Algebra, Sections 5.1 - 5.3

Users and movies are vectors in the real inner product space R^8, whose ordered
basis is a list of eight genres. A user's preference vector p says how much they
like each genre (negative means dislike). A movie's feature vector m says how
much each genre shows up in it. The inner product (p, m) measures how well the
two align, and the predicted rating for an unwatched movie comes straight from it.

Run:   python3 movie_recommender.py
Needs: numpy (pip install numpy). tkinter ships with Python; on some Linux
       installs it is a separate package: sudo apt install python3-tk
"""

import copy
import math
import random
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np


# ----------------------------------------------------------------------------
# BLOCK 1: INNER PRODUCT (Section 5.1 Theorem 5.1, Section 5.3 Definition 5.1)
# Concept: the standard inner product on R^n is the dot product. Section 5.3
# generalizes it: any function (u, v) -> real number that satisfies the four
# axioms is an inner product. The weighted dot product, sum(w_i * u_i * v_i),
# satisfies all four as long as every weight is strictly positive. Here the
# weights act as "genre importance": a bigger w_i makes genre i count for more
# in every score, and the space stays a valid inner product space.
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
# The recommender tests them on random vectors before using the inner product,
# so a bad choice of genre weights is caught immediately.
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
# Concept: ||u|| = sqrt((u, u)). For a movie, the norm is how strongly it
# expresses its genres (a pure comedy and a genre-blending epic point the same
# way for a fan but have different lengths). For a user, the norm is how
# strong their tastes are overall.
# ----------------------------------------------------------------------------
def norm(u, inner):
    return float(np.sqrt(inner(u, u)))


# ----------------------------------------------------------------------------
# BLOCK 4: DISTANCE (Section 5.1 Distance, Section 5.3 Definition 5.3)
# Concept: d(u, v) = ||u - v||. Expanding with the axioms gives
#   d(u, v)^2 = ||u||^2 - 2 (u, v) + ||v||^2,
# so ranking movies by distance to a taste vector is the same as ranking by the
# inner product with a penalty for long movie vectors (the Math lab tab checks
# this identity on live numbers).
# ----------------------------------------------------------------------------
def distance(u, v, inner):
    return norm(np.asarray(u, dtype=float) - np.asarray(v, dtype=float), inner)


# ----------------------------------------------------------------------------
# BLOCK 5: UNIT VECTORS (Section 5.1 Unit Vectors)
# Concept: for a nonzero x, x / ||x|| has length 1 and the same direction as x.
# The unit vector p / ||p|| is a user's pure taste direction with the
# "how intensely do they feel" part removed.
# ----------------------------------------------------------------------------
def unit_vector(x, inner):
    x = np.asarray(x, dtype=float)
    n = norm(x, inner)
    if n == 0:
        raise ValueError("The zero vector has no direction.")
    return x / n


# ----------------------------------------------------------------------------
# BLOCK 6: DIRECTION AND ANGLE (Section 5.1 Direction)
# Concept: cos(theta) = (u, v) / (||u|| ||v||), 0 <= theta <= pi. A small angle
# means the movie sits in the same genre direction as the user's taste; an
# angle near 90 degrees means no overlap; near 180 degrees means the movie
# points at everything the user dislikes. Cauchy-Schwarz (Block 7) guarantees
# the ratio lies in [-1, 1], so arccos is always defined.
# ----------------------------------------------------------------------------
def cosine(u, v, inner):
    nu, nv = norm(u, inner), norm(v, inner)
    if nu == 0 or nv == 0:
        return 0.0  # a user with no recorded taste has no direction
    return float(np.clip(inner(u, v) / (nu * nv), -1.0, 1.0))


def angle_degrees(u, v, inner):
    return float(np.degrees(np.arccos(cosine(u, v, inner))))


# ----------------------------------------------------------------------------
# BLOCK 7: CAUCHY-SCHWARZ AND TRIANGLE INEQUALITIES (Section 5.3 Theorem 5.3)
# Concept: |(u, v)| <= ||u|| ||v||  and  ||u + v|| <= ||u|| + ||v||.
# Cauchy-Schwarz is what makes the rating formula in Block 11 safe: dividing
# (p, m) by ||p|| times the longest movie norm always lands in [-1, 1], so a
# predicted rating can never fall outside the 1 to 5 star range.
# ----------------------------------------------------------------------------
def cauchy_schwarz_holds(u, v, inner, tol=1e-9):
    return abs(inner(u, v)) <= norm(u, inner) * norm(v, inner) + tol


def triangle_holds(u, v, inner, tol=1e-9):
    u, v = np.asarray(u, float), np.asarray(v, float)
    return norm(u + v, inner) <= norm(u, inner) + norm(v, inner) + tol


# ----------------------------------------------------------------------------
# BLOCK 8: ORTHOGONALITY (Section 5.3 Definitions 5.4 and 5.5, Theorem 5.4)
# Concept: u and v are orthogonal when (u, v) = 0. A set is orthogonal when
# every distinct pair is orthogonal. Theorem 5.4: a finite orthogonal set of
# nonzero vectors is linearly independent, so such a set in R^8 has at most 8
# vectors. For movies, orthogonal means the two share no genre at all. The
# "variety marathon" finds the biggest set of movies where no two overlap,
# using a clique search over the "is orthogonal to" relation.
# ----------------------------------------------------------------------------
def are_orthogonal(u, v, inner, tol=1e-9):
    return abs(inner(u, v)) < tol


def are_orthogonal_set(vectors, inner, tol=1e-9):
    return all(are_orthogonal(vectors[i], vectors[j], inner, tol)
               for i in range(len(vectors)) for j in range(i + 1, len(vectors)))


def maximal_cliques(adj):
    """Bron-Kerbosch: every maximal set of pairwise-adjacent nodes."""
    found = []

    def expand(R, P, X):
        if not P and not X:
            found.append(R)
            return
        for v in list(P):
            expand(R + [v], P & adj[v], X & adj[v])
            P = P - {v}
            X = X | {v}

    expand([], set(adj), set())
    return found


# ----------------------------------------------------------------------------
# BLOCK 9: DATA (Section 5.1 vectors as coordinates relative to an ordered basis)
# Concept: the ordered basis of R^8 is the genre list below. A movie's
# coordinates are genre intensities from 0 to 1 (1.0 is the defining genre,
# smaller numbers are secondary flavors). A user's coordinates run from -5
# (strongly dislikes the genre) to +5 (loves it). The movie weights are the
# author's own rough estimates, so edit them freely.
# ----------------------------------------------------------------------------
GENRES = ["Action", "Comedy", "Drama", "Sci-Fi", "Horror", "Romance", "Thriller", "Animation"]
N = len(GENRES)

_RAW_MOVIES = [
    ("The Dark Knight", 2008, {"Action": 1.0, "Drama": 0.4, "Thriller": 0.8}),
    ("Inception", 2010, {"Action": 0.6, "Sci-Fi": 1.0, "Thriller": 0.7, "Drama": 0.3}),
    ("Interstellar", 2014, {"Sci-Fi": 1.0, "Drama": 0.8, "Thriller": 0.2}),
    ("The Matrix", 1999, {"Action": 0.9, "Sci-Fi": 1.0, "Thriller": 0.3}),
    ("Mad Max: Fury Road", 2015, {"Action": 1.0, "Sci-Fi": 0.5, "Thriller": 0.3}),
    ("Die Hard", 1988, {"Action": 1.0, "Thriller": 0.7, "Comedy": 0.2}),
    ("John Wick", 2014, {"Action": 1.0, "Thriller": 0.6}),
    ("The Godfather", 1972, {"Drama": 1.0, "Thriller": 0.4}),
    ("Parasite", 2019, {"Drama": 0.8, "Thriller": 0.7, "Comedy": 0.4}),
    ("Forrest Gump", 1994, {"Drama": 1.0, "Romance": 0.5, "Comedy": 0.4}),
    ("The Shawshank Redemption", 1994, {"Drama": 1.0, "Thriller": 0.2}),
    ("Superbad", 2007, {"Comedy": 1.0}),
    ("The Grand Budapest Hotel", 2014, {"Comedy": 0.9, "Drama": 0.4}),
    ("Bridesmaids", 2011, {"Comedy": 1.0, "Romance": 0.3}),
    ("Airplane!", 1980, {"Comedy": 1.0}),
    ("The Notebook", 2004, {"Romance": 1.0, "Drama": 0.7}),
    ("Pride & Prejudice", 2005, {"Romance": 1.0, "Drama": 0.6}),
    ("La La Land", 2016, {"Romance": 0.8, "Drama": 0.6, "Comedy": 0.3}),
    ("Crazy Rich Asians", 2018, {"Romance": 0.8, "Comedy": 0.8, "Drama": 0.2}),
    ("When Harry Met Sally", 1989, {"Romance": 0.9, "Comedy": 0.8}),
    ("Titanic", 1997, {"Romance": 1.0, "Drama": 0.8}),
    ("Get Out", 2017, {"Horror": 0.8, "Thriller": 0.8, "Comedy": 0.2}),
    ("The Conjuring", 2013, {"Horror": 1.0, "Thriller": 0.4}),
    ("A Quiet Place", 2018, {"Horror": 0.8, "Sci-Fi": 0.4, "Thriller": 0.7, "Drama": 0.2}),
    ("Halloween", 1978, {"Horror": 1.0, "Thriller": 0.6}),
    ("Toy Story", 1995, {"Animation": 1.0, "Comedy": 0.6}),
    ("Spirited Away", 2001, {"Animation": 1.0, "Drama": 0.3}),
    ("Inside Out", 2015, {"Animation": 1.0, "Comedy": 0.5, "Drama": 0.5}),
    ("WALL-E", 2008, {"Animation": 1.0, "Sci-Fi": 0.6, "Romance": 0.3}),
    ("Spider-Man: Into the Spider-Verse", 2018, {"Animation": 0.9, "Action": 0.8, "Sci-Fi": 0.4, "Comedy": 0.3}),
    ("Shrek", 2001, {"Animation": 0.9, "Comedy": 0.9, "Romance": 0.3}),
    ("Blade Runner 2049", 2017, {"Sci-Fi": 1.0, "Drama": 0.6, "Thriller": 0.4}),
    ("Arrival", 2016, {"Sci-Fi": 1.0, "Drama": 0.8}),
    ("Se7en", 1995, {"Thriller": 1.0, "Horror": 0.3, "Drama": 0.5}),
    ("Gone Girl", 2014, {"Thriller": 1.0, "Drama": 0.7}),
    ("Knives Out", 2019, {"Thriller": 0.8, "Comedy": 0.6, "Drama": 0.2}),
    ("The Silence of the Lambs", 1991, {"Thriller": 1.0, "Horror": 0.6, "Drama": 0.4}),
    ("Everything Everywhere All at Once", 2022, {"Sci-Fi": 0.8, "Comedy": 0.8, "Action": 0.7, "Drama": 0.7}),
]

MOVIE_YEAR = {t: y for t, y, _ in _RAW_MOVIES}
MOVIE_VEC = {t: np.array([g.get(name, 0.0) for name in GENRES]) for t, _, g in _RAW_MOVIES}

# Sample users: a preference vector on the same ordered basis, plus the movies
# they already watched with their 1-5 star ratings.
SAMPLE_USERS = {
    "Alex": {"prefs": [5, 0, -1, 5, -3, -4, 3, 0],
             "watched": {"The Matrix": 5, "Die Hard": 4, "Interstellar": 5}},
    "Bianca": {"prefs": [-2, 4, 1, -3, -4, 5, -1, 2],
               "watched": {"The Notebook": 5, "Bridesmaids": 4, "La La Land": 4}},
    "Carlos": {"prefs": [1, -1, 1, 0, 5, -3, 5, -2],
               "watched": {"Halloween": 4, "Se7en": 5, "The Conjuring": 4}},
    "Dana": {"prefs": [0, 3, 1, 1, -5, 1, -3, 5],
             "watched": {"Toy Story": 5, "Inside Out": 5, "Shrek": 4}},
    "Eli": {"prefs": [-1, 0, 5, 1, -2, 2, 3, 0],
            "watched": {"The Godfather": 5, "The Shawshank Redemption": 5, "Parasite": 4}},
}

TASTE_RATE = 0.25  # how far one rating moves a preference vector (Block 12)


# ----------------------------------------------------------------------------
# BLOCK 10: THE RECOMMENDER, PREDICTION FORMULA (Sections 5.1 and 5.3)
# Concept: the predicted rating for user p and movie m is
#   stars = 3 + 2 (p, m) / (||p|| * M),   M = the longest movie norm.
# The numerator (p, m) rewards alignment in both direction and magnitude. By
# Cauchy-Schwarz, |(p, m)| <= ||p|| ||m|| <= ||p|| M, so the fraction always
# sits in [-1, 1] and stars always sit in [1, 5]. A user with no taste yet
# (p = 0) gets a neutral 3 stars everywhere.
# ----------------------------------------------------------------------------
class MovieRecommender:
    MODES = ("inner product", "cosine", "distance", "blend")

    def __init__(self, users, weights=None):
        self.users = users                      # live dict, shared with the app
        self.weights = None if weights is None else [float(w) for w in weights]
        self.inner = make_inner_product(self.weights)
        verify_inner_product_axioms(self.inner, N)                       # Block 2
        self.M = max(norm(v, self.inner) for v in MOVIE_VEC.values())    # longest movie
        for name in self.users:                                          # Block 7
            for title in MOVIE_VEC:
                assert cauchy_schwarz_holds(self.pref(name), MOVIE_VEC[title], self.inner)
                assert triangle_holds(self.pref(name), MOVIE_VEC[title], self.inner)

    # ---- vectors ---------------------------------------------------------
    def pref(self, user):
        return np.asarray(self.users[user]["prefs"], dtype=float)

    def watched(self, user):
        return self.users[user]["watched"]

    # ---- prediction ------------------------------------------------------
    def scaled_ip_vec(self, p, m):
        n = norm(p, self.inner)
        return 0.0 if n == 0 else float(np.clip(self.inner(p, m) / (n * self.M), -1.0, 1.0))

    def stars_vec(self, p, m):
        return 3.0 + 2.0 * self.scaled_ip_vec(p, m)

    def stars(self, user, title):
        return self.stars_vec(self.pref(user), MOVIE_VEC[title])

    # ---- ranking modes ---------------------------------------------------
    #   "inner product": (p, m), magnitude and direction together
    #   "cosine"       : direction only
    #   "distance"     : smaller ||p - m|| ranks higher (score is -d)
    #   "blend"        : alpha * cosine + (1 - alpha) * scaled inner product
    def score(self, user, title, mode="inner product", alpha=0.5):
        p, m = self.pref(user), MOVIE_VEC[title]
        if mode == "inner product":
            return self.inner(p, m)
        if mode == "cosine":
            return cosine(p, m, self.inner)
        if mode == "distance":
            return -distance(p, m, self.inner)
        if mode == "blend":
            return alpha * cosine(p, m, self.inner) + (1 - alpha) * self.scaled_ip_vec(p, m)
        raise ValueError(f"mode must be one of {self.MODES}")

    def recommend(self, user, k=5, mode="inner product", alpha=0.5, include_watched=False):
        seen = set() if include_watched else set(self.watched(user))
        rows = [(t, self.score(user, t, mode, alpha)) for t in MOVIE_VEC if t not in seen]
        return sorted(rows, key=lambda r: r[1], reverse=True)[:k]

    def explain(self, user, title):
        p, m = self.pref(user), MOVIE_VEC[title]
        cos = cosine(p, m, self.inner)
        stars = self.stars(user, title)
        return {"dot": self.inner(p, m), "norm_p": norm(p, self.inner), "norm_m": norm(m, self.inner),
                "distance": distance(p, m, self.inner), "cosine": cos,
                "angle_deg": angle_degrees(p, m, self.inner), "stars": stars,
                "verdict": verdict_label(stars)}

    def contributions(self, user, title):
        """Genre-by-genre terms w_i p_i m_i of the inner product (p, m)."""
        w = np.ones(N) if self.weights is None else np.asarray(self.weights)
        terms = w * self.pref(user) * MOVIE_VEC[title]
        return list(zip(GENRES, terms))


# ----------------------------------------------------------------------------
# BLOCK 11: LEARNING FROM RATINGS (Section 5.1 linear combinations of vectors)
# Concept: after a user rates a movie r (1 to 5 stars), their preference vector
# moves by a scalar multiple of that movie's feature vector:
#   p_new = p + 0.25 (r - 3) m.
# Love it (r = 5) and p moves toward m. Hate it (r = 1) and p moves away from
# m. A 3-star rating changes nothing. Over time p becomes a linear combination
# of the movies the user has rated, so taste lives inside the span of what
# they have watched.
# ----------------------------------------------------------------------------
def taste_shift(title, rating):
    return TASTE_RATE * (rating - 3) * MOVIE_VEC[title]


def apply_rating(users, user, title, rating):
    delta = taste_shift(title, rating)
    users[user]["prefs"] = [float(x) for x in np.asarray(users[user]["prefs"], float) + delta]
    users[user]["watched"][title] = rating
    return delta


# ----------------------------------------------------------------------------
# BLOCK 12: VERDICT TEXT AND VARIETY MARATHON (Section 5.1 and Theorem 5.4)
# Concept: the predicted star rating maps to a verdict, and the cosine adds a
# comment on how well the movie's direction lines up with the user's. The
# variety marathon applies Block 8: among unwatched movies, find the largest
# group with pairwise inner product zero (no shared genre), breaking ties by
# the group's total predicted stars. Theorem 5.4 says the group is linearly
# independent, so its size can never exceed the dimension N = 8.
# ----------------------------------------------------------------------------
MODE_HELP = {
    "inner product": "(p, m): magnitude and direction together",
    "cosine": "direction only: cos θ",
    "distance": "closest to your taste vector: smallest ‖p − m‖",
    "blend": "mix of direction and the scaled inner product",
}

LESSONS = [
    "A recommendation is an angle: the smaller it is between you and a movie, the better the night.",
    "Cauchy-Schwarz keeps every predicted rating between 1 and 5 stars.",
    "Orthogonal movies share no genre. Independent picks make the best marathon.",
    "Taste is a vector. Every rating you give rotates it a little.",
]


def stars_label(stars):
    full = int(round(min(5, max(0, stars))))
    return "★" * full + "☆" * (5 - full)


def verdict_label(stars):
    for cut, text in ((4.5, "Perfect match"), (4.0, "Great pick"), (3.5, "Good bet"),
                      (3.0, "Mild interest"), (2.5, "Meh"), (2.0, "Probably skip")):
        if stars >= cut:
            return text
    return "Hard pass"


def movie_quip(stars, cos):
    if abs(cos) < 0.1:
        return "Near-orthogonal to your taste: it neither pulls you in nor pushes you away."
    for cut, text in ((4.5, "Right down the middle of your taste. Press play."),
                      (4.0, "Strong alignment. Safe bet for tonight."),
                      (3.5, "Solid overlap with a few surprises."),
                      (3.0, "Mild pull toward your taste. Fine if nothing else is on."),
                      (2.5, "Slight friction with what you usually like."),
                      (2.0, "Pointing away from your taste. Bring snacks.")):
        if stars >= cut:
            return text
    return "The far end of your taste space. Hard pass."


def fmt_vec(v):
    return "[" + ", ".join(f"{x:g}" for x in v) + "]"


def top_genres(title, count=3):
    vec = MOVIE_VEC[title]
    order = [i for i in np.argsort(-vec) if vec[i] > 0][:count]
    return " · ".join(GENRES[i] for i in order)


def variety_marathon(rec, user):
    candidates = [t for t in MOVIE_VEC if t not in rec.watched(user)]
    adj = {a: {b for b in candidates if b != a and are_orthogonal(MOVIE_VEC[a], MOVIE_VEC[b], rec.inner)}
           for a in candidates}
    cliques = maximal_cliques(adj)
    best = max(cliques, key=lambda c: (len(c), sum(rec.stars(user, t) for t in c)))
    return sorted(best, key=lambda t: -rec.stars(user, t))


# ----------------------------------------------------------------------------
# BLOCK 13: THEME
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
    style.map("TButton", background=[("active", BLUE), ("pressed", BLUE)], foreground=[("active", BG)])
    style.configure("Accent.TButton", background=MAGENTA, foreground=BG, font=("TkDefaultFont", 10, "bold"))
    style.map("Accent.TButton", background=[("active", CYAN)])
    style.configure("TCheckbutton", background=BG, foreground=FG)
    style.map("TCheckbutton", background=[("active", BG)])
    style.configure("TCombobox", arrowcolor=FG, selectbackground=RAISED, selectforeground=FG)
    style.map("TCombobox", fieldbackground=[("readonly", PANEL)], foreground=[("readonly", FG)])
    style.configure("TNotebook", background=BG, borderwidth=0)
    style.configure("TNotebook.Tab", background=PANEL, foreground=MUTED, padding=(14, 7))
    style.map("TNotebook.Tab", background=[("selected", RAISED)], foreground=[("selected", FG)])
    style.configure("Treeview", background=PANEL, fieldbackground=PANEL, foreground=FG, rowheight=28, borderwidth=0)
    style.configure("Treeview.Heading", background=RAISED, foreground=MUTED, relief="flat")
    style.map("Treeview", background=[("selected", BLUE)], foreground=[("selected", BG)])
    style.configure("Horizontal.TScale", background=BG, troughcolor=RAISED)
    style.map("Horizontal.TScale", background=[("disabled", BG), ("active", MAGENTA)])
    # Some Tk 8.6 builds ignore Treeview row tag colors unless this map is trimmed.
    for opt in ("foreground", "background"):
        trimmed = [e for e in style.map("Treeview", query_opt=opt) if e[:2] != ("!disabled", "!selected")]
        style.map("Treeview", **{opt: trimmed})


def stars_color(stars):
    return GREEN if stars >= 3.5 else RED if stars <= 2.5 else YELLOW


def center_over(dialog, parent):
    dialog.update_idletasks()
    x = parent.winfo_rootx() + (parent.winfo_width() - dialog.winfo_reqwidth()) // 2
    y = parent.winfo_rooty() + (parent.winfo_height() - dialog.winfo_reqheight()) // 2
    dialog.geometry(f"+{max(x, 0)}+{max(y, 0)}")


# ----------------------------------------------------------------------------
# BLOCK 14: NEW PROFILE DIALOG (Section 5.1 vectors in R^n)
# Concept: each slider sets one coordinate of the preference vector on the
# genre basis. While you drag, the dialog computes the predicted rating
# 3 + 2 (p, m) / (||p|| M) for every movie, so you watch your top picks change
# as your vector rotates through genre space.
# ----------------------------------------------------------------------------
class NewProfileDialog(tk.Toplevel):
    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.title("New profile")
        self.configure(bg=BG, padx=18, pady=14)
        self.resizable(False, False)
        self.transient(app)
        self.name = tk.StringVar()
        ttk.Label(self, text="New profile", style="Title.TLabel").grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(self, text="Name").grid(row=1, column=0, sticky="w", pady=(10, 2))
        entry = ttk.Entry(self, textvariable=self.name, width=28)
        entry.grid(row=1, column=1, columnspan=2, sticky="w", pady=(10, 2))
        entry.focus_set()
        ttk.Label(self, text="How much do you like each genre?  -5 hate it, +5 love it",
                  style="Muted.TLabel").grid(row=2, column=0, columnspan=3, sticky="w", pady=(10, 4))

        self.scales, self.captions = [], []
        for i, genre in enumerate(GENRES):
            ttk.Label(self, text=genre, width=10).grid(row=3 + i, column=0, sticky="w")
            scale = tk.Scale(self, from_=-5, to=5, orient="horizontal", resolution=1, length=300, showvalue=False,
                             bg=BG, fg=FG, troughcolor=PANEL, highlightthickness=0, activebackground=MAGENTA,
                             command=lambda _v: self.update_preview())
            scale.set(0)
            scale.grid(row=3 + i, column=1, sticky="w")
            caption = ttk.Label(self, text="+0", width=4, anchor="e")
            caption.grid(row=3 + i, column=2, sticky="e")
            self.scales.append(scale)
            self.captions.append(caption)

        self.preview = ttk.Label(self, text="", justify="left", font=("TkFixedFont", 10))
        self.preview.grid(row=3 + N, column=0, columnspan=3, sticky="w", pady=(14, 4))
        buttons = ttk.Frame(self)
        buttons.grid(row=4 + N, column=0, columnspan=3, sticky="e", pady=(6, 0))
        ttk.Button(buttons, text="Cancel", command=self.destroy).pack(side="left", padx=6)
        ttk.Button(buttons, text="Create", style="Accent.TButton", command=self.submit).pack(side="left")
        self.bind("<Return>", lambda _e: self.submit())
        self.bind("<Escape>", lambda _e: self.destroy())
        self.update_preview()
        center_over(self, app)
        self.grab_set()

    def vector(self):
        return [float(s.get()) for s in self.scales]

    def update_preview(self):
        vec = np.array(self.vector())
        for cap, v in zip(self.captions, vec):
            cap.configure(text=f"{int(v):+d}")
        if not vec.any():
            self.preview.configure(text="All zeros is the zero vector: no taste direction yet.", foreground=YELLOW)
            return
        rec = self.app.rec
        ranked = sorted(((rec.stars_vec(vec, m), t) for t, m in MOVIE_VEC.items()), reverse=True)[:3]
        lines = [f"{i}. {t[:30]:<30} {s:.1f} {stars_label(s)}" for i, (s, t) in enumerate(ranked, 1)]
        self.preview.configure(text="Top picks right now\n" + "\n".join(lines), foreground=FG)

    def submit(self):
        name = self.name.get().strip()
        if not name:
            messagebox.showwarning("Name needed", "Type a name first.", parent=self)
            return
        if name in self.app.users:
            messagebox.showwarning("Name taken", "That profile already exists.", parent=self)
            return
        self.app.users[name] = {"prefs": self.vector(), "watched": {}}
        self.destroy()
        self.app.after_new_user(name)


# ----------------------------------------------------------------------------
# BLOCK 15: RATE A MOVIE DIALOG (Block 11, Section 5.1 linear combinations)
# Concept: pick a movie you have seen and give it 1 to 5 stars. The preview
# shows the vector 0.25 (r - 3) m that will be added to your preference
# vector, genre by genre, before you confirm.
# ----------------------------------------------------------------------------
class RateDialog(tk.Toplevel):
    LABELS = {1: "Hated it", 2: "Disliked it", 3: "It was fine", 4: "Liked it", 5: "Loved it"}

    def __init__(self, app, title):
        super().__init__(app)
        self.app, self.user = app, app.current_user()
        self.title("Rate a movie")
        self.configure(bg=BG, padx=18, pady=14)
        self.resizable(False, False)
        self.transient(app)
        ttk.Label(self, text="Rate a movie", style="Title.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(self, text=f"{self.user} just watched", style="Muted.TLabel").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(6, 2))
        unseen = [t for t in MOVIE_VEC if t not in app.rec.watched(self.user)]
        self.movie = tk.StringVar(value=title if title in unseen else unseen[0])
        box = ttk.Combobox(self, textvariable=self.movie, values=unseen, state="readonly", width=38)
        box.grid(row=2, column=0, columnspan=2, sticky="w")
        box.bind("<<ComboboxSelected>>", lambda _e: self.update_preview())
        self.scale = tk.Scale(self, from_=1, to=5, orient="horizontal", resolution=1, length=300, showvalue=False,
                              bg=BG, fg=FG, troughcolor=PANEL, highlightthickness=0, activebackground=MAGENTA,
                              command=lambda _v: self.update_preview())
        self.scale.set(4)
        self.scale.grid(row=3, column=0, sticky="w", pady=(12, 0))
        self.caption = ttk.Label(self, text="", width=22)
        self.caption.grid(row=3, column=1, sticky="w", pady=(12, 0))
        self.preview = ttk.Label(self, text="", justify="left", font=("TkFixedFont", 10))
        self.preview.grid(row=4, column=0, columnspan=2, sticky="w", pady=(12, 4))
        buttons = ttk.Frame(self)
        buttons.grid(row=5, column=0, columnspan=2, sticky="e", pady=(6, 0))
        ttk.Button(buttons, text="Cancel", command=self.destroy).pack(side="left", padx=6)
        ttk.Button(buttons, text="Save rating", style="Accent.TButton", command=self.submit).pack(side="left")
        self.bind("<Escape>", lambda _e: self.destroy())
        self.update_preview()
        center_over(self, app)
        self.grab_set()

    def update_preview(self):
        r = int(self.scale.get())
        self.caption.configure(text=f"{r} {stars_label(r)}  {self.LABELS[r]}")
        delta = taste_shift(self.movie.get(), r)
        moves = sorted(((abs(d), g, d) for g, d in zip(GENRES, delta) if abs(d) > 1e-9), reverse=True)
        if not moves:
            self.preview.configure(text="3 stars is neutral. Your taste vector will not move.")
            return
        lines = [f"{g:<10} {d:+.2f}" for _a, g, d in moves[:4]]
        self.preview.configure(text="Taste vector shifts by 0.25 (r - 3) m:\n" + "\n".join(lines))

    def submit(self):
        apply_rating(self.app.users, self.user, self.movie.get(), int(self.scale.get()))
        self.destroy()
        self.app.after_rating()


# ----------------------------------------------------------------------------
# BLOCK 16: GENRE IMPORTANCE DIALOG (Section 5.3 Definition 5.1)
# Concept: the weighted inner product sum(w_i u_i v_i) is an inner product as
# long as every weight is strictly positive. The recommender is rebuilt after
# any change, which re-verifies all four axioms. A weight of zero or below
# breaks axiom 1 and gets rejected.
# ----------------------------------------------------------------------------
class WeightsDialog(tk.Toplevel):
    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.title("Genre importance")
        self.configure(bg=BG, padx=18, pady=14)
        self.resizable(False, False)
        self.transient(app)
        ttk.Label(self, text="Genre importance", style="Title.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(self, text="(u,v) = Σ wᵢ uᵢ vᵢ   Weights must be positive.",
                  style="Muted.TLabel").grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 10))
        current = app.weights or [1.0] * N
        self.vars = []
        for i, (genre, w) in enumerate(zip(GENRES, current)):
            ttk.Label(self, text=genre).grid(row=i + 2, column=0, sticky="w", pady=3)
            var = tk.StringVar(value=f"{w:g}")
            ttk.Entry(self, textvariable=var, width=8).grid(row=i + 2, column=1, sticky="e", padx=(16, 0))
            self.vars.append(var)
        buttons = ttk.Frame(self)
        buttons.grid(row=N + 2, column=0, columnspan=2, sticky="e", pady=(14, 0))
        ttk.Button(buttons, text="Reset to dot product", command=self.reset).pack(side="left", padx=6)
        ttk.Button(buttons, text="Apply", style="Accent.TButton", command=self.apply).pack(side="left")
        self.bind("<Escape>", lambda _e: self.destroy())
        center_over(self, app)
        self.grab_set()

    def apply(self):
        try:
            self.app.set_weights([float(v.get()) for v in self.vars])
        except (ValueError, AssertionError):
            messagebox.showerror("Not an inner product",
                                 "Every weight must be a number greater than zero, "
                                 "otherwise axiom 1 fails.", parent=self)
            return
        self.destroy()

    def reset(self):
        self.app.set_weights(None)
        self.destroy()


# ----------------------------------------------------------------------------
# BLOCK 17: MAIN WINDOW LAYOUT
# Concept: none from the chapter. Left: every profile. Right: five tabs
# (Recommendations, Compare, Taste profile, Math lab, Variety marathon) that
# all read the same recommender object, so every tab agrees with every other.
# ----------------------------------------------------------------------------
class MovieApp(tk.Tk):
    def __init__(self):
        super().__init__()
        try:  # sharper text on high-DPI Windows screens
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
        self.title("Cine Vectors  |  movie recommendations from inner product spaces")
        self.geometry("1260x780")
        self.minsize(1040, 660)
        self.configure(bg=BG)
        apply_theme(self)

        self.users = copy.deepcopy(SAMPLE_USERS)
        self.weights = None
        self.rec = MovieRecommender(self.users)
        self.mode = tk.StringVar(value="inner product")
        self.alpha = tk.DoubleVar(value=0.5)
        self.k = tk.IntVar(value=8)
        self.show_watched = tk.BooleanVar(value=False)
        self.cmp_movie = tk.StringVar()

        self.build_topbar()
        self.build_body()
        self.build_statusbar()
        self.fill_users()
        self.people.selection_set(0)
        self.on_user_select()

    def build_topbar(self):
        head = ttk.Frame(self, padding=(16, 12, 16, 0))
        head.pack(fill="x")
        ttk.Label(head, text="Cine Vectors", style="Title.TLabel").pack(side="left")
        ttk.Label(head, text="users and movies as vectors in R⁸, one axis per genre",
                  style="Muted.TLabel").pack(side="left", padx=(10, 0), pady=(6, 0))
        ttk.Button(head, text="Genre importance…", command=lambda: WeightsDialog(self)).pack(side="right")

        bar = ttk.Frame(self, padding=(16, 10, 16, 6))
        bar.pack(fill="x")
        ttk.Label(bar, text="Rank by").pack(side="left", padx=(0, 6))
        self.mode_box = ttk.Combobox(bar, values=MovieRecommender.MODES, textvariable=self.mode,
                                     state="readonly", width=14)
        self.mode_box.pack(side="left", padx=(0, 20))
        self.mode_box.bind("<<ComboboxSelected>>", lambda _e: self.on_mode_change())
        ttk.Label(bar, text="Blend: inner product").pack(side="left", padx=(0, 6))
        self.alpha_scale = ttk.Scale(bar, from_=0, to=1, variable=self.alpha, length=130,
                                     command=lambda _v: self.refresh_recs())
        self.alpha_scale.pack(side="left")
        ttk.Label(bar, text="direction").pack(side="left", padx=(6, 20))
        ttk.Checkbutton(bar, text="Include watched movies", variable=self.show_watched,
                        command=self.refresh_recs).pack(side="left")

    def build_body(self):
        body = ttk.Frame(self, padding=(16, 4, 16, 8))
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left = ttk.Frame(body, style="Panel.TFrame", padding=10)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 12))
        ttk.Label(left, text="Profiles", style="Head.TLabel", background=PANEL).pack(anchor="w")
        self.count_label = ttk.Label(left, text="", style="Muted.TLabel", background=PANEL)
        self.count_label.pack(anchor="w", pady=(0, 6))
        self.people = tk.Listbox(left, width=22, exportselection=False, activestyle="none", bg=PANEL, fg=FG,
                                 selectbackground=BLUE, selectforeground=BG, highlightthickness=0,
                                 borderwidth=0, font=("TkDefaultFont", 11))
        self.people.pack(fill="y", expand=True)
        self.people.bind("<<ListboxSelect>>", lambda _e: self.on_user_select())
        ttk.Button(left, text="Rate a movie…", command=lambda: RateDialog(self, self.cmp_movie.get())
                   ).pack(fill="x", pady=(10, 0))
        ttk.Button(left, text="+ New profile", style="Accent.TButton",
                   command=lambda: NewProfileDialog(self)).pack(fill="x", pady=(6, 0))

        self.tabs = ttk.Notebook(body)
        self.tabs.grid(row=0, column=1, sticky="nsew")
        self.tab_recs = ttk.Frame(self.tabs, padding=12)
        self.tab_compare = ttk.Frame(self.tabs, padding=12)
        self.tab_profile = ttk.Frame(self.tabs, padding=12)
        self.tab_math = ttk.Frame(self.tabs, padding=12)
        self.tab_variety = ttk.Frame(self.tabs, padding=12)
        for tab, title in ((self.tab_recs, "Recommendations"), (self.tab_compare, "Compare"),
                           (self.tab_profile, "Taste profile"), (self.tab_math, "Math lab"),
                           (self.tab_variety, "Variety marathon")):
            self.tabs.add(tab, text=title)
        self.build_recs_tab()
        self.build_compare_tab()
        self.build_profile_tab()
        self.math_text = self.make_text(self.tab_math)
        self.variety_text = self.make_text(self.tab_variety)

    def build_statusbar(self):
        self.status = ttk.Label(self, text="", style="Muted.TLabel", padding=(16, 4, 16, 10))
        self.status.pack(fill="x")
        self.status.configure(text="Tip: double-click a recommendation to compare it. “" + random.choice(LESSONS) + "”")

    def make_text(self, parent, height=None, expand=True, pady=0):
        holder = ttk.Frame(parent)
        holder.pack(fill="both", expand=expand, pady=pady)
        box = tk.Text(holder, bg=PANEL, fg=FG, wrap="word", relief="flat", padx=16, pady=14,
                      font=("TkFixedFont", 11), highlightthickness=0, state="disabled", spacing3=3,
                      **({"height": height} if height else {}))
        bar = ttk.Scrollbar(holder, command=box.yview)
        box.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        box.pack(side="left", fill="both", expand=True)
        box.tag_configure("h", foreground=MAGENTA, font=("TkFixedFont", 12, "bold"), spacing1=12)
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
# BLOCK 18: RECOMMENDATIONS TAB (Sections 5.1 and 5.3)
# Concept: rank movies against the selected user's preference vector using the
# chosen mode. The cos and angle columns always show direction, and the star
# column always shows the Cauchy-Schwarz-scaled prediction from Block 10,
# whatever the ranking mode, so you can see when the modes disagree.
# ----------------------------------------------------------------------------
    def build_recs_tab(self):
        top = ttk.Frame(self.tab_recs)
        top.pack(fill="x")
        self.recs_title = ttk.Label(top, text="", style="Head.TLabel")
        self.recs_title.pack(side="left")
        ttk.Label(top, text="  show top").pack(side="left")
        ttk.Spinbox(top, from_=1, to=40, width=3, textvariable=self.k,
                    command=self.refresh_recs).pack(side="left", padx=6)
        self.mode_help = ttk.Label(top, text="", style="Muted.TLabel")
        self.mode_help.pack(side="right")

        cols = ("rank", "title", "genres", "stars", "score", "cos", "angle")
        self.tree = ttk.Treeview(self.tab_recs, columns=cols, show="headings", selectmode="browse")
        for col, text, width, anchor in (("rank", "#", 36, "center"), ("title", "Movie", 300, "w"),
                                         ("genres", "Genres", 190, "w"), ("stars", "Predicted rating", 140, "w"),
                                         ("score", "Score", 76, "e"), ("cos", "cos θ", 66, "e"),
                                         ("angle", "Angle", 60, "e")):
            self.tree.heading(col, text=text)
            self.tree.column(col, width=width, anchor=anchor, stretch=(col == "title"))
        self.tree.tag_configure("pos", foreground=GREEN)
        self.tree.tag_configure("neg", foreground=RED)
        self.tree.tag_configure("mid", foreground=YELLOW)
        self.tree.pack(fill="both", expand=True, pady=(10, 8))
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self.on_rec_select())
        self.tree.bind("<Double-1>", lambda _e: self.tabs.select(self.tab_compare))
        self.best_label = ttk.Label(self.tab_recs, text="", wraplength=760, justify="left")
        self.best_label.pack(anchor="w")

    def score_text(self, key, mode):
        if mode == "distance":
            return f"d = {-key:.2f}"
        return f"{key:+.2f}" if mode == "inner product" else f"{key:+.3f}"

    def refresh_recs(self):
        user = self.current_user()
        if not user:
            return
        mode = self.mode.get()
        self.mode_help.configure(text=MODE_HELP[mode])
        self.recs_title.configure(text=f"Picks for {user}")
        self.tree.delete(*self.tree.get_children())
        rows = self.rec.recommend(user, self.k.get(), mode, self.alpha.get(), self.show_watched.get())
        seen = self.rec.watched(user)
        for rank, (title, key) in enumerate(rows, 1):
            info = self.rec.explain(user, title)
            label = f"{title} ({MOVIE_YEAR[title]})" + ("  ✓ seen" if title in seen else "")
            tag = "pos" if info["stars"] >= 3.5 else "neg" if info["stars"] <= 2.5 else "mid"
            self.tree.insert("", "end", iid=title, tags=(tag,),
                             values=(rank, label, top_genres(title), f"{info['stars']:.1f}  {stars_label(info['stars'])}",
                                     self.score_text(key, mode), f"{info['cosine']:+.3f}", f"{info['angle_deg']:.1f}°"))
        if rows:
            best = self.rec.explain(user, rows[0][0])
            self.best_label.configure(text=f"Top pick: {rows[0][0]}. " + movie_quip(best["stars"], best["cosine"]))


# ----------------------------------------------------------------------------
# BLOCK 19: COMPARE TAB (Section 5.1 Length, Direction, Unit Vectors)
# Concept: any two vectors in R^8 span a plane, so this canvas draws that
# plane with true geometry. The cyan arrow is the unit taste direction
# p / ||p||. The magenta arrow is the movie m, drawn at its true length ||m||
# and at the true angle theta from cos(theta) = (p, m) / (||p|| ||m||). The
# yellow segment is the movie's pull along your taste, (m, p / ||p||) =
# ||m|| cos(theta), the component of m in your taste direction. The gauge
# underneath maps cos(theta) in [-1, 1] (guaranteed by Cauchy-Schwarz).
# ----------------------------------------------------------------------------
    def build_compare_tab(self):
        pick = ttk.Frame(self.tab_compare)
        pick.pack(fill="x")
        self.cmp_user_label = ttk.Label(pick, text="", foreground=CYAN, font=("TkDefaultFont", 11, "bold"))
        self.cmp_user_label.pack(side="left", padx=(0, 18))
        ttk.Label(pick, text="Movie", foreground=MAGENTA, font=("TkDefaultFont", 11, "bold")).pack(side="left")
        self.box_movie = ttk.Combobox(pick, textvariable=self.cmp_movie, values=list(MOVIE_VEC),
                                      state="readonly", width=38)
        self.box_movie.pack(side="left", padx=(6, 0))
        self.box_movie.bind("<<ComboboxSelected>>", lambda _e: self.refresh_pair())

        self.canvas = tk.Canvas(self.tab_compare, bg=PANEL, highlightthickness=0, height=330)
        self.canvas.pack(fill="both", expand=True, pady=(10, 8))
        self.canvas.bind("<Configure>", lambda _e: self.draw_pair())
        self.readout = ttk.Label(self.tab_compare, text="", font=("TkFixedFont", 11), justify="left")
        self.readout.pack(anchor="w")

    def draw_pair(self):
        c = self.canvas
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        user, title = self.current_user(), self.cmp_movie.get()
        if w < 80 or not user or title not in MOVIE_VEC:
            return
        info = self.rec.explain(user, title)
        lm, theta = info["norm_m"], math.radians(info["angle_deg"])
        ox, oy = w / 2, h - 80
        longest = max(1.0, lm)
        scale = min((w / 2 - 140) / longest, (oy - 70) / longest)

        c.create_line(24, oy, w - 24, oy, fill=GRID)
        tip_p = (ox + scale, oy)
        tip_m = (ox + lm * scale * math.cos(theta), oy - lm * scale * math.sin(theta))
        drop = (tip_m[0], oy)
        c.create_line(ox, oy, *drop, fill="#6b5a2e", width=9, capstyle="round")
        c.create_line(*tip_m, *drop, dash=(5, 4), fill=YELLOW, width=2)
        c.create_oval(ox - 3, oy - 3, ox + 3, oy + 3, fill=MUTED, outline="")

        radius = max(20, min(58, 0.4 * min(1.0, lm) * scale))
        c.create_arc(ox - radius, oy - radius, ox + radius, oy + radius, start=0, extent=info["angle_deg"],
                     style="arc", outline=YELLOW, width=2)
        half = theta / 2
        c.create_text(ox + (radius + 22) * math.cos(half), oy - (radius + 22) * math.sin(half),
                      fill=YELLOW, font=("TkDefaultFont", 11, "bold"), text=f"{info['angle_deg']:.1f}°")
        c.create_line(ox, oy, *tip_p, arrow="last", arrowshape=(13, 15, 5), width=4, fill=CYAN)
        c.create_line(ox, oy, *tip_m, arrow="last", arrowshape=(13, 15, 5), width=4, fill=MAGENTA)

        pull = lm * math.cos(theta)
        legend = ((CYAN, f"{user}'s taste direction  p/‖p‖   (‖p‖ = {info['norm_p']:.2f})"),
                  (MAGENTA, f"{title}  m   (‖m‖ = {lm:.2f})"),
                  (YELLOW, f"pull along your taste  (m, p/‖p‖) = {pull:+.2f}"))
        c.create_text(16, 12, anchor="nw", fill=MUTED, font=("TkDefaultFont", 9),
                      text="The plane spanned by p and m, drawn with the true angle and the true movie length")
        for i, (color, text) in enumerate(legend):
            c.create_text(16, 36 + i * 20, anchor="nw", fill=color, font=("TkDefaultFont", 10, "bold"), text="━  " + text)

        gx0, gx1, gy = w * 0.18, w * 0.82, h - 26
        mid = (gx0 + gx1) / 2
        c.create_rectangle(gx0, gy - 5, mid, gy + 5, fill="#4a2a36", outline="")
        c.create_rectangle(mid, gy - 5, gx1, gy + 5, fill="#2f4a2f", outline="")
        c.create_line(mid, gy - 10, mid, gy + 10, fill=MUTED)
        for x, label in ((gx0, "-1"), (mid, "0"), (gx1, "+1")):
            c.create_text(x, gy + 18, fill=MUTED, text=label, font=("TkDefaultFont", 9))
        px = gx0 + (info["cosine"] + 1) / 2 * (gx1 - gx0)
        color = stars_color(info["stars"])
        c.create_oval(px - 9, gy - 9, px + 9, gy + 9, fill=color, outline=BG, width=2)
        c.create_text(gx0 - 12, gy, anchor="e", fill=FG, font=("TkDefaultFont", 10), text="cos θ")
        c.create_text(gx1 + 12, gy, anchor="w", fill=color, font=("TkDefaultFont", 11, "bold"),
                      text=f"{info['cosine']:+.3f}")

    def refresh_pair(self):
        user, title = self.current_user(), self.cmp_movie.get()
        self.cmp_user_label.configure(text=f"{user}" if user else "")
        self.draw_pair()
        if user and title in MOVIE_VEC:
            i = self.rec.explain(user, title)
            seen = self.rec.watched(user).get(title)
            seen_text = f"   (already watched, rated {seen} {stars_label(seen)})" if seen else ""
            self.readout.configure(
                text=(f"(p,m) = {i['dot']:+.2f}    ‖p‖ = {i['norm_p']:.2f}    ‖m‖ = {i['norm_m']:.2f}    "
                      f"d(p,m) = {i['distance']:.2f}    θ = {i['angle_deg']:.1f}°\n"
                      f"Predicted rating {i['stars']:.1f} / 5  {stars_label(i['stars'])}   "
                      f"{i['verdict']}{seen_text}\n{movie_quip(i['stars'], i['cosine'])}"))
        self.refresh_math()


# ----------------------------------------------------------------------------
# BLOCK 20: TASTE PROFILE TAB (Section 5.1 coordinates relative to a basis)
# Concept: the preference vector shown coordinate by coordinate. Each bar is
# one component p_i along the genre basis vector e_i; the length of the whole
# vector is ||p||. Rating movies in the app adds multiples of movie vectors
# (Block 11), so you can watch these bars move.
# ----------------------------------------------------------------------------
    def build_profile_tab(self):
        self.profile_canvas = tk.Canvas(self.tab_profile, bg=PANEL, highlightthickness=0, height=310)
        self.profile_canvas.pack(fill="both", expand=True)
        self.profile_canvas.bind("<Configure>", lambda _e: self.draw_profile())
        self.profile_text = self.make_text(self.tab_profile, height=7, expand=False, pady=(10, 0))

    def draw_profile(self):
        c = self.profile_canvas
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        user = self.current_user()
        if w < 120 or not user:
            return
        p = self.rec.pref(user)
        limit = max(5.0, float(np.max(np.abs(p))))
        left, right, top = 130, w - 70, 56
        mid = (left + right) / 2
        row = (h - top - 14) / N
        c.create_text(16, 12, anchor="nw", fill=FG, font=("TkDefaultFont", 11, "bold"),
                      text=f"{user}'s preference vector   p = {fmt_vec(p)}")
        c.create_text(16, 32, anchor="nw", fill=MUTED, font=("TkDefaultFont", 9),
                      text=f"‖p‖ = {norm(p, self.rec.inner):.2f}   (one bar per coordinate along the genre basis)")
        c.create_line(mid, top - 6, mid, h - 8, fill=GRID)
        for i, genre in enumerate(GENRES):
            y = top + i * row + row / 2
            c.create_text(left - 14, y, anchor="e", fill=FG, font=("TkDefaultFont", 10), text=genre)
            x = mid + p[i] / limit * (right - mid)
            color = GREEN if p[i] > 0 else RED if p[i] < 0 else MUTED
            c.create_rectangle(min(mid, x), y - row * 0.28, max(mid, x), y + row * 0.28, fill=color, outline="")
            c.create_text(x + (10 if p[i] >= 0 else -10), y, anchor="w" if p[i] >= 0 else "e", fill=color,
                          font=("TkDefaultFont", 10, "bold"), text=f"{p[i]:+.2f}".rstrip("0").rstrip("."))

    def refresh_profile(self):
        self.draw_profile()
        user = self.current_user()
        if not user:
            return
        seen = self.rec.watched(user)
        chunks = [("Watched and rated\n", "h")]
        if seen:
            chunks += [(f"{t} ({MOVIE_YEAR[t]})  {stars_label(r)}\n", "") for t, r in seen.items()]
        else:
            chunks.append(("Nothing yet. Rate a movie to start steering this vector.\n", "dim"))
        self.write(self.profile_text, chunks)


# ----------------------------------------------------------------------------
# BLOCK 21: MATH LAB TAB (Sections 5.1 and 5.3)
# Concept: the theory on live numbers for the selected user and movie:
#   - (p, m) split genre by genre, the sum of terms w_i p_i m_i
#   - Cauchy-Schwarz |(p,m)| <= ||p|| ||m||, the bound behind the 1-5 stars
#   - the prediction formula 3 + 2 (p, m) / (||p|| M)
#   - the triangle inequality ||p + m|| <= ||p|| + ||m||
#   - the distance identity d^2 = ||p||^2 - 2 (p,m) + ||m||^2
#   - unit vectors, whose inner product is cos(theta)
# ----------------------------------------------------------------------------
    def refresh_math(self):
        user, title = self.current_user(), self.cmp_movie.get()
        if not user or title not in MOVIE_VEC:
            return
        ip, p, m = self.rec.inner, self.rec.pref(user), MOVIE_VEC[title]
        chunks = [(f"p = {user} {fmt_vec(p)}\nm = {title} {fmt_vec(m)}\n", "dim"),
                  ("Where (p, m) comes from, genre by genre\n", "h")]
        terms = [(g, t) for g, t in self.rec.contributions(user, title) if abs(t) > 1e-9]
        top = max([abs(t) for _g, t in terms] + [1e-9])
        for genre, t in terms:
            chunks.append((f"{genre:<10}{t:+7.2f}  " + "█" * max(1, round(abs(t) / top * 24)) + "\n",
                           "ok" if t > 0 else "bad"))
        if not terms:
            chunks.append(("No shared genre: every term is zero.\n", "hot"))
        dot = ip(p, m)
        chunks.append((f"{'total (p,m)':<10}{dot:+7.2f}\n", ""))

        bound1, bound2 = norm(p, ip) * norm(m, ip), norm(p, ip) * self.rec.M
        holds = abs(dot) <= bound1 + 1e-9
        chunks += [("Cauchy-Schwarz (5.3 Theorem 5.3)\n", "h"),
                   (f"|(p,m)| = {abs(dot):.3f}   ≤   ‖p‖‖m‖ = {bound1:.3f}   ≤   ‖p‖M = {bound2:.3f}   ", ""),
                   ("holds\n" if holds else "VIOLATED\n", "ok" if holds else "bad"),
                   (f"M = {self.rec.M:.3f} is the longest movie vector in the catalog.\n", "dim"),
                   ("Prediction formula\n", "h"),
                   (f"stars = 3 + 2 (p,m) / (‖p‖ M) = 3 + 2 · {dot:+.2f} / {bound2:.2f} = "
                    f"{self.rec.stars(user, title):.2f}\n" if bound2 > 0 else "stars = 3 (no taste yet)\n", ""),
                   ("Because of the bound above, the fraction stays in [-1, 1] and stars stay in [1, 5].\n", "dim")]

        tri_l, tri_r = norm(p + m, ip), norm(p, ip) + norm(m, ip)
        chunks += [("Triangle inequality (5.3 Corollary)\n", "h"),
                   (f"‖p+m‖ = {tri_l:.3f}   ≤   ‖p‖+‖m‖ = {tri_r:.3f}   ", ""),
                   ("holds\n" if tri_l <= tri_r + 1e-9 else "VIOLATED\n", "ok" if tri_l <= tri_r + 1e-9 else "bad")]

        d = distance(p, m, ip)
        rhs = norm(p, ip) ** 2 - 2 * dot + norm(m, ip) ** 2
        chunks += [("Distance identity (5.1 Distance, axioms of 5.3)\n", "h"),
                   (f"d(p,m)² = {d * d:.3f}   =   ‖p‖² − 2(p,m) + ‖m‖² = {rhs:.3f}\n", ""),
                   ("So ranking by distance equals ranking by the inner product minus a length penalty.\n", "dim"),
                   ("Unit vectors (5.1)\n", "h")]
        if min(norm(p, ip), norm(m, ip)) == 0:
            chunks.append(("The user has no taste yet (zero vector), so there is no direction to compare.\n", "hot"))
        else:
            up, um = unit_vector(p, ip), unit_vector(m, ip)
            chunks += [(f"(p/‖p‖, m/‖m‖) = {ip(up, um):+.3f}   cos θ = {cosine(p, m, ip):+.3f}\n", ""),
                       ("Normalizing strips the length away. Direction is all that remains.\n", "dim")]
        w = self.weights
        chunks.append((f"\nInner product in use: {'standard dot product' if w is None else 'weights ' + str(w)}\n", "dim"))
        self.write(self.math_text, chunks)


# ----------------------------------------------------------------------------
# BLOCK 22: VARIETY MARATHON TAB (Section 5.3 Defs 5.4, 5.5, Theorem 5.4)
# Concept: two things here. First, the "neutral zone": unwatched movies whose
# angle to your taste is within 6 degrees of 90, so (p, m) is close to zero and
# the movie neither pulls you in nor pushes you away. Second, the variety
# marathon: the largest group of unwatched movies that is pairwise orthogonal.
# Theorem 5.4 says a finite orthogonal set of nonzero vectors is linearly
# independent, so the group can never outgrow the dimension of the space, and
# the matrix rank confirms it.
# ----------------------------------------------------------------------------
    def refresh_variety(self):
        user = self.current_user()
        if not user:
            return
        ip = self.rec.inner
        p = self.rec.pref(user)
        seen = self.rec.watched(user)
        neutral = [(t, self.rec.explain(user, t)) for t in MOVIE_VEC if t not in seen]
        neutral = [(t, i) for t, i in neutral if abs(i["cosine"]) < 0.105]
        chunks = [(f"Neutral zone for {user}: (p,m) ≈ 0\n", "h")]
        if norm(p, ip) == 0:
            chunks.append(("No taste yet, so every movie is orthogonal to a zero vector.\n", "dim"))
        elif neutral:
            chunks += [(f"{t:<36} cos θ = {i['cosine']:+.2f}   angle {i['angle_deg']:.0f}°\n", "") for t, i in neutral]
        else:
            chunks.append(("No unwatched movie sits at a right angle to this taste.\n", "dim"))

        squad = variety_marathon(self.rec, user)
        rank = np.linalg.matrix_rank(np.vstack([MOVIE_VEC[t] for t in squad]))
        ok = bool(are_orthogonal_set([MOVIE_VEC[t] for t in squad], ip)) and rank == len(squad)
        chunks.append(("Variety marathon: no two movies share a genre\n", "h"))
        for t in squad:
            s = self.rec.stars(user, t)
            chunks.append((f"{t:<36} {s:.1f} {stars_label(s)}   {top_genres(t)}\n", ""))
        chunks += [(f"\nsize {len(squad)}   rank {rank}   dimension of the space {N}   ", ""),
                   ("pairwise orthogonal, linearly independent\n" if ok else "check failed\n", "ok" if ok else "bad"),
                   ("Theorem 5.4 caps any such marathon at the dimension of the space.\n", "dim")]
        self.write(self.variety_text, chunks)


# ----------------------------------------------------------------------------
# BLOCK 23: EVENTS AND REFRESH
# Concept: none from the chapter. Changing the user, the mode, the weights, or
# a rating re-runs the same recommender calls and redraws every tab.
# ----------------------------------------------------------------------------
    def current_user(self):
        sel = self.people.curselection()
        return self.people.get(sel[0]) if sel else None

    def fill_users(self):
        self.people.delete(0, "end")
        for name in self.users:
            self.people.insert("end", name)
        self.count_label.configure(text=f"{len(self.users)} preference vectors in R{N}")
        self.on_mode_change(refresh=False)

    def on_mode_change(self, refresh=True):
        self.alpha_scale.state(["!disabled"] if self.mode.get() == "blend" else ["disabled"])
        if refresh:
            self.refresh_recs()

    def on_user_select(self):
        user = self.current_user()
        if not user:
            return
        rows = self.rec.recommend(user, 1, self.mode.get(), self.alpha.get())
        self.cmp_movie.set(rows[0][0] if rows else next(iter(MOVIE_VEC)))
        self.refresh_all()

    def on_rec_select(self):
        sel = self.tree.selection()
        if sel:
            self.cmp_movie.set(sel[0])
            self.refresh_pair()

    def set_weights(self, weights):
        new = MovieRecommender(self.users, weights)   # raises if an axiom fails
        self.weights, self.rec = weights, new
        self.refresh_all()

    def after_new_user(self, name):
        self.fill_users()
        index = list(self.users).index(name)
        self.people.selection_clear(0, "end")
        self.people.selection_set(index)
        self.people.see(index)
        self.on_user_select()
        self.tabs.select(self.tab_recs)

    def after_rating(self):
        self.on_user_select()
        self.tabs.select(self.tab_profile)

    def refresh_all(self):
        self.refresh_recs()
        self.refresh_pair()
        self.refresh_profile()
        self.refresh_variety()


if __name__ == "__main__":
    MovieApp().mainloop()