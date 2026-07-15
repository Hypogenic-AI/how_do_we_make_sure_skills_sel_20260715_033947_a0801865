"""Controlled counterfactual tasks with computable ground truth.

Two task families where we can DIAL the elicitation-vs-novelty regime:

  base-b arithmetic:  b=10 is the ELICITATION regime (in-distribution, latent);
                      b in {9,11} is the NOVELTY regime (out-of-distribution
                      procedure that the model has not memorized).  Canonical
                      counterfactual from "Reasoning or Reciting?" (2307.02477).

  substitution cipher: identity permutation = ELICITATION (plain decode);
                       a fixed random permutation = NOVELTY (a genuinely new
                       symbol->symbol procedure to apply).

Everything is verified deterministically against exact ground truth (no LLM judge).
"""
import random
import string
import re

# ----------------------------- base-b arithmetic -----------------------------

def to_base(n, b):
    """Represent non-negative int n in base b as a string of digits (b<=16)."""
    if n == 0:
        return "0"
    digits = "0123456789abcdef"
    out = []
    while n > 0:
        out.append(digits[n % b])
        n //= b
    return "".join(reversed(out))


def gen_arithmetic(b, n_items, seed=42, op="add", n_digits=2):
    """Generate n_items base-b arithmetic problems. Operands drawn so that in
    base-b they have n_digits digits. Returns list of dicts with prompt+answer."""
    rng = random.Random(seed * 1000 + b)
    lo, hi = b ** (n_digits - 1), b ** n_digits - 1
    items = []
    seen = set()
    while len(items) < n_items:
        x = rng.randint(lo, hi)
        y = rng.randint(lo, hi)
        if (x, y) in seen:
            continue
        seen.add((x, y))
        if op == "add":
            res = x + y
            sym = "+"
        else:
            res = x * y
            sym = "*"
        xb, yb, rb = to_base(x, b), to_base(y, b), to_base(res, b)
        items.append({
            "family": "arithmetic", "base": b, "op": op,
            "x": x, "y": y, "x_b": xb, "y_b": yb,
            "question": f"{xb} {sym} {yb}",
            "answer": rb,  # ground-truth in base b
        })
    return items


def arithmetic_skill(b, op="add"):
    """The CORRECT explicit procedure skill for base-b arithmetic."""
    sym = "addition" if op == "add" else "multiplication"
    return (
        f"# SKILL: Base-{b} {sym}\n"
        f"You are working entirely in BASE {b}. Digits range 0..{b-1}; there is NO "
        f"digit '{b}' or higher. Do NOT convert to base 10 in your head as if it were "
        f"base 10.\n"
        f"Procedure for base-{b} {sym} of two numbers written in base {b}:\n"
        f"1. Convert each base-{b} operand to an ordinary integer: for digits d_k..d_0, "
        f"value = sum(d_i * {b}**i).\n"
        f"2. Perform the {sym} on those integers normally.\n"
        f"3. Convert the integer RESULT back to base {b}: repeatedly take result mod {b} "
        f"(rightmost digit) and integer-divide by {b}, until 0; read digits in reverse.\n"
        f"4. Carries/place-values use {b}, not 10 (e.g. in base {b}, the value after digit "
        f"{b-1} rolls over to '10').\n"
        f"Give ONLY the final base-{b} result."
    )


# ----------------------------- substitution cipher -----------------------------

def make_permutation(seed):
    """A fixed random lowercase-letter permutation for the NOVELTY regime."""
    rng = random.Random(seed)
    letters = list(string.ascii_lowercase)
    shuffled = letters[:]
    rng.shuffle(shuffled)
    return dict(zip(letters, shuffled))


IDENTITY_PERM = {c: c for c in string.ascii_lowercase}

_WORDS = ("the quick brown fox jumps over a lazy dog while nine ships sail east "
          "under bright silver moons and cold winter winds carry old songs home "
          "across quiet fields where tall green grass hides small red foxes").split()


def gen_cipher(perm, n_items, seed=42, n_words=4):
    """Encode short phrases with `perm`; task is to DECODE back to plaintext.
    perm maps plaintext->ciphertext, so decoding requires the inverse map."""
    rng = random.Random(seed * 7 + 3)
    inv = {v: k for k, v in perm.items()}
    items = []
    for _ in range(n_items):
        words = [rng.choice(_WORDS) for _ in range(n_words)]
        plain = " ".join(words)
        cipher = "".join(perm.get(c, c) for c in plain)
        items.append({
            "family": "cipher", "plain": plain, "cipher": cipher,
            "question": cipher, "answer": plain, "inv": inv,
        })
    return items


def cipher_skill(perm):
    """The CORRECT explicit decoding-key skill (novelty regime)."""
    inv = {v: k for k, v in perm.items()}
    mapping = ", ".join(f"{c}->{inv[c]}" for c in string.ascii_lowercase)
    return (
        "# SKILL: Substitution-cipher decoding key\n"
        "The text is encoded with a fixed letter substitution. To DECODE, replace each "
        "ciphertext letter using this exact key (ciphertext->plaintext), leaving spaces "
        "unchanged:\n" + mapping + "\n"
        "Apply the key letter-by-letter and output ONLY the decoded plaintext."
    )


# ----------------------------- vocabulary translation -----------------------------
# A matched pair that separates ELICITATION from NOVELTY-INFORMATION:
#   Spanish glossary  -> the model already knows these words (latent) => elicitation
#   invented "Zeth"   -> the glossary is genuinely-new info the model cannot have

# 24 common concepts with their real Spanish translations (latent knowledge control).
SPANISH = {
    "river": "rio", "mountain": "montana", "fire": "fuego", "water": "agua",
    "stone": "piedra", "light": "luz", "night": "noche", "wind": "viento",
    "tree": "arbol", "bird": "pajaro", "king": "rey", "song": "cancion",
    "house": "casa", "sun": "sol", "moon": "luna", "sea": "mar",
    "friend": "amigo", "book": "libro", "day": "dia", "road": "camino",
    "hand": "mano", "heart": "corazon", "city": "ciudad", "star": "estrella",
}
ENGLISH_CONCEPTS = list(SPANISH.keys())


def make_glossary(seed):
    """Invented-language ('Zeth') glossary english->word. Genuinely-new information:
    the mapping is randomly generated, so it cannot be in the model's training data."""
    rng = random.Random(seed)
    cons, vow = "kzvxtplmnrsg", "aeiou"
    used = set()
    gloss = {}
    for e in ENGLISH_CONCEPTS:
        while True:
            w = "".join(rng.choice(cons) + rng.choice(vow) for _ in range(2))
            if w not in used:
                used.add(w)
                gloss[e] = w
                break
    return gloss


def gen_vocab(gloss, n_items, seed=42, n_words=4):
    """Generate translation problems: a foreign phrase -> English. `gloss` maps
    english->foreign; task is to translate foreign back to English."""
    rng = random.Random(seed * 13 + 1)
    inv = {v: k for k, v in gloss.items()}
    items = []
    for _ in range(n_items):
        concepts = [rng.choice(ENGLISH_CONCEPTS) for _ in range(n_words)]
        english = " ".join(concepts)
        foreign = " ".join(gloss[c] for c in concepts)
        items.append({
            "family": "vocab", "question": foreign, "answer": english, "inv": inv,
        })
    return items


def vocab_skill(gloss, lang_name):
    """The CORRECT glossary skill (foreign = English)."""
    inv = {v: k for k, v in gloss.items()}
    lines = "; ".join(f"{f} = {inv[f]}" for f in sorted(inv))
    return (
        f"# SKILL: {lang_name}-to-English glossary\n"
        f"Use this glossary to translate {lang_name} words to English "
        f"({lang_name} word = English meaning):\n" + lines + "\n"
        "Translate each word of the phrase and output ONLY the English translation."
    )


def extract_vocab(response):
    """Last non-empty line, lowercased, letters/spaces only."""
    lines = [l.strip() for l in response.strip().splitlines() if l.strip()]
    if not lines:
        return None
    last = lines[-1].lower()
    last = re.sub(r"^[a-z ]*[:\-]\s*", "", last) if ":" in last[:20] else last
    last = re.sub(r"[^a-z ]", " ", last)
    return " ".join(last.split()) or None


# ----------------------------- length-matched filler -----------------------------

_FILLER_POOL = (
    "Effective communication is a cornerstone of successful collaboration. Teams that "
    "share context openly tend to make better decisions. Regular retrospectives help "
    "surface hidden assumptions. Documentation should be concise yet complete. When in "
    "doubt, prefer clarity over cleverness. Small, frequent commits ease code review. "
    "A well-scoped task is easier to reason about than a sprawling one. Prioritization "
    "is the art of saying no to good ideas so great ones can ship. Feedback loops that "
    "are short and honest accelerate learning across an organization and its partners."
).split()


def length_matched_filler(target_text, seed=42):
    """Produce irrelevant prose of ~the same character length as target_text.
    Used to isolate any pure context-length / prompt-shape effect from content."""
    rng = random.Random(seed)
    target_len = len(target_text)
    words, cur = [], 0
    while cur < target_len:
        w = rng.choice(_FILLER_POOL)
        words.append(w)
        cur += len(w) + 1
    text = "# NOTE: General best-practices memo (unrelated to the task)\n" + " ".join(words)
    return text[:max(target_len, 1)]


# ----------------------------- answer extraction -----------------------------

def extract_arithmetic(response, b):
    """Pull the final base-b number from a model response."""
    valid = set("0123456789abcdef"[:b])
    # Prefer an explicit 'answer' style final token; else last valid token.
    tokens = re.findall(r"[0-9a-fA-F]+", response.lower())
    cand = [t for t in tokens if set(t) <= valid and t != ""]
    if not cand:
        return None
    return cand[-1]


def extract_cipher(response):
    """Take the last non-empty line of lowercase letters/spaces as the decode."""
    lines = [l.strip() for l in response.strip().splitlines() if l.strip()]
    if not lines:
        return None
    last = lines[-1].lower()
    # strip common prefixes like 'decoded:' or 'plaintext:'
    last = re.sub(r"^[a-z ]*[:\-]\s*", "", last) if ":" in last[:20] else last
    last = re.sub(r"[^a-z ]", "", last).strip()
    return last if last else None


def verify(item, response):
    """Deterministic verifier: returns True iff the response matches ground truth."""
    if item["family"] == "arithmetic":
        pred = extract_arithmetic(response, item["base"])
        if pred is None:
            return False
        norm = lambda s: s.lstrip("0") or "0"
        return norm(pred) == norm(item["answer"])
    elif item["family"] == "vocab":
        pred = extract_vocab(response)
        return pred is not None and pred == item["answer"]
    else:
        pred = extract_cipher(response)
        return pred is not None and pred == item["answer"]


if __name__ == "__main__":
    # sanity
    a = gen_arithmetic(9, 3, op="add")
    print("base9 add sample:", a[0]["question"], "=", a[0]["answer"])
    perm = make_permutation(1)
    c = gen_cipher(perm, 2)
    print("cipher sample:", c[0]["cipher"], "->", c[0]["answer"])
    print(cipher_skill(perm)[:120])
