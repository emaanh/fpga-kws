"""Synthetic training data from text-to-speech (Kokoro-82M), for any keywords.

    uv run python -m kws.tts_data                       # the default 10 keywords
    uv run python -m kws.tts_data --keywords "lights on" computer

Speakers are random blends of two of Kokoro's English voices, so there are many more speakers
than base voices. HELDOUT_VOICES never appear in training speakers; validation speakers are
built only from them, so validation measures generalization to unseen voices.

Negatives are synthesized with exactly the same voices and settings as the keywords, so being
synthetic says nothing about the class: the model cannot use "sounds like TTS" as a shortcut.

Output: data/tts/clips.npz with every utterance trimmed to its speech and resampled to 16 kHz.
data.py places the clips in 1 s windows and matches the real dataset's loudness.
"""

import argparse
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import signal
from tqdm import tqdm

from .config import DATA_DIR, KEYWORDS

TTS_DIR = DATA_DIR / "tts"
TTS_SR = 24_000
SR = 16_000

VOICES = [
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole",
    "af_nova", "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir",
    "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa", "bf_alice", "bf_emma",
    "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
]
HELDOUT_VOICES = ["af_nova", "am_eric", "bf_lily", "bm_lewis", "af_river"]

# Speech Commands' other words, so the real "unknown" words also exist as TTS.
SC_OTHER = ["backward", "bed", "bird", "cat", "dog", "eight", "five", "follow", "forward", "four",
            "happy", "house", "learn", "marvin", "nine", "one", "seven", "sheila", "six", "three",
            "tree", "two", "visual", "wow", "zero"]

# Words that sound close to a keyword (exact homophones like know/no or write/right are left
# out on purpose: they are the same sound, so they cannot be negatives).
CONFUSERS = {
    "yes": ["yet", "yeah", "yell", "less", "guess", "bless", "dress", "chess", "yup", "mess"],
    "no": ["nope", "now", "note", "nose", "snow", "slow", "toe", "low", "nor", "gnaw"],
    "up": ["cup", "pup", "sup", "hop", "app", "ape", "uh", "hub", "pop", "op"],
    "down": ["town", "gown", "drown", "dawn", "done", "don", "clown", "brown", "round", "noun"],
    "left": ["lift", "loft", "laughed", "lefty", "theft", "lest", "deft", "west", "lot", "lend"],
    "right": ["light", "bright", "ride", "white", "fight", "rice", "ripe", "height", "rate", "rot"],
    "on": ["own", "an", "in", "gone", "yawn", "con", "john", "odd", "honk", "fawn"],
    "off": ["of", "oft", "cough", "doff", "scoff", "often", "awe", "ought", "loft", "boss"],
    "stop": ["shop", "top", "step", "spot", "stock", "stomp", "strap", "stoop", "sock", "store"],
    "go": ["goat", "grow", "glow", "dough", "toe", "ago", "gold", "coat", "ghost", "goal"],
    "emaan": ["Iman", "amen", "Oman", "human", "lemon", "email", "demon", "Ahmad", "Ramen",
              "salmon", "Amon", "a man", "the man", "Eamon", "Imam", "Shaman"],
    "heidari": ["Haider", "hydra", "Harry", "Atari", "Ferrari", "Hadar", "safari", "Dari",
                "hey Dari", "hey darling", "Hydari", "Ottari", "Lidari", "Hey Dad", "calamari",
                "Kathari"],
}

COMMON = """
about after again air all also always and animal answer any apple area around ask away baby
back ball bank be bear beautiful because been before begin being best better big bit black blue
boat body book both box boy bread break bring brother build busy but buy call came can car card
care carry case catch chair change child city class clean clear close cloud cold color come
common could country course cover cross cry cut dark day dear deep did different dinner do does
door draw dream drink drive dry each early earth easy eat edge egg end enough even evening ever
every eye face fact fall family far farm fast father feel few field fill find fine fire first
fish floor flower fly food foot for found free fresh friend from front full fun game garden gave
get girl give glass good great green ground group grow hair half hand hard have head hear heart
heavy help her here high hill him his hold home horse hot hour how hundred idea if inside into
island it jump just keep kind king kitchen know-how lake land large last late laugh lead learn
leave letter life like line list listen little live long look lost love made make man many map
market may me mean meet milk mind minute miss money month moon more morning most mother mountain
move much music must name near need never new next nice night north nothing number ocean old
only open or order other our out over page paper park part party pass past pay people person
pick picture piece place plan plant play please point power pretty pull push put question quick
quiet rain read ready red remember rest river road rock room run said same sand save say school
sea season second see seem sell send set shape she ship short should show side sing sister sit
sky sleep small smile so some song soon sound south space speak special spring stand star start
stay still story street strong study such summer sun sure table take talk tall teacher tell than
thank that the their them then there these thing think this time today together told tomorrow
took travel try turn under until use very voice wait walk wall want warm was watch water way we
weather week well went were what when where which while why wide will wind window winter wish
with without woman wonder wood word work world would year young your
"""


def negative_words(keywords):
    kw = {k.lower() for k in keywords}
    words = set(SC_OTHER) | set(COMMON.split())
    for k in kw:
        words |= set(CONFUSERS.get(k, []))
    words = {w for w in words if "-" not in w}
    return sorted(words - kw)


# ------------------------------------------------------------------------------------------
# Synthesis workers
# ------------------------------------------------------------------------------------------
_pipes, _voices = {}, {}


def _init_worker(phonemes=None):
    import torch
    from kokoro import KPipeline

    torch.set_num_threads(1)
    PHONEMES.update(phonemes or {})
    for lang in ("a", "b"):
        _pipes[lang] = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M")
    for v in VOICES:
        _voices[v] = _pipes["a"].load_voice(v)


def _trim(x, margin=0.03):
    """Cut leading/trailing silence: keep frames within 40 dB of the peak, plus a margin."""
    frame = 160
    n = len(x) // frame
    if n == 0:
        return x
    e = (x[: n * frame].reshape(n, frame) ** 2).mean(1)
    loud = np.nonzero(e > e.max() * 1e-4)[0]
    m = int(margin * SR)
    return x[max(0, loud[0] * frame - m) : min(len(x), (loud[-1] + 1) * frame + m)]


def _synth(job):
    text, v1, v2, alpha, speed = job
    voice = alpha * _voices[v1] + (1 - alpha) * _voices[v2]
    pipe = _pipes["b" if v1.startswith("b") else "a"]
    audio = np.concatenate([o.audio.numpy() for o in pipe(text, voice=voice, speed=speed)])
    x = _trim(signal.resample_poly(audio.astype(np.float64), 2, 3))  # 24 kHz -> 16 kHz
    x = x[:SR] / (np.abs(x).max() + 1e-9) * 0.5
    return np.round(x * 32767).astype(np.int16)


PHONEMES = {}  # keyword -> Kokoro (misaki) phonemes, for words it would mispronounce


def text_variants(word, rng):
    """Punctuation changes Kokoro's intonation: statement, command, question, plain.
    Words with a pronunciation override use Kokoro's [word](/phonemes/) markup."""
    w = word if rng.random() < 0.5 else word.capitalize()
    if word in PHONEMES:
        w = f"[{w}](/{PHONEMES[word]}/)"
    return w + rng.choice(["", ".", "!", "?", ","])


def make_jobs(keywords, n_kw_train, n_kw_val, n_neg_train, n_neg_val, seed):
    rng = np.random.default_rng(seed)
    train_voices = [v for v in VOICES if v not in HELDOUT_VOICES]

    def speaker(pool):
        v1, v2 = rng.choice(pool, 2, replace=False)
        return str(v1), str(v2), float(rng.uniform(0.2, 0.8))

    jobs = []  # (text, v1, v2, alpha, speed), label word, split
    negs = negative_words(keywords)
    for split, pool, n_kw, n_neg in [("train", train_voices, n_kw_train, n_neg_train),
                                     ("val", HELDOUT_VOICES, n_kw_val, n_neg_val)]:
        for word in keywords:
            for _ in range(n_kw):
                jobs.append(((text_variants(word, rng), *speaker(pool), float(rng.uniform(0.8, 1.25))),
                             word, split))
        for i in range(n_neg):
            word = negs[i % len(negs)] if i < len(negs) else str(rng.choice(negs))
            jobs.append(((text_variants(word, rng), *speaker(pool), float(rng.uniform(0.8, 1.25))),
                         word, split))
    return jobs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--keywords", nargs="*", default=KEYWORDS,
                   help='words, optionally with phonemes: "emaan=imˈɑn"')
    p.add_argument("--kw-train", type=int, default=400, help="train utterances per keyword")
    p.add_argument("--kw-val", type=int, default=60)
    p.add_argument("--neg-train", type=int, default=4000, help="non-keyword train utterances")
    p.add_argument("--neg-val", type=int, default=600)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="clips.npz")
    args = p.parse_args()
    keywords = []
    for k in args.keywords:
        word, _, ph = k.partition("=")
        keywords.append(word.lower())
        if ph:
            PHONEMES[word.lower()] = ph

    jobs = make_jobs(keywords, args.kw_train, args.kw_val, args.neg_train, args.neg_val, args.seed)
    print(f"{len(jobs)} utterances: {len(keywords)} keywords, "
          f"{len(negative_words(keywords))} distinct negative words")
    with ProcessPoolExecutor(args.workers, initializer=_init_worker,
                             initargs=(dict(PHONEMES),)) as pool:
        clips = list(tqdm(pool.map(_synth, [j[0] for j in jobs], chunksize=8), total=len(jobs)))

    TTS_DIR.mkdir(parents=True, exist_ok=True)
    lengths = np.array([len(c) for c in clips])
    np.savez(TTS_DIR / args.out,
             audio=np.concatenate(clips), offsets=np.concatenate([[0], np.cumsum(lengths)[:-1]]),
             lengths=lengths, words=np.array([j[1] for j in jobs]),
             splits=np.array([j[2] for j in jobs]), keywords=np.array(keywords),
             speakers=np.array([f"{j[0][1]}+{j[0][2]}@{j[0][3]:.2f}" for j in jobs]))
    print(f"wrote {TTS_DIR / args.out}: {lengths.sum() / SR / 60:.1f} min of speech, "
          f"median word {np.median(lengths) / SR * 1000:.0f} ms")


if __name__ == "__main__":
    main()
