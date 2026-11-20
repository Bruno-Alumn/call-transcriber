"""Small built-in word lists for counted language features.

The categories follow the spirit of LIWC (Pennebaker et al.) but are our own
short lists, tuned for casual spoken English in voice chat (Whisper output).
All entries are lowercase and matched as whole tokens produced by
`tokenize`. A few multi-word entries ("you know", "kind of") are matched as
consecutive tokens.

`counts(tokens)` returns per-category counts keyed by the short names in
CATEGORIES ("i", "we", "you", "swear", "positive", ...). The returned dict
also answers to common aliases ("I_WORDS", "pos", "neg", "filler", ...) so
callers can use whichever spelling they prefer.
"""

from __future__ import annotations

import re

# --- Pronouns ---------------------------------------------------------------

I_WORDS = frozenset("""
i me my mine myself i'm i've i'll i'd im ive
""".split())

WE_WORDS = frozenset("""
we us our ours ourselves we're we've we'll we'd lets let's
""".split())

YOU_WORDS = frozenset("""
you your yours yourself yourselves you're you've you'll you'd ya yall y'all
youre u ur
""".split())

# --- Swearing ---------------------------------------------------------------

SWEAR = frozenset("""
fuck fucks fucked fucker fuckers fucking fuckin fking fkn motherfucker
motherfuckers motherfucking shit shits shitty shitting shitted bullshit
horseshit dipshit shithead shitshow ass asses asshole assholes dumbass
jackass badass smartass damn damned dammit goddamn goddamnit goddammit hell
bitch bitches bitching bitchy son-of-a-bitch crap crappy piss pissed pissing
bastard bastards dick dicks dickhead cock cunt cunts twat wanker bollocks
bloody bugger arse arsehole prick pricks douche douchebag screw screwed
frick fricking freaking frickin freakin heck wtf stfu lmfao omfg
""".split())

# --- Affect -----------------------------------------------------------------

POSITIVE = frozenset("""
good great nice awesome amazing love loved loving lovely liked best
better happy glad fun funny cool sweet perfect beautiful excellent fantastic
wonderful brilliant enjoy enjoyed enjoying excited exciting yay
thanks thank appreciate appreciated cute pretty wow haha hahaha
lol lmao laugh laughing laughed win won winning winner gg wp clutch smooth
easy clean solid insane sick dope fire lit epic legend legendary goat proud
pleased delighted cheers welcome congrats congratulations hope hopefully
kind friendly helpful safe comfortable relaxed
calm peaceful favorite favourite yum delicious fresh lucky impressive
incredible hype hyped stoked chill blessed free interesting adorable
""".split())

NEGATIVE = frozenset("""
bad worse worst terrible horrible awful hate hated hating hates sucks suck
sucked stupid dumb idiot idiots annoying annoyed angry mad upset sad
unhappy wrong broken lose lost losing loser fail failed failing failure
problem problems trouble ugly gross disgusting boring bored tired sick
hurt hurts hurting pain painful cringe trash garbage useless pointless
worried worry scared afraid fear nervous stress stressed stressful awkward
weird lame cheap unfair rude mean nasty toxic depressing depressed lonely
miss missed missing sorry regret ugh damn bruh crap shit hell
hard difficult impossible confused confusing frustrated
frustrating disappointed disappointing jealous guilty ashamed embarrassed
miserable pathetic ridiculous nonsense dangerous risky laggy lag lagging
""".split())

ANGER = frozenset("""
angry anger mad furious rage raging pissed annoyed annoying irritated
irritating hate hated hating hates fuming outraged livid frustrated
frustrating frustration yell yelling scream screaming shout
shouting argue arguing argument hostile aggressive violent idiot idiots
stupid moron morons shut shutup stfu toxic jerk bully insult insulted
offended offensive cheating cheater cheaters rigged unfair bullshit
ridiculous disgusting resent resentful bitter grr ugh tilted tilting salty
""".split())

ANXIETY = frozenset("""
worried worry worries worrying anxious anxiety nervous nerves scared scary
afraid fear fears fearful panic panicking panicked stress stressed
stressful tense tension uneasy uncomfortable unsure uncertain doubt
doubtful terrified terrifying frightened freaking freaked overwhelmed
overwhelming paranoid insecure risky risk dread dreading hesitant
awkward embarrassed embarrassing shaky shaking trembling vulnerable
confused confusing lost desperate concern concerned
""".split())

SADNESS = frozenset("""
sad sadly sadness unhappy depressed depressing depression lonely alone
miss missed missing cry crying cried tears tear hurt hurts hurting heartbroken
grief grieve grieving mourn loss lost lose losing sorry regret regrets
miserable misery gloomy hopeless helpless disappointed disappointing
disappointment down low empty tired exhausted broken upset pain painful
suffer suffering sigh rip unfortunately unlucky funeral
gone abandoned rejected ignored worthless
""".split())

# --- Cognitive style ------------------------------------------------------

TENTATIVE = frozenset("""
maybe perhaps guess guessing probably possibly possible kinda sorta idk
think thinking thought unsure uncertain seem seems seemed seemingly
apparently somewhat somehow something someone somewhere sometime sometimes
almost nearly might may could would should suppose supposedly assume
assuming wonder wondering roughly approximately about around ish dunno
hopefully likely unlikely depends doubt doubtful hmm hm imo imho tbh
believe feel feels felt suspect apparently generally usually often
""".split()) | {"kind of", "sort of", "i guess", "not sure"}

CERTAINTY = frozenset("""
always never definitely clearly sure surely obviously absolutely certainly
certain completely totally entirely exactly precisely literally truly
really actually indeed undoubtedly guaranteed guarantee 100 every everyone
everything everybody nothing nobody none all must fact facts true truth
correct right confirmed confident positive proven undeniable forever
constantly completely fully whole perfect perfectly essential clear
""".split()) | {"for sure", "of course", "no doubt"}

# --- Social and achievement --------------------------------------------------

SOCIAL = frozenset("""
friend friends friendly buddy buddies pal pals mate mates guys guy dude dudes
bro bros brother sister sis homie homies fam family mom mum dad parents
everyone everybody anyone anybody someone somebody people person together
team teammate teammates squad group party crew community talk talking talked
talks say said says saying tell telling told ask asked asking asks call
called calling chat chatting meet meeting met hang hanging hangout join
joined joining invite invited share shared sharing help helped helping
listen listening hear heard thanks thank love hug hugs kiss girlfriend
boyfriend wife husband partner neighbor neighbour kids kid son daughter
he she him her they them their his hers boys girls ladies gentlemen
discord server stream streaming welcome hello hi hey sup yo bye goodnight
""".split())

ACHIEVE = frozenset("""
win wins won winning winner goal goals plan plans planned planning done
finish finished finishing complete completed completing achieve achieved
achievement success successful succeed succeeded try tried trying attempt
work worked working effort practice practicing practiced improve improved
improving progress level leveled levelled rank ranked ranking grind
grinding farm farming beat beating beaten earn earned earning gain gained
master mastered best better first champion victory accomplish accomplished
target objective task tasks mission quest build built building create
created make made solve solved fix fixed learn learned learning skill skills
strategy focus lead leading leader carry carried clutch perfect score
""".split())

FILLERS = frozenset("""
um umm uh uhh uhm er erm hmm mm basically literally actually
""".split()) | {"you know", "i mean", "kind of", "sort of"}

# --- Assent / backchannel ----------------------------------------------------
# Kept separate from POSITIVE, as LIWC keeps "assent" apart from "posemo": in
# voice chat these are mostly backchannels ("yeah", "ok", "right") that say
# "I'm listening", not positive feeling. Counting them as positive made
# pos_rate measure how often someone says "yeah".
ASSENT = frozenset("""
yes yeah yea yep yup ya ok okay k kk alright aight right true fine sure
agreed agree correct exactly totally mhm mm-hmm uh-huh
""".split())

# --- Gameplay stoplist ---------------------------------------------------------
# Words that are routine game vocabulary in voice chat ("he killed me", "I'm
# dead", "attack the base") and were removed from NEGATIVE / ANGER / SADNESS
# so emotion rates are not driven by what game is being played. Listed here
# for reference; --social-only is the stronger way to keep game talk out.
GAMEPLAY_STOPLIST = frozenset("""
kill kills killed killing murder destroy destroyed smash fight fighting fought
attack attacked dead die died dying death
""".split())

# Category name -> word set. The order here is the order of `counts` output.
CATEGORIES: dict[str, frozenset] = {
    "i": I_WORDS,
    "we": WE_WORDS,
    "you": YOU_WORDS,
    "swear": SWEAR,
    "positive": POSITIVE,
    "negative": NEGATIVE,
    "anger": ANGER,
    "anxiety": ANXIETY,
    "sadness": SADNESS,
    "tentative": TENTATIVE,
    "certainty": CERTAINTY,
    "social": SOCIAL,
    "achieve": ACHIEVE,
    "fillers": FILLERS,
    "assent": ASSENT,
}

# Extra names accepted when looking up a count (constant names, feature-name
# stems, plurals). Iteration over a counts() result yields only the canonical keys.
_ALIASES: dict[str, str] = {}
for _name, _const in [("i", "I_WORDS"), ("we", "WE_WORDS"), ("you", "YOU_WORDS"),
                      ("swear", "SWEAR"), ("positive", "POSITIVE"), ("negative", "NEGATIVE"),
                      ("anger", "ANGER"), ("anxiety", "ANXIETY"), ("sadness", "SADNESS"),
                      ("tentative", "TENTATIVE"), ("certainty", "CERTAINTY"),
                      ("social", "SOCIAL"), ("achieve", "ACHIEVE"), ("fillers", "FILLERS"),
                      ("assent", "ASSENT")]:
    _ALIASES[_const] = _name
    _ALIASES[_const.lower()] = _name
_ALIASES.update({"pos": "positive", "neg": "negative", "filler": "fillers",
                 "swears": "swear", "sad": "sadness", "anx": "anxiety"})


class CategoryCounts(dict):
    """A plain dict of category -> count that also resolves aliases on lookup."""

    def __missing__(self, key):
        canon = _ALIASES.get(key)
        if canon is None and isinstance(key, str):
            canon = _ALIASES.get(key.lower(), key.lower())
        if canon != key and dict.__contains__(self, canon):
            return dict.__getitem__(self, canon)
        raise KeyError(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key):
        try:
            self[key]
            return True
        except KeyError:
            return False


# Curly quotes from some transcripts become plain apostrophes before splitting.
_APOS = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "`": "'"})
# A word: letters/digits, optionally joined by inner apostrophes or hyphens
# ("i'm", "y'all", "son-of-a-bitch"). Everything else separates words.
_WORD_RE = re.compile(r"[0-9a-z]+(?:['\-][0-9a-z]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase and split into words, keeping apostrophes inside words.

    "I'm  NOT sure -- y'all?" -> ["i'm", "not", "sure", "y'all"].
    Leading/trailing apostrophes (quotes) are dropped.
    """
    text = (text or "").lower().translate(_APOS)
    return _WORD_RE.findall(text)


# Multi-word entries, split once: category -> list of token tuples.
_PHRASES: dict[str, list[tuple[str, ...]]] = {
    name: [tuple(w.split()) for w in words if " " in w]
    for name, words in CATEGORIES.items()
}


def counts(tokens: list[str]) -> dict[str, int]:
    """Per-category counts for a token list (as returned by `tokenize`).

    Single words count once per occurrence. Bare "like" is in no category
    (it is mostly a verb or preposition, and as a filler it cannot be told
    apart by word alone); assent words ("yeah", "ok") count only as assent. Multi-word entries count once per
    occurrence of the whole phrase (so "you know" adds 1 to fillers; its
    "you" is still also counted in the "you" category).
    """
    toks = [t.lower() for t in tokens]
    out = CategoryCounts((name, 0) for name in CATEGORIES)
    for tok in toks:
        # "shit's" / "guy's": fall back to the stem when the full form is unknown.
        stem = tok[:-2] if tok.endswith("'s") else None
        for name, words in CATEGORIES.items():
            if tok in words or (stem and stem in words):
                out[name] += 1
    for name, phrases in _PHRASES.items():
        for ph in phrases:
            n = len(ph)
            for i in range(len(toks) - n + 1):
                if tuple(toks[i:i + n]) == ph:
                    out[name] += 1
    return out
