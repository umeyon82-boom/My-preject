import numpy as np, wave, sys
SR = 48000; DUR = 6.4
T0 = [0.3, 2.0, 3.7]; CLICK = [1.15, 2.85, 4.55]; END = 5.7   # same timings as index.html
rng = np.random.default_rng(7)
N = int(SR * DUR)
L = np.zeros(N); R = np.zeros(N)

def tt(d): return np.arange(int(SR * d)) / SR
def env(x, a=0.003, d=0.1):
    t = tt(len(x) / SR); return np.minimum(1, t / a) * np.exp(-t / d)
def lp(x, fc):
    a = 1 - np.exp(-2 * np.pi * fc / SR); y = np.empty_like(x); s = 0
    for i, v in enumerate(x): s += a * (v - s); y[i] = s
    return y
def hp(x, fc): return x - lp(x, fc)
def sweep(f0, f1, d):
    t = tt(d); f = f0 * (f1 / f0) ** (t / d); return np.sin(2 * np.pi * np.cumsum(f) / SR)
def place(x, t, pan=0.0, g=1.0):
    i = int(t * SR); n = min(len(x), N - i)
    if n <= 0: return
    L[i:i+n] += x[:n] * g * np.sqrt((1 - pan) / 2) * 1.414
    R[i:i+n] += x[:n] * g * np.sqrt((1 + pan) / 2) * 1.414

def pop_in():
    x = sweep(280, 900, 0.16) * env(np.zeros(int(SR*.16)), .004, .05)
    return x * 0.8 + hp(lp(rng.standard_normal(int(SR*.16)), 5000), 800) * env(np.zeros(int(SR*.16)), .01, .05) * 0.25
def whoosh(d, up=True):
    t = tt(d); n = rng.standard_normal(len(t)); y = np.zeros_like(n)
    out = np.zeros_like(n)
    seg = 8; step = len(t) // seg
    for k in range(seg):
        fc = (500 + 3000 * (k / seg)) if up else (3500 - 3000 * (k / seg))
        out[k*step:(k+1)*step] = lp(n[k*step:(k+1)*step], fc)
    w = np.sin(np.pi * t / d) ** 2
    return hp(out, 200) * w * 0.6
def click():
    t = tt(0.05)
    return (rng.standard_normal(len(t)) * np.exp(-t / 0.004) * 0.5 + np.sin(2*np.pi*1900*t) * np.exp(-t / 0.012)) * 0.8
def bell():
    d = 1.4; t = tt(d); x = np.zeros_like(t)
    for f, a, dec in [(1318, 1, .5), (1760, .6, .35), (2637, .4, .25), (3951, .2, .15), (3320, .25, .2)]:
        x += a * np.sin(2*np.pi*f*t) * np.exp(-t / dec)
    return x * np.minimum(1, t / 0.002) * 0.5
def sparkle(notes, gap=0.045):
    out = np.zeros(int(SR * (gap * len(notes) + 0.5)))
    for i, f in enumerate(notes):
        t = tt(0.4); s = (np.sin(2*np.pi*f*t) + .3*np.sin(2*np.pi*2*f*t)) * np.exp(-t / 0.1) * np.minimum(1, t / .002)
        j = int(i * gap * SR); out[j:j+len(s)] += s * 0.45
    return out
def thumb_pop():
    t = tt(0.12); return np.sin(2*np.pi*(500 + 700 * t / 0.12) * t) * np.exp(-t / 0.04) * 0.9
def zap():
    d = 0.3; x = sweep(180, 2600, d); x = np.tanh(3 * x)
    return x * env(x, .002, .12) * 0.5
def boom():
    d = 0.7; t = tt(d); f = 110 * np.exp(-t * 3.0) + 38
    return np.sin(2*np.pi*np.cumsum(f)/SR) * np.exp(-t / 0.22) * 0.9
def confetti(d=0.9, n=26):
    out = np.zeros(int(SR * d))
    for _ in range(n):
        st = rng.random() ** 1.6 * (d - 0.1); t = tt(0.06); f = rng.uniform(2500, 6500)
        s = (np.sin(2*np.pi*f*t) * 0.5 + rng.standard_normal(len(t)) * 0.3) * np.exp(-t / 0.012)
        j = int(st * SR); out[j:j+len(s)] += s * rng.uniform(0.15, 0.4)
    return out

pans = [-0.45, 0.0, 0.45]
for i in range(3):
    place(pop_in(), T0[i], pans[i], 0.8)
    place(whoosh(0.55), CLICK[i] - 0.6, pans[i], 0.35)
    place(click(), CLICK[i] - 0.01, pans[i], 0.9)
place(bell(), CLICK[0] + 0.02, pans[0], 0.9)
place(bell() * 0.6, CLICK[0] + 0.30, pans[0], 0.9)           # second ding as the bell swings
place(thumb_pop(), CLICK[1], pans[1], 0.9)
place(sparkle([880, 1175, 1568, 2093]), CLICK[1] + 0.05, pans[1], 0.8)
place(zap(), CLICK[2], pans[2], 0.8)
place(boom(), CLICK[2] + 0.02, pans[2], 1.0)
place(confetti(), CLICK[2] + 0.08, pans[2], 0.7)
place(sparkle([1046, 1568, 2093, 3136], 0.05), CLICK[2] + 0.2, pans[2], 0.6)
place(whoosh(0.5, up=False), END - 0.05, 0.0, 0.7)

peak = max(np.abs(L).max(), np.abs(R).max())
g = 0.7 / peak
st = np.stack([L, R], 1) * g
st = np.tanh(st * 1.2) / np.tanh(1.2)  # soft limit
pcm = (np.clip(st, -1, 1) * 32767).astype('<i2')
with wave.open(sys.argv[1], 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
print('ok', peak)
