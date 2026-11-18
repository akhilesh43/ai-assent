# Mark LVI — Local Ear, Song-Silent Mode, Image Studio, Hologram FX

Yeh file batati hai kya badla, kaise chalana hai, aur kya test hua / kya nahi.

---

## 1. LOCAL EAR — "PC pehle sune, phir Gemini ke paas jaye"

### Pehle kya problem thi
Purane modes me mic ka audio seedha Google ko jata tha aur "kab bolna band kiya" ka faisla bhi Google ka server leta tha. PC kuch check nahi kar sakta tha: yeh gaana hai, khansi hai, ya bahut dheemi awaaz.

**Ek asli bug bhi mila:** purana `AudioFrontend` har 64 ms ke block pe, jiski awaaz RMS 0.006 se kam thi, usko *noise samajhke 35% tak dabata tha*, aur boost sirf 2.4x tha. Matlab **dheere bolne wali awaaz kharab/unclear ho jati thi**. Naya frontend loudness ko chhedta hi nahi; loudness poore sentence pe ek baar set hoti hai.

### Ab kya hota hai

```
mic → ClearVoiceFrontend → LocalEarGate → Gemini
       (hum/hiss kam)       1. sentence ka end PC khud dhundta hai
                            2. poore sentence ki loudness barabar:
                               dheemi awaaz upar (max +30 dB), tez/chillane wali neeche
                            3. (accurate mode) local Whisper se text nikalta hai,
                               noise / galat-suna hua / gaane ke lyrics hata deta hai
                            4. Gemini ko ek saaf sentence, explicit start/end ke saath
```
Gemini ka apna VAD band rehta hai (code me pehle se `automatic_activity_detection.disabled`), isliye turn ki seema ab **PC** tay karta hai.

### Do speed
| Mode | Kya karta hai | Extra delay |
|---|---|---|
| `fast` | End-of-speech audio se; Whisper sirf gaane/sleep me | ~0.55 s |
| `accurate` | Har sentence local check hota hai | ~0.8 s + Whisper (GPU ~0.3 s, CPU 1–2 s) |

### Chalane ka tarika
```
pip install faster-whisper scipy
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12     # sirf NVIDIA GPU ke liye
python enable_local_ear.py                           # ya: --speed fast --model small --device cuda
```
Phir JARVIS restart. App me 🎧 LISTEN button se mode badal sakte ho (LOCAL EAR sabse pehle aata hai), ya Controls me dropdown se.

**Aapke PC (8 GB RAM, 3.9 GB GPU) ke liye:** `model: small`, `device: cuda` (ya `auto`). `medium` mat lo.

### Settings (`config/api_keys.json` → `"local_ear"`)
| Key | Default | Matlab |
|---|---|---|
| `speed` | `accurate` | `fast` / `accurate` |
| `model` | `small` | tiny/base/small/medium |
| `device` | `auto` | auto / cuda / cpu (GPU fail ho to khud CPU pe aa jata hai) |
| `language` | `""` | `"hi"`/`"en"` pin karne ke liye; khali = har baar detect |
| `end_silence_ms` | 0 (auto) | kitna chup rehne pe sentence khatam; cut jaye to badhao (jaise 1000) |
| `margin` | 2.2 | kam = dheemi awaaz bhi pakdega (par noise bhi) |
| `media_margin` | 2.6 | gaane ke time awaaz kitni upar honi chahiye |
| `denoise` | 0.5 | 0 band … 1 sabse tez. Zyada denoise Whisper ko ulta bigaad deta hai |
| `target_rms` | 0.10 | Gemini ko bheji awaaz ka level |
| `media_wake` | true | gaane/sleep me "Hello" mode |
| `wake_words` / `name_words` | hello, hallo, हेलो… / jarvis… | apne shabd jodo |

---

## 2. Gaana chal raha ho to chup, "Hello" bolne pe active

- Jab HUD me gaana/video **awaaz ke saath** chal raha ho (ya "sleep" bola ho), mic ka audio **Google ko jata hi nahi**.
- Local Whisper sunta hai. Sirf wo sentence jo **shuru me** "Hello" / "Hello Jarvis" / "Hey Jarvis" / "Jarvis" se start ho, assistant ko jagata hai.
- Jagne pe: gaana pause → assistant LISTENING. Agar sirf "Hello" bola → bas jagta hai. Agar "Hello, agla gaana chalao" bola → "Hello" ke baad ka hissa hi Gemini ko jata hai.
- Lyrics me beech me "hello" aaye to **nahi** jagta (wake word sirf sentence ke shuru me).

**Sach:** gaane ke upar "Hello" pehchanna best-effort hai. Headphones lagao, ya gaane ki volume apni awaaz se kam rakho. `faster-whisper` ke bina gaane ke time assistant chup hi rehta hai (wake word detector ya push-to-talk use karo).

---

## 3. IMAGE STUDIO — JARVIS khud image banata hai

- **Voice:** "ek sher ki photo banao", "draw a neon city" → plugin `image_studio`.
- HUD par bada preview (click ya ✕ se band), `generated_images/` me save.
- **Phone/web dashboard:** footer me 🖼 button → IMAGE STUDIO panel: prompt, style, aspect, GENERATE, gallery. Voice se bani image bhi wahan turant dikhti hai.
- **Backends:** (1) Gemini image model (same key; model names ki list `image_gen.gemini_models` me, ek na chale to agla try hota hai), (2) **Pollinations** (free, bina key).

**Limits (sach):**
- Sabhi free Gemini keys pe image generation allowed nahi hoti; tab quota error aata hai aur Pollinations chalta hai.
- Pollinations anonymous: lagbhag 1 image / 15 sec, kabhi watermark. Code khud wait karta hai.
- Dashboard ke image routes (`/api/image/generate`, `/api/images`, `/generated/…`) bina encryption ke chalte hain (jaise file upload), pairing token se protected.

---

## 4. HOLOGRAM FX
`core/holo_fx.py`: projector base + light cone, upar uthte particles, scanlines + scan band, kabhi-kabhi glitch, aur state ring:
LISTENING = awaaz ke saath badhti ripple · THINKING = ghoomte arcs · SLEEPING = head dim + saans leti ring, status me "SLEEPING · SAY “HELLO”".
Band karna ho: `"holo_fx": false`. Koi bhi error aaye to HUD painting nahi rukti.

---

## 5. Files

**Naye:** `core/local_ear.py`, `core/image_gen.py`, `core/holo_fx.py`, `plugins/image_studio.py`, `enable_local_ear.py`, `config/api_keys.example.json`, `tests/…`
**Badle:** `main.py`, `ui.py`, `memory/config_manager.py`, `dashboard/server.py`, `dashboard/static/app.html`, `plugins/voice_profiles.py`, `requirements.txt`, `.gitignore`

Nested copies: aapki zip me `Mark-LVI/` ke andar purani copies hain. **Sirf top-level `Mark-LVI-main/` badla gaya hai.** Wahi chalao.

---

## 6. Kya test hua, kya nahi

Chalaye gaye tests (`python -m unittest tests.test_local_ear tests.test_image_gen tests.test_holo_fx tests.test_main_local_ear`):
- Local Ear DSP + gate (synthetic awaaz, fake Whisper): turn detection, quiet→loud, loud→control, noise/click ignore, PTT flush, stale drop, gaana me lyrics ignore, Hello wake, Hello+command trim.
- **Asli `JarvisLive` methods** ke saath fake Gemini session: gate/frontend mode ke hisaab se, Gemini VAD off, ek sentence = ek start…end turn, noise kabhi nahi bheja gaya, gaane me lyrics nahi bheje, Hello gaana pause karta hai.
- Image backend fallback / save / listener (mock backends).
- Hologram FX fake-Qt recorder pe (typo/arithmetic; **dikhne me kaisa lagta hai wo nahi**).
- Purane `_voice_fingerprint_tests`: original aur naya, dono pe same 4 test stub-artifacts ki wajah se fail — koi regression nahi.

**Mere paas nahi tha, isliye test NAHI hua:** asli mic/speakers, asli Whisper model, asli Gemini Live, PyQt6 me hologram/overlay ka look, FastAPI routes ka live run, asli Pollinations/Gemini image call. Pehli baar chalane pe ye sab aapke PC pe verify honge — kuch gadbad dikhe to log ki `EAR:` / `IMG:` lines bhejna.

## 7. Tuning cheat-sheet
| Dikkat | Kya karo |
|---|---|
| Sentence beech me kat jata hai | `end_silence_ms` 1000–1200 |
| Dheemi awaaz pakadta hi nahi | `margin` 1.8, `abs_min_rms` 0.0015 |
| Noise/pankhe pe trigger | `margin` 3.0 |
| Reply dheere aata hai | `speed: fast`, ya GPU pe `small` |
| Hinglish galat sunta hai | `language: "hi"` try karo, ya `model: medium` (RAM dekho) |
| Gaane pe "Hello" nahi pakadta | headphones; `media_margin` 2.0 |
| Whisper GPU error | `device: cpu` |
