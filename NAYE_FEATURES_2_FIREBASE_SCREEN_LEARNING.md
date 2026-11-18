# Update 2 — Browser close fix · Firebase check · Screen skills · Live learning

(Yeh file aapki upload ki hui spec wali file ki jagah leti hai. Spec ka bahut kuch yahan bana hai, par
kuch cheezein alag hain; neeche "Spec se farq" me likha hai.)

---

## 1. Inbuilt browser band kyun nahi hota tha
Do asli bugs the (code padhke confirm kiye):

1. **✕ button / "browser band karo" sirf panel chhupata tha** (`set_visible(False)`). Andar Chromium chalta
   rehta tha, isliye video/gaana bajta rehta aur lagta tha band nahi hua.
2. **Aakhri tab band karne pe Google dobara khul jata tha** (`close_tab` me `goto("google.com")`).

**Fix** (`core/inbuilt_browser.py`, `ui.py`, `plugins/inbuilt_browser.py`):
- Naya `park()`: media pause, extra tabs band, page blank (`about:blank` — isse video ka document hi khatam
  hota hai), panel HUD pe wapas. Agar page jaane se mana kare to naya page bana ke purana band.
- Aakhri tab band karne pe ab browser poora band hota hai (HUD pe wapas).
- ✕ aur voice "browser band karo" (`inbuilt_browser action=close`) → `park()`.
- **Header toggle aur doosre panel pe jana** ab bhi sirf hide karte hain (gaana peeche chalta rahe) — yeh jaan-boojhke.
- Chromium process zinda rehta hai (dobara kholna turant, logins bache rehte hain). Poora shutdown chahiye to
  app band karo.

## 2. Firebase connect
Aapki zip me `config/firebase_service_account.json` **maujood hai** (project `ailive-1a4b1`) aur sync code
(`memory/firebase_sync.py`) pehle se wired hai. Kami thi ek "yeh sach me chal raha hai?" check ki:

```
python enable_firebase.py                    # config/ wali key check
python enable_firebase.py --key "C:\path\key.json"
python enable_firebase.py --url https://NAME-default-rtdb.asia-southeast1.firebasedatabase.app
```
Yeh key validate karta hai, token leta hai, Realtime DB me write→read→delete test karta hai, galat region ho to
sahi URL khud dhundke `firebase_rtdb_url` me save karta hai, Firestore check karta hai (optional), aur aakhir me
**app wale hi code path** se read-back karta hai. Token/private key kabhi print nahi hota.
Agar "no database" aaye: Firebase console → Build → Realtime Database → **Create database** (ek baar).

**Yaad rakhna (aapka sawaal: "har bar new command/memory sikhaye to yaad rakhe"):** yeh pehle se bana tha:
`train_command` ("jab main X bolu to Y karo", important memory) local DB + Firebase me jata hai, startup pe
`pull_and_merge` wapas laata hai. Ab **screen skills bhi isi se jude hain** (neeche).

## 3. Screen skills — "ek baar dikhao, baar-baar karo"
**Sikhana:** "seekho: Chrome band karna" → aap mouse se clicks karo → "bas / ho gaya".
**Dohrana:** "Chrome band karo" (naam ke aas-paas ke shabd bhi chalte hain). Skills ka naam prompt me automatically
jata hai, to JARVIS dobara samjhane nahi bolta.

Har click ke liye record hota hai: button, kitna ruke, kaunsi app/window, window ke andar kahan, aur **do chhoti
photos** (48px aur 120px) cursor ke aas-paas ki — **sirf us window ke andar se** (window ke peeche ka wallpaper
agli baar alag hota hai, isliye).

**Replay kaise dhundta hai (is order me), andhe click kabhi nahi:**
1. window dhundo (process + title) aur aage lao,
2. us window me saved photo dhundo (OpenCV; na ho to numpy fallback),
3. Gemini se label ke naam se dhundhwao (sirf `cloud_assist` on ho),
4. window ke andar wahi relative jagah — **sirf agar window ka size 12% ke andar wahi hai**.
Kuch na mile to **ruk jata hai** aur batata hai kaunsa step fail hua.
Hamesha refuse karta hai: JARVIS ki apni window pe click, screen ke bahar click.
Abort: Esc / "stop" (JARVIS ka interrupt ab chalte skill ko bhi rokta hai), ya mouse ko screen ke top-left corner pe le jao (pyautogui failsafe).
Extra: `add_step` se hotkey / type / wait jodo, `list`, `forget`, `status`. 60 sec koi click na ho to sikhna khud band (clicks ho chuke hon to save ho jata hai).

## 4. Live learning — "screen dekh dekh ke seekhta jaye" (opt-in)
Aapne manga: bina har baar "seekho" bole, kaam karte-karte seekhna. Maine yeh **chalu-bandh (opt-in)** banaya:

- "**live seekhna chalu karo**" → JARVIS aapke **pichhle 60 clicks** ki chhoti memory rakhta hai (sirf RAM me, 2 ghante tak).
- Kaam normal karo. Phir bolo "**jo abhi kiya wo yaad rakh lo: Chrome band karna**" → aakhri sequence skill ban jata hai.
  (Screen pe hint bhi aata hai: "👁 3 clicks dekhe — 'ye yaad rakh lo: <naam>' bolo".)
- "**live seekhna band karo**" → band + memory **mita di jati hai**.
- Disk pe **kuch nahi** likha jata jab tak aap "yaad rakh lo" na bolo (test se verify). Restart ke baad yeh hamesha off hota hai.

**Maine jaan-boojhke yeh nahi banaya:** "hamesha chupchap poori screen dekhta rahe aur khud seekh le". Wajah: privacy
(aapki screen par password/chat dikhte hain), aur galat seekha hua click kuch bhi band/delete kar sakta hai.
Live learning me bhi sirf **clicks + chhoti photos** hain — poori screen ya keystrokes nahi.

## 5. Privacy — kya kahan jata hai
- Photos (crops): **sirf is PC pe** (`config/ui_skills/`). Firebase pe sirf text (naam, window, positions, labels).
- Label banane ke liye 120px photo **Gemini ko jati hai** (jab aap sikha/save kar rahe ho). Replay me, agar photo match na ho,
  us window ka screenshot Gemini ko jata hai. Band karna: `config/api_keys.json` → `"screen_skills": {"cloud_assist": false}`.
- Doosre PC se aaye skills bina photos ke aate hain (label/position se chalte hain, ya wahan dobara sikhao).

## 6. Spec se farq (honest)
| Spec me | Yahan |
|---|---|
| Firebase key "zip me nahi" | Aapki zip me hai. Meri **full zip me jaan-boojhke nahi daali** (secret) — apni purani `config/` rakhna. |
| Fallback "window me wahi relative jagah" | Sirf tab jab window size ≤12% badla ho (warna galat jagah click ka risk). |
| Photo crop poori screen se | Sirf clicked window ke andar se. |
| Live seekhna: "nahi banaya" | Opt-in rolling-buffer version banaya (upar). |
| Ctrl-C jaisa abort | Esc/stop → `JarvisLive.interrupt()` → running skill abort. |

## 7. Limits (sach)
- Window ke naam/jagah **Windows pe** pehchane jate hain. Mac/Linux pe sirf photo + screen position (aur `pynput` zaroori).
- Windows pe agar koi app **Administrator** se chal rahi hai, to JARVIS bhi Administrator se chalna chahiye, warna uske clicks dikhte/lagte nahi. UAC prompt pe click nahi ho sakta.
- Multi-monitor/negative coordinates tested (fake screen pe). **DPI scaling** wali asli screens pe exact point mere paas test nahi hua; JARVIS (Qt6) per-monitor-DPI aware hai, jo mss/pyautogui ke saath match karta hai.
- Full-screen games/DRM video me screen capture kaali aa sakti hai → photo match nahi hoga → ruk jayega.
- Template matching exact pixels pe tikta hai: theme/zoom/icon badle to ruk ke label (Gemini) pe jata hai.

## 8. Naye / badle files
**Naye:** `core/screen_skills.py`, `actions/screen_teach.py`, `enable_firebase.py`, `tests/test_screen_skills.py`,
`tests/test_screen_teach_action.py`, `tests/test_browser_close.py`, `tests/test_enable_firebase.py`, `tests/test_main_screen_skills.py`
**Badle:** `core/inbuilt_browser.py`, `ui.py`, `plugins/inbuilt_browser.py`, `main.py` (prompt + interrupt), `requirements.txt` (pynput, google-auth), `.gitignore`, `config/api_keys.example.json`
(Update 1 ke saare files bhi isi package me hain.)

Install: `pip install pynput google-auth` (baaki — opencv, mss, pyautogui, psutil — pehle se requirements me hain).

## 9. Test
`python -m unittest tests.test_screen_skills tests.test_screen_teach_action tests.test_browser_close tests.test_enable_firebase tests.test_main_screen_skills`
Poora naya suite (Update 1 + 2): **136 tests pass**. Purane `_voice_fingerprint_tests` ka result original aur naye project pe barabar.

**Kya asli me test hua:** fake desktop (canvas + windows + mouse) pe record→replay, window hilne pe, window gayab, button badla (ruk jata hai),
JARVIS-window safety, multi-monitor, live buffer (disk pe kuch nahi), asli `InbuiltBrowserEngine` thread fake Playwright ke saath,
asli `JarvisLive.interrupt()` → running skill abort, Firebase checker fake HTTP ke saath.

**Test NAHI hua (mere paas nahi tha):** asli Windows mouse hook / pynput, asli Chromium close, asli Firebase (network band tha),
asli Gemini label/locate, DPI-scaled asli screen, PyQt6 me UI ka behaviour (sirf code padhke + AST check).
Pehli baar chalao to `SYS:` / `[ScreenSkills]` log lines bhejna.
