# ASL-Quest: Complete Project Overview

*A sign language learning platform that recognizes the learner's signs through a webcam, gives instant feedback, and rewards practice.*

**Mini Project III · Department of Computer Science and Engineering · Mepco Schlenk Engineering College (Autonomous)**

| | |
|---|---|
| Team | David Immanuel Gobson (24BCS037), Kaushik N R (24BCS045), Akash P (24BCSL06) |
| Guide | Mrs. T. Bhuvaneswari, AP (Sr.G.) |
| Source of facts | The project's technical guide and complete documentation (written from the repository) |

> **How to read this file.** Anyone who reads it from top to bottom should understand what ASL-Quest is, how every part works, what was measured, and what is still weak. Statements marked **MEASURED** come from a script or test result. **BY DESIGN** is a deliberate rule in the code. **KNOWN LIMIT** is a real weakness that exists today. Nothing here describes a feature that is not built.

---

## Table of contents

1. [What ASL-Quest is](#1-what-asl-quest-is)
2. [The problem and the objectives](#2-the-problem-and-the-objectives)
3. [The three learning modes](#3-the-three-learning-modes)
4. [System architecture](#4-system-architecture)
5. [Technology stack](#5-technology-stack)
6. [Datasets](#6-datasets)
7. [Computer vision: how the hand is seen](#7-computer-vision-how-the-hand-is-seen)
8. [Machine learning models in depth](#8-machine-learning-models-in-depth)
9. [Alphabet recognition end to end](#9-alphabet-recognition-end-to-end)
10. [Word spelling](#10-word-spelling)
11. [Native sign recognition end to end](#11-native-sign-recognition-end-to-end)
12. [Gamification](#12-gamification)
13. [Frontend](#13-frontend)
14. [Backend and API](#14-backend-and-api)
15. [Database](#15-database)
16. [Authentication and security](#16-authentication-and-security)
17. [Privacy and data handling](#17-privacy-and-data-handling)
18. [AURA assistant and SIGNO companion](#18-aura-assistant-and-signo-companion)
19. [Admin, mentor and analytics](#19-admin-mentor-and-analytics)
20. [Results: what was measured](#20-results-what-was-measured)
21. [Known limits and open decisions](#21-known-limits-and-open-decisions)
22. [Testing](#22-testing)
23. [Setup and run](#23-setup-and-run)
24. [Configuration reference](#24-configuration-reference)
25. [Deployment](#25-deployment)
26. [Future scope](#26-future-scope)
27. [Likely review questions and answers](#27-likely-review-questions-and-answers)
28. [Glossary](#28-glossary)
29. [Points to verify before you present](#29-points-to-verify-before-you-present)
30. [References](#30-references)

---

## 1. What ASL-Quest is

ASL-Quest is a gamified web application for learning American Sign Language (ASL) with the learner's own webcam. It works like a language-learning app: the learner picks something to practise, sees a reference, signs in front of the camera, and the app recognizes the sign and gives feedback. Correct practice earns XP, streaks, badges, levels and visible mastery progress.

It is a full-stack system, not a notebook. It contains:

- A **React** single-page web app for the learner, mentor and admin.
- A **FastAPI** backend that serves the API and runs the recognition models.
- A **relational database** (SQLite locally, PostgreSQL in production).
- **Computer vision and machine learning models**: ResNet18 for images, MediaPipe hand landmarks with a landmark classifier, and an I3D video network for whole-word signs.
- An **AI assistant (AURA)** and an in-app companion character (SIGNO).
- **Admin and mentor dashboards**, and a development-only webcam data collection tool.

**What it recognizes today**

| Mode | Input | Recognizer | Vocabulary |
|---|---|---|---|
| Alphabet | one hand pose from the live camera | MediaPipe landmarks in the browser plus a landmark classifier on the server. ResNet18 is the fallback | A–Z static handshapes |
| Word spelling | the same, letter by letter | same as Alphabet | 24 words |
| Native signs | a recorded clip, up to 10 seconds | frozen I3D video backbone plus a trained linear head | 10 whole-word signs |

**What it does not do.** It is not a sentence translator. J and Z are motion letters in real ASL, but the alphabet recognizer looks at one still pose, so it cannot truly recognize their motion. They are practised as held shapes.

---

## 2. The problem and the objectives

### The problem
- ASL is a visual language, yet most people around signers do not know it, so everyday communication is difficult.
- People who want to learn ASL get little interactive practice. Videos and books are passive, and they cannot tell a learner whether their handshape is correct.
- Learners also lack personal progress tracking and motivation in one place.

### Problem statement
To develop an AI-driven ASL learning platform that combines deep-learning sign recognition, guided practice, real-time feedback and gamification for learning and practising American Sign Language.

### Objectives
1. Build a web platform that supports learning, practice and recognition of ASL.
2. Train and evaluate a ResNet18 model for the 26 static alphabet classes.
3. Add MediaPipe hand-landmark recognition to make live webcam recognition more robust.
4. Recognize isolated whole-word signs from video using an I3D model.
5. Keep learners engaged with a guided learning path, real-time feedback, XP, streaks and badges.
6. Support learners and mentors with progress analytics, an assistant (AURA) and admin dashboards.

---

## 3. The three learning modes

### 3.1 Alphabet (Learn, Alphabet and Practice pages)
A learning path of 26 letters. Five beginner letters (C, D, F, H, O) are unlocked first. The learner opens a letter, sees a lesson introduction and a reference image, then practises with the camera. The app confirms a letter only after several consistent, confident readings (a vote window), and then awards XP.

### 3.2 Word spelling (Words page)
The learner spells a word one letter at a time with the same camera recognizer. The catalog has **24 words in five categories**: Beginner (7), Everyday (5), Greetings (4), Family (4) and School (4). Each word has a difficulty, a tip and an estimated XP. This is fingerspelling, not whole-word sign recognition.

### 3.3 Native signs (Native Signs page)
Whole ASL signs instead of letters. The flow has three steps:
1. **Watch** the reference video.
2. **Copy**: a 3-2-1 countdown, then a recording of up to 10 seconds.
3. **Check**: the server classifies the clip with a video model.

A live MediaPipe hand overlay is drawn on the preview while recording, but it is never part of the recorded clip.

### Cross-cutting features
Profile and settings, progress analytics, achievements, daily/alphabet/speed/word challenges, the AURA assistant, the SIGNO companion, and the admin and mentor area.

---

## 4. System architecture

```text
Browser (React 19 single-page app)
  Pages, game state (XP, letters, badges) in React + localStorage + server sync
  Webcam -> MediaPipe HandLandmarker (WebAssembly, runs in the browser)
              |  21 landmarks + raw handedness (no image leaves the browser)
              v
        POST /predict_landmarks (JSON)
  Stable-letter tracker: vote window -> confirmed letter -> correct / incorrect

                         |
                         v
FastAPI backend
  /predict_landmarks : validate -> mirror -> canonicalize -> 87 features -> classifier -> top-k
  /predict (image)   : MediaPipe crop -> ResNet18  (fallback)
  /native/predict    : frozen I3D -> linear head -> top-k
  /progress, /words, /native, /analytics, /admin, /chatbot ...

                         |
                         v
SQLite (local) or PostgreSQL        Model files: *.pth, *.joblib, *.pt
```

### Key design decisions
- **Landmarks are detected in the browser.** A single MediaPipe tracker follows the hand between frames. Only 21 numeric points (about 1 KB of JSON) go to the server, so no image is uploaded and the camera preview never waits on the network.
- **One request in flight.** The recognizer sends only the newest landmarks. Stale frames are dropped, never queued.
- **The server owns correctness for Native Signs and Words.** XP and mastery are computed on the server. The A–Z flow keeps its game state in the browser and pushes it to the server.
- **XP on the server never goes down.** The server keeps the larger of the server XP and the client XP, so an old browser tab cannot erase XP that the server awarded.
- **ResNet18 stays as the fallback.** It is used automatically when no landmark model is loaded.
- **The frontend and backend are separate.** React handles presentation, webcam capture and responsive game state. FastAPI owns persistence and protected decisions. Model inference runs in Python on the backend, except hand detection, which runs in the browser.

---

## 5. Technology stack

### Backend and machine learning (Python 3.11)
| Layer | Technology | Role |
|---|---|---|
| Web framework | FastAPI, Uvicorn | API, automatic OpenAPI docs at `/docs` |
| Validation | Pydantic | request and response models |
| ORM and migrations | SQLAlchemy 2, Alembic | tables, queries, schema changes |
| Databases | SQLite (local), PostgreSQL (production) | persistent data |
| Authentication | bcrypt, python-jose (JWT, HS256) | password hashing, tokens |
| Deep learning | PyTorch, Torchvision | ResNet18 (letters), I3D (native signs) |
| Hand tracking | MediaPipe (Hand Landmarker) | landmarks and hand crops |
| Classical ML | scikit-learn, joblib | landmark classifier |
| Imaging | Pillow, OpenCV, NumPy, Matplotlib | preprocessing and reports |
| HTTP client | httpx | AURA calls to language models |
| Tests | pytest, FastAPI TestClient | backend tests |

### Frontend (TypeScript)
| Layer | Technology |
|---|---|
| UI | React 19, TypeScript 5 |
| Build and dev server | Vite 7 (dev proxy sends `/api` to the backend on port 8000) |
| Styling | Tailwind CSS 4, design tokens as CSS variables |
| Charts | Recharts 3 |
| Browser hand tracking | `@mediapipe/tasks-vision` (HandLandmarker, video mode) |
| Embedded video | `@vimeo/player` (detects reference-video failures) |
| Tests | Vitest, Testing Library, jsdom |

### Assets and external services
| Asset | Role |
|---|---|
| `best_model.pth` (about 43 MB) | ResNet18 alphabet checkpoint |
| `hand_landmarker.task` (about 7.5 MB) | MediaPipe Hand Landmarker model |
| `ASL_citizen_I3D_weights.pt` (about 58 MB) | frozen I3D backbone, MIT licence, Microsoft Research |
| `head_best.pt` (about 43 KB) | trained linear head for the 10 native signs |
| `landmark_L2_s0.joblib` | the landmark classifier the app uses |
| OpenRouter or local Ollama | optional language model for AURA |
| Vimeo | reference sign videos (9 of 10 signs) |
| Vercel and Render | deployment targets |

Large model files are not stored in Git. They must be placed on each machine or host.

---

## 6. Datasets

### 6.1 Alphabet images: Kaggle "ASL Alphabet"
| Item | Value |
|---|---|
| Images | 78,000 |
| Classes | 26 (A–Z) |
| Images per class | 3,000 |
| Image type | 200×200 close-up photos, one signer and setup |
| Split recorded in the repo | 70 / 15 / 15 by numeric filename order, so about 54,600 train, 11,700 validation and 11,700 test (450 per class) |
| Task | 26-class image classification |

The split indices are stored in `outputs/splits.json`. *Check this against your slides*: an earlier version of the slides said 80/20 (62,400 / 15,600). The repository records 70/15/15, and the 94.1% test accuracy is reported on the 11,700-image test set.

### 6.2 Native signs: ASL Citizen (Microsoft Research)
| Item | Value |
|---|---|
| Full dataset | 83,399 videos, 2,731 signs, 52 signers |
| Official split | 40,154 train, 10,304 validation, 32,941 test |
| Project subset ("native-10") | 10 signs: BOOK, EAT1, HELLO, HELP, MOTHER, NO, PLEASE, THANKYOU, WATER, YES |
| Subset split | 149 train, 39 validation, 129 test clips |
| Licence | licensed separately, never stored in or redistributed by the repository |

### 6.3 Real webcam frames (collected by the team)
- Collected with the development-only collection tool: 420 real webcam frames covering only A, B, C, F, U, V, W, X and Y were used to train the landmark classifier.
- A separate held-out pilot of **94 frames, 9 letters, one session** was used for evaluation.
- **No M, N, T or Z webcam samples exist.**

### 6.4 Extracted landmark data
- 9,458 landmark sets extracted from the Kaggle training images (used to train the landmark classifier).
- 4,054 held-out detected frames used in investigations.

---

## 7. Computer vision: how the hand is seen

Computer vision is the step that turns a messy webcam picture into clean input for a model.

### 7.1 Webcam capture
The browser asks for camera permission and opens a live video stream. For the landmark path (the default), MediaPipe runs on that video inside the browser and the picture never leaves the page. For the ResNet18 fallback, the browser grabs a frame, encodes it as a JPEG and sends it to the server.

### 7.2 MediaPipe Hands: 21 landmarks
MediaPipe Hands is Google's real-time hand tracker. It works as two neural networks:
1. A **palm detector** finds where a palm is in the frame. It detects palms because a palm is roughly rigid and easier to box than a hand with fingers in every position.
2. A **landmark model** works on the cropped hand and predicts **21 three-dimensional points**: the wrist (point 0), the base and joints of each finger, and the four fingertips plus the thumb tip. x and y are image positions, and z is relative depth measured from the wrist.

It also returns a hand-presence confidence and a left/right label. In video mode the landmarks from the previous frame predict where the hand will be next, so the heavy detector only runs again when tracking is lost.

**Browser settings.** Detection, presence and tracking confidence are each 0.5. The browser uses the GPU delegate and falls back to CPU. The server-side detector uses a confidence of 0.35.

**Why landmarks help.** Landmarks discard skin colour, background, lighting and clothing, and keep only hand geometry. That is why they handled real webcam frames better than raw images.

**Measured behaviour.** Detection takes about 30–50 ms. MediaPipe missed the hand in 65.4% of the Kaggle close-up photos but found it in 100% of the 94 webcam frames.

### 7.3 Square crop and resize (ResNet18 path)
The hand's bounding box is taken from the landmarks and padded by 30%. It is made square, and any part outside the frame is filled by repeating edge pixels. The crop is then resized to 224×224. A square crop matters: a stretched rectangle gave 69.9% accuracy on the same held-out images, while the padded square gave 92.9%.

### 7.4 Normalization (ResNet18 path)
Each pixel is scaled to 0–1, then each colour channel has its mean subtracted and is divided by its standard deviation, using ImageNet's values: mean (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224, 0.225). ResNet18 was pretrained on ImageNet, so new images must be put on the same scale. Training and live use apply the same transform.

### 7.5 Landmark canonicalization (landmark path)
Before classification the 21 points are normalized so that a letter looks the same anywhere in the frame:
1. If the hand is a left hand, mirror the x values so all hands look like one hand.
2. Subtract the wrist position so the wrist is the origin.
3. Divide by the palm length (wrist to the base of the middle finger) so hand size and distance do not matter.

### 7.6 Video preprocessing (native signs)
OpenCV decodes the clip, and the preprocessing is a port of Microsoft's official loader:
- Frames are in BGR order and pixel values are scaled from 0–255 to the range −1 to 1.
- Up to **64 frames** are sampled. Clips under 96 frames use every frame, clips of 96 or more use every second frame, and clips of 160 or more use every third. Short clips are padded.
- Frames are resized so the short side is at least 226 pixels and the long side at most 256, keeping the aspect ratio.
- The middle **224×224** area is cropped.

---

## 8. Machine learning models in depth

### 8.1 ResNet18 (alphabet image classifier)

**What it is.** A convolutional neural network (He, Zhang, Ren and Sun, CVPR 2016). Very deep networks were hard to train because gradients faded as they moved backward through many layers. ResNet's fix is the **residual block**: instead of learning a mapping H(x), a block learns the difference F(x) = H(x) − x, and its output is F(x) + x. The "+ x" is a **skip connection** that carries the input around the block, so layers only learn a small correction and gradients flow straight back.

**Architecture.** One 7×7 convolution, then four stages of two basic blocks each (every block has two 3×3 convolutions with batch normalization and ReLU). The channels widen from 64 to 128 to 256 to 512 while the image size halves. Global average pooling then feeds one fully connected layer. The 18 counts the weighted layers.

**In this project.**
- The final 1,000-way layer is replaced by `Linear(512, 26)`. That gives about 11.2 million parameters.
- The backbone starts from ImageNet weights (**transfer learning**), so it already knows edges, textures and shapes. The 78,000 hand images only have to teach it what the 26 letters look like.
- A **softmax** turns the 26 scores into probabilities. The largest is the predicted letter. A score below **0.70** is flagged `low_confidence`.
- Input: 224×224 RGB with ImageNet normalization.

**Training recipe.** Batch size 64, 8 epochs, AdamW-style optimizer with learning rate 1e-4 and weight decay 1e-4, early stopping with patience 3, seed 42. Augmentation: resize and random crop, rotation up to 12 degrees, colour jitter, and optionally JPEG artefacts and Gaussian noise. It **does not mirror images**, because mirroring turns a right-hand sign into a left-hand sign.

**Result.** 94.1% on the 11,700-image Kaggle test set (whole frame), but only 52.1% on the 94-frame real webcam pilot. This gap is **domain shift**: Kaggle photos have one signer and consistent close-ups, and a webcam has different rooms, lighting, distance and skin tones. Fine-tuning ResNet18 on the small webcam set made it worse, so ResNet18 became the fallback.

### 8.2 MediaPipe Hand Landmarker
Described in section 7.2. It is a pretrained model from Google. The project does not train it. The `.task` file is not bundled with the repository because its redistribution terms are not stated, so the app fetches it from Google at runtime.

### 8.3 Landmark classifier (the default live recognizer)

This part is **classical machine learning, not deep learning.**

**Features (87 per hand).**
- 60 normalized coordinates (landmarks 1–20, each with x, y, z)
- 10 distances between fingertips
- 5 distances from each fingertip to the wrist
- 4 distances from the thumb tip to the finger bases
- 4 distances from the thumb tip to the finger middle joints
- 8 finger-bend measures (cosines of the angle at each finger's two upper joints)
- 2 thumb-bend measures

These capture what separates letters: which fingers are curled and which tips are close together. The depth value z matters: setting it to zero dropped letter M from 98% to 22% in a sensitivity test.

**Classifier.** A pipeline of `StandardScaler` (rescale each feature to zero mean and unit variance) followed by **multinomial logistic regression** with L2 regularization (`C = 2.0`). Logistic regression computes a weighted sum of the features for each of the 26 classes, applies softmax, and is trained by minimizing cross-entropy. Class order is fixed A to Z, and a model whose stored class list differs is refused at load.

**Model selection.** The code compares logistic regression, a small multilayer perceptron and gradient boosting, each with and without rotation. It scores them on validation macro-F1 over rooms the model has not seen, and it refuses to choose from data from a single room. Training adds augmented copies (small rotations, scaling and noise).

**The deployed model (L2).** Trained on 9,878 rows: 9,458 landmark sets from the Kaggle photos plus 420 real webcam frames (A, B, C, F, U, V, W, X, Y only). Model L1 is a neural-network variant trained without the webcam rows.

**Speed.** Landmark features plus classification take about 1 ms. MediaPipe detection dominates the delay.

**Handedness and mirroring.** MediaPipe labels a hand as if the image were a mirrored selfie. The app sends unmirrored frames, so a table maps the raw label to the real hand, and training and live use apply the same table. Measurements showed the raw label matches the person's own labels (257 of 259), so the *names* in the table are probably swapped, but the behaviour is consistent. The label is used for one thing, mirroring x, and it matters: flipping it dropped letter T from 100% to 2.4% (232 of 382 frames read as Z).

### 8.4 I3D (native sign video classifier)

**What it is.** Inflated 3D ConvNet (Carreira and Zisserman, CVPR 2017). A normal CNN looks at one image, but a sign is a movement. I3D takes a successful 2D image network (Inception) and **inflates** it: every 2D filter of size k×k becomes a 3D filter of size t×k×k, with the pretrained 2D weights repeated along the time axis and scaled. This lets the network inherit knowledge from large image datasets instead of learning motion from scratch. Inception modules run several filter sizes (1×1, 3×3, 5×5) and pooling in parallel and join the results, so each layer sees detail at several scales.

**In this project.**
- The pretrained backbone is Microsoft Research's ASL Citizen checkpoint (MIT licence), and it is **frozen**: its weights are never changed.
- The 64-frame clip becomes a **1,024-number feature vector**, averaged over time.
- A small head is trained on those features: `Dropout(0.5)` then `Linear(1024, 10)`, about 10,000 parameters, with learning rate 1e-3, weight decay 1e-3 and early stopping (best epoch 12 of 27).
- Output: softmax over the 10 classes, giving the prediction, a confidence and a top-k list.

**Why frozen.** There are only 149 training clips. Fine-tuning millions of parameters on that little data would overfit, whereas training a small head on good pretrained features is stable.

**Result.** On the 129 dataset test clips: top-1 **96.9%**, top-5 99.2%, macro-F1 0.969. These are dataset clips, not webcam clips. On live webcam clips the head's output is often nearly flat (for example 20%, 11%, 10%), and webcam accuracy has not been measured.

**Runtime.** I3D is heavier than ResNet18. It loads on the first native request and is cached afterwards. CUDA is used when available, otherwise CPU.

### 8.5 Supporting techniques

| Technique | What it does here |
|---|---|
| Softmax | turns scores into probabilities that sum to one |
| Cross-entropy loss | measures how far the probabilities are from the true label; training lowers it |
| AdamW | optimizer that adapts the step size per parameter and applies weight decay separately |
| Early stopping | halts training when the validation score stops improving, preventing overfitting |
| Data augmentation | crops, small rotations, colour changes and noise so the model does not memorize one setup |
| Batch normalization | keeps layer outputs on a stable scale during training |
| Dropout | randomly switches off units so the model cannot rely on any one |
| Global average pooling | collapses a feature map into one number per channel |
| Transfer learning | reuses weights from a large dataset (ImageNet for ResNet18, ASL Citizen for I3D) |
| L2 regularization | penalizes large weights to reduce overfitting |
| Macro-F1 | averages the F1 score of every class so each class counts equally |
| Top-k accuracy | correct if the true class is among the k highest scores |

### 8.6 Rule-based parts that are not machine learning
- **Stable-letter tracker** (section 9.4): a deterministic vote window that confirms a letter.
- **Recommendations**: fixed rules over recorded per-letter accuracy.
- **AURA scope guard and FAQ**: pattern matching. Only the optional language model is AI.

---

## 9. Alphabet recognition end to end

### 9.1 Which recognizer runs
The app uses the **landmark recognizer** when the backend reports the landmark model as loaded. Otherwise it falls back to the **ResNet18** recognizer. There is no user-facing switch. The landmark model is enabled by setting `ASL_QUEST_EXPERIMENTAL_LANDMARK_MODEL` to a `.joblib` file under `models/`.

### 9.2 Live flow (landmark path)
1. The learner's target letter is chosen (the next letter on the path, or the one they opened).
2. The browser creates one MediaPipe tracker and opens the camera (960×720 requested, unmirrored).
3. A loop runs on each new video frame, draws the 21 points and skeleton on an overlay, and sends **only the newest** landmarks to `POST /predict_landmarks`, at most every 100 ms and with one request in flight. Late replies and replies for a hand that has left the frame are discarded.
4. The server validates the data, mirrors if required, builds the 87 features, classifies and returns the top-k letters with confidence and timings.
5. The stable-letter tracker decides when a letter is confirmed.
6. A confirmed correct letter awards XP and shows feedback. A confirmed wrong letter shows a specific message such as "Almost! You're showing X…" with hints.
7. After a confirmed correct sign the component stops; a new target resets the votes and restarts the loop.

### 9.3 Server validation for `/predict_landmarks`
The request has `landmarks` (21 triples), optional `handedness` (`Left` or `Right`), `image_mirrored` and `top_k` (1–10). Failures are `422`, never a server error: exactly 21 points, finite numbers, `-0.5 ≤ x,y ≤ 1.5`, `-5 ≤ z ≤ 5`, valid handedness, and a hand with non-zero palm length. No model loaded returns `503`. If `image_mirrored` is true, every `x` becomes `1 − x` and the label is swapped.

### 9.4 Stable-letter tracker (confirming a letter)
Raw predictions flicker, so a letter is accepted only after consistent readings.

| Setting | Value | Meaning |
|---|---|---|
| Vote window | 7 | recent confident readings considered |
| Minimum stable votes | 3 | agreeing votes (and a majority) needed |
| No-hand grace | 2 frames | missed-hand frames tolerated before votes are cleared |
| Low-confidence reset | 6 frames | weak frames tolerated before votes are cleared |
| Hand-moved overlap | 0.35 | if the hand box overlaps its previous box less than this, old votes are dropped |
| Request pacing (image path) | 120 ms | minimum gap between image requests |
| Tip after | 12 frames | troubleshooting tips for persistent problems |

Low-confidence frames neither add votes nor erase good ones immediately. A Python copy (`stability.py`) mirrors these rules to measure confirmation delay offline.

### 9.5 Image path (ResNet18 fallback)
`POST /predict` accepts a multipart image. With `require_hand=true` the hand is detected, cropped to a padded square and classified by ResNet18. If no hand is found the API returns an explicit status (`no_hand_detected` or `hand_detector_unavailable`) and **never guesses a letter**. The response includes the prediction, confidence, top predictions, the low-confidence flag, the hand box and optional diagnostics.

---

## 10. Word spelling

- **Catalog.** 24 static words in five categories, stored in the `words` table and seeded by a migration.
- **Flow.** The learner spells a word letter by letter using the same recognizer as Practice. Each stable correct letter advances to the next. At the end the page sends one summary (per-letter results, mistakes, total time) to `POST /words/practice/session`. The server does not infer a word from motion.
- **Server rewards** (recorded atomically in the XP ledger):

| Reward | XP |
|---|---|
| Word completed | +50 |
| Perfect (no mistakes) | +25 |
| Fast (4 seconds or less per letter) | +20 |
| Word challenge (5 words) | +150 |

- **Mastery** of a word = completions ÷ 3. Three words are suggested each day, and recommendations are based on weak letters. There are five word achievements.

---

## 11. Native sign recognition end to end

### 11.1 Learner flow
1. **Watch.** A reference video is shown. If the sign has a Vimeo or YouTube reference it is embedded; otherwise a link card or a "demo video on its way" placeholder is shown. Recording is gated behind watching the reference and pressing "I'm ready".
2. **Copy.** The browser shows a live preview, a 3-2-1 countdown and records up to 10 seconds using `MediaRecorder` (WebM or MP4, depending on browser support) with a "Stop now" button. A hand overlay (up to two hands) is drawn on a separate canvas, so the **recorded clip is the raw camera stream** and the model sees unmodified frames.
3. **Check.** The clip is uploaded to `POST /native/predict`.

### 11.2 Server flow
1. Validate: extension is `.mp4`, `.mov`, `.avi`, `.mkv` or `.webm`; the file is not empty and is at most 100 MB. The `native_sign_id` is resolved from the database **before** inference (404 if unknown or inactive).
2. Write the clip to a private temporary file, run the I3D model, and always delete the temporary file in a `finally` block (503 if the model files are missing, 422 if the video cannot be read).
3. **Correctness.** The attempt is correct only if the top-1 prediction equals the target sign (case-insensitive). *Check this before presenting:* an older version of the guide describes a lenient top-3 rule. The complete documentation says the rule was changed to strict top-1 because the top-3 rule credited wrong attempts about 30% of the time with 10 classes, and it lists that change as uncommitted.
4. For a practice request, the server records a session, updates the learner's progress for that sign (attempts, correct attempts, accuracy, mastery, best confidence, best response time), and awards **+20 XP** only if correct. It then checks the native achievements and always writes a prediction record.
5. The response contains the prediction, confidence, top-k list, model version, latency, correctness, expected sign, response time, XP earned and any new achievements.

### 11.3 The 10 signs and their references
BOOK, EAT1, HELLO, HELP, MOTHER, NO, PLEASE, THANKYOU, WATER, YES. Nine have an unlisted Vimeo reference video by *Learn How to Sign* (learnhowtosign.com). EAT1 has none, because the only candidate video had the wrong title. The reference table stores links only, never video files. Embedding permission was inferred from Vimeo's privacy behaviour, not granted in writing.

### 11.4 Catalog versus model vocabulary
A sign can exist in the database catalog without being recognized by the loaded model. The model only knows the 10 classes in its trained head.

---

## 12. Gamification

### 12.1 Alphabet rules (browser)
| Item | Value |
|---|---|
| Correct sign | +20 XP |
| Fast correct sign (within 5 seconds) | +30 XP ("Quick and clear!") |
| Level | `floor(xp / 1000) + 1` (1,000 XP per level) |
| Daily challenge | 10 correct signs gives +100 XP; a perfect run gives +150 XP more |
| Alphabet challenge | every letter at least once gives +500 XP |
| Speed challenge | 5 correct within 45 seconds gives +150 XP |
| Letter status | "learning" at 3 correct, "mastered" at 10 correct (mastery = correct ÷ 10, capped at 100%) |
| Streak | consecutive practice days |

### 12.2 Learning path
26 nodes with four states: completed, current, available and locked. The five beginner letters (C, D, F, H, O) are available first. They were chosen because the production ResNet scored perfectly on them and they are static, distinct shapes. A was excluded because its precision was only 0.55. After all beginner letters are mastered the next letter becomes current. A completed node shows a tick over a faint watermark of the letter.

### 12.3 Native signs
Correct attempt: +20 XP, recorded in the XP ledger with the reason `native_sign_correct`.

### 12.4 Achievements (16)
`first_sign`, `streak_7`, `alphabet_beginner` (5 letters), `alphabet_explorer` (13), `alphabet_master` (26), `speed_signer`, `fast_signer` (5 fast signs), `daily_perfect`, `word_starter`, `word_learner` (5 words), `word_explorer` (3 categories), `word_master` (10 mastered), `vocabulary_builder` (25 completions), `native_first_sign`, `native_explorer` (3 categories), `native_master` (every native sign mastered).

---

## 13. Frontend

### 13.1 Structure
| Folder | Contents |
|---|---|
| `auth/` | login and register, token handling, local-progress import offer |
| `game/` | constants, XP rules, game context, local storage and server sync |
| `pages/` | Home, Learn, Alphabet, Practice, Words, Word Practice, Challenges, Progress, Achievements, Profile, Settings, Admin |
| `components/` | app shell, `LandmarkPractice`, `CameraPractice`, learning path, reference panel, celebration, dialogs, SIGNO companion |
| `native/` | Native Signs pages, `NativeVideoCapture`, Vimeo readiness check |
| `shared/` | stable-letter tracker, MediaPipe set-up, camera errors |
| `analytics/`, `admin/`, `chatbot/`, `words/`, `collect/`, `debug/` | analytics charts, admin tabs, AURA chat, word catalog, development tools |

### 13.2 Navigation
Screens are selected from application state, not from URLs (there is no router library). The desktop sidebar has the main pages, an "Explore paths" popup (Alphabet, Words, Native Signs with "N / 26 letters mastered") and a "More" group. Mobile uses a bottom bar and a "More" sheet. Admin pages show only for admins.

### 13.3 Browser storage and sync
- `asl-quest-token`: the login token.
- `asl-learning-progress-v1`: the cached game state (XP, streak, per-letter stats, badges).
- State is written locally first, then synchronized to the server with a debounced `PUT /progress` (about 900 ms), so a request is not sent for every small action.
- After login, the server state is loaded. If the server account has no meaningful data and the browser has some, the user is offered a one-time import.
- Demo mode (`VITE_DEMO_MODE=true`) is a frontend-only profile that does not create a server account.

### 13.4 Design
Tailwind 4 with CSS variables. A Duolingo-inspired look with rounded cards, soft shadows and animations that are switched off when the learner prefers reduced motion. Interactive parts carry accessibility labels.

### 13.5 Reliability details worth knowing
- **Vimeo failure detection.** An iframe's `load` event also fires on Vimeo's "video does not exist" page, so readiness is decided by Vimeo's Player SDK, with a 15-second timeout and a fallback link.
- **Word practice loop restart.** Reusing the landmark component for multi-letter words exposed a bug where the loop did not restart after the first letter. It was fixed and tested.

---

## 14. Backend and API

### 14.1 Application
`FastAPI(title="ASL Quest")` with CORS using an exact list of origins (never `*`, because credentials are enabled). At startup it creates the database tables, seeds the 16 achievements, loads the ResNet18 checkpoint and the hand detector, and optionally loads a landmark classifier (only from a `.joblib` under `models/`, because pickle files can run code). A model-loading error is stored so `/health` reports "degraded" instead of crashing.

Layers: **routers** parse HTTP and apply dependencies, **services** hold business rules, and **SQLAlchemy models** map to tables. Each request gets a database session that is closed afterwards.

### 14.2 Endpoint overview
| Area | Endpoints | Access |
|---|---|---|
| Health and recognition | `GET /health`, `GET /history`, `POST /predict`, `POST /predict_landmarks` | public |
| Authentication | `POST /auth/register`, `POST /auth/login`, `GET /auth/me` | register and login public |
| Account | `/users/me/profile`, `/preferences`, `/change-password`, `/export`, `DELETE /users/me` | user |
| Progress | `GET/PUT /progress`, `/progress/letters`, `/progress/migrate-local`, `/progress/migration-offer` | user |
| Practice and challenges | `POST /practice/session`, `/challenges`, `/achievements`, `/recommendations` | user |
| Words | `/words`, `/words/progress`, `/words/recommendations`, `/words/analytics`, `/words/challenge`, `POST /words/practice/session` | user |
| Native signs | `/native/signs`, `/native/progress`, `POST /native/predict` | user |
| Analytics | `/analytics/dashboard`, `overview`, `activity`, `accuracy`, `letters`, `response-time`, `history`, `heatmap`, `streak`, `xp`, `insights`, `funnel`, `weekly-summary` | user |
| Chatbot | `/chatbot/status`, `/chatbot/conversations`, messages, feedback | user |
| Admin | overview, activity, users, mentor dashboard, native signs, achievements, models, predictions, dataset sessions, audit log | admin |
| Development collection | `/collect/*` | only when the feature flag is on |

OpenAPI documentation is available at `/docs` when running locally.

### 14.3 The in-memory history
`GET /history` returns the last 50 recognitions and per-letter counts from a bounded in-memory list (200 entries). It is **not** durable. It resets when the server restarts and is not the learner's attempt record.

---

## 15. Database

SQLite file `data/asl_quest.db` locally, PostgreSQL in production (selected by `DATABASE_URL`). Schema changes are managed by Alembic migrations 0001 to 0004. Some flexible fields are stored as JSON text.

| Table | Purpose |
|---|---|
| `users` | account, email, bcrypt hash, role (`student` or `admin`), active flag |
| `user_progress` | one row per user: XP, level, streaks, totals, game metadata |
| `letter_progress` | per user per letter: attempts, correct attempts, mastery |
| `challenge_progress` | daily, alphabet, speed and word challenge state |
| `achievements`, `user_achievements` | the 16 badges and who earned each |
| `practice_sessions` | each A–Z attempt: target, prediction, correctness, confidence, response time |
| `xp_events` | ledger of every XP award with a reason |
| `words`, `word_progress`, `word_practice_sessions` | word catalog, progress and attempts |
| `native_signs`, `native_sign_references` | sign catalog and reference-video links |
| `native_sign_progress`, `native_sign_sessions` | per-sign progress and per-attempt results |
| `model_versions`, `sign_predictions` | prediction audit records |
| `user_preferences` | interface preferences |
| `chatbot_conversations`, `chatbot_messages`, `chatbot_feedback` | AURA history and feedback |
| `admin_audit_log` | who did which admin action to what |

**Rules.** Unique constraints stop duplicate rows for user+letter, user+word, user+native sign, user+achievement and user+challenge. Deleting a user cascades to the user's own records, while the audit log keeps its rows with the admin link set to null. A table's existence does not prove every handler writes to it.

---

## 16. Authentication and security

1. Registration stores a **bcrypt** password hash, never the password.
2. Login checks the password and returns a **JWT** (HS256) holding the user ID and an expiry. The default lifetime is 10,080 minutes (7 days).
3. The browser sends `Authorization: Bearer <token>` on protected requests.
4. The server decodes the token, reloads the user and rejects missing or inactive accounts.
5. Admin routes also check that the stored role is `admin`. This is enforced on the server, not just by hiding buttons.
6. A JWT is signed, not encrypted. `ASL_QUEST_SECRET_KEY` must be a private random value in deployment. The built-in default is only for local development.
7. Variables starting with `VITE_` are bundled into public frontend code and must never hold secrets.
8. Uploaded native-sign videos use a private temporary file and a size cap, and only listed extensions are accepted.
9. Admin management actions are written to the audit log.

---

## 17. Privacy and data handling

**Safe summary to use.** Camera frames are processed in memory and are not saved by the standard recognition endpoints. In the default live mode, only 21 hand-landmark points leave the browser. Native-sign clips are held in a temporary file that is deleted after inference. Only the result of each attempt is stored. Developer-only collection and diagnostics modes, off by default, can save images locally.

**What is stored per learner.** Each practice attempt (letter, prediction, correctness, response time, confidence), XP events, level, streak, per-letter progress, word and native-sign session results, account details, and AURA conversations.

**What is not stored by default.** Raw frames, video and faces. The landmark points are not stored either.

**Things to know and say honestly.**
- `/predict` and `/predict_landmarks` need no login, so they are not tied to an account.
- `/history` is process-wide and holds labels, not images.
- The development tools (`ASL_QUEST_DEV_COLLECT`, `ASL_QUEST_DEV_DIAGNOSTICS`) write images to disk when switched on. They are off by default and must stay off in production.
- AURA conversations are saved in the database, and by default the text is sent to the OpenRouter service. Use the local Ollama option to keep it on the machine.
- Web hosts can keep request logs, which this project's documents do not cover.

---

## 18. AURA assistant and SIGNO companion

### 18.1 AURA
AURA answers questions about ASL-Quest only.
1. **Scope gate.** Off-topic questions are refused, and greetings are handled.
2. **Offline FAQ.** 38 verified topics matched by weighted patterns.
3. **Knowledge base.** Verified facts about the project's database, architecture, modes, XP rules and limitations.
4. **Language model**, when configured.

Providers (`CHATBOT_PROVIDER`): `openrouter` (default, model `openrouter/free`), `local` (Ollama at `127.0.0.1:11434`, model `qwen3:4b`, fully offline) or `auto` (prefer local, fall back to OpenRouter). Timeout is 30 seconds, the reply limit is 1,024 tokens, and the history sent is limited to 10 messages. If no model is reachable, AURA still answers from the FAQ and knowledge base. For XP and streak questions, the backend adds the learner's real numbers to the prompt. Conversations, messages and thumbs feedback are saved.

AURA's scope check is not a general safety filter, and its strongest source is the project's own knowledge content.

### 18.2 SIGNO
A small animated companion with speech bubbles that react to real events (lesson opened, practice started, correct, incorrect, hints). Local reactions are fixed rules. Free-form questions go through AURA's API, so there is one chat system and no keys in the frontend.

---

## 19. Admin, mentor and analytics

- **Admin page** (admins only): Users (view, edit, deactivate, reset progress, delete), Content (native signs and achievements), Models (versions and predictions), Dataset (collected webcam sessions) and Audit log. Every management action is audited.
- **Mentor dashboard:** per-learner summary: level, XP, accuracy and native-sign mastery.
- **Learner analytics:** accuracy over time, activity heatmap, response time, streak, XP history, a progression funnel, strong and weak letters, and a weekly summary, drawn with Recharts.
- **Recommendations:** the weakest and strongest letters and the next letters to practise, from fixed rules.

---

## 20. Results: what was measured

### 20.1 Alphabet on the Kaggle test split (11,700 images)
| System | End-to-end accuracy | Note |
|---|---|---|
| ResNet18, whole frame, no hand detection | **94.1%** | how the checkpoint was validated |
| ResNet18 with MediaPipe crop | 32.3% | MediaPipe found a hand in only 34.6% of these close-ups; accuracy when it answered: 93.3% |
| Landmark classifiers L1 / L2 | 30.7% / 30.6% | same detection ceiling; when answered 88.7% / 88.3% |

The Kaggle images are 200×200 close-ups of one signer. They are not representative of webcam use and they cap every detector-based system.

### 20.2 Alphabet on real webcam frames (pilot: 94 frames, 9 letters, no J or Z, one session)
| System | End-to-end | Accuracy on accepted predictions (confidence 0.70 or more) |
|---|---|---|
| ResNet18 with crop | 52.1% | 78.8% (33 of 94 accepted) |
| Landmark L1 | 69.1% | 71.9% |
| Landmark L2 | **84.0%** | 94.5% |

C scored 0% on both landmark models and V scored 42% and 8%. This is a **small pilot, not a verdict**: the confidence intervals are wide and frames within a class are near-duplicates.

### 20.3 Native signs (129 ASL Citizen test clips)
Top-1 **96.9%**, top-5 99.2%, macro-F1 0.969. **No webcam measurement exists.**

### 20.4 Confidence is not probability
For the L2 classifier on held-out Kaggle frames, a reported confidence of 0.80–0.90 corresponded to 66% actual accuracy, 0.90–0.95 to 78%, and 0.99 or more to 97.9%. The expected calibration error was 0.056 even on this favourable same-signer data. The 0.70 threshold is a guess, not a measured operating point.

### 20.5 Investigations
- **M and T read as Z.** On held-out Kaggle frames, M recall was 96.6% but its precision was only 47.7% (M is a catch-all for other fist letters). T recall was 61.4%, with 13 of 70 frames read as Z. The live M-to-Z error is *not explained* by handedness, rotation, aspect ratio or depth on held-out data. The decisive missing evidence is real webcam M, T and Z data.
- **Left versus right hand.** On Kaggle frames, raw "Left" accuracy was 48.7% against 94.4% for "Right". It is not a mirroring, tilt, size, confidence or letter-mix effect. The unproven hypothesis is that MediaPipe's landmarks are poor on dark, backlit silhouettes.
- **Detection failures.** MediaPipe failed on 65.4% of Kaggle test images (worst letters: P, X, B, C, T, I, H, N) and on 0% of webcam frames.
- **Timing (dev machine, medians).** MediaPipe 30–50 ms, ResNet18 forward 12–22 ms, landmark features plus classifier about 1 ms.

---

## 21. Known limits and open decisions

**Recognition quality**
- The L2 result of 84% comes from a 94-frame, 9-letter, single-session pilot. The full held-out protocol (a new session and a new person) is not complete.
- C and V are weak. **M, N, T and Z have never been evaluated on real webcam frames.**
- J and Z are motion letters, and a still-pose model cannot recognize them truthfully.
- The landmark model depends on MediaPipe's handedness label, which is unreliable on some frames and strongly affects T, S, I, G, R, L and K.
- Confidence scores are not calibrated.

**Native signs**
- Webcam accuracy is unmeasured, and the model's webcam output is often nearly flat.
- Mastery equals running accuracy, so one correct attempt shows 100% "mastered". A proposed rule is `min(100, correct_attempts ÷ 5 × 100)`, awaiting a decision (5 or 3).
- Earlier attempts graded by the old lenient rule left incorrect historical rows for one learner (XP and mastery overstated). A repair plan awaits approval.
- Vimeo occasionally reports "video does not exist" for a working video (2 of about 16 loads). EAT1 has no reference video.

**Process and engineering**
- The MediaPipe model file's licence is not stated publicly, so it is fetched at runtime rather than bundled.
- The first load of the landmark recognizer depends on jsDelivr and Google being reachable.
- The production bundle is about 1 MB in the main chunk, with no code splitting.
- `/predict` has no upload size limit and there is no rate limiting.
- A hard crash during a native-sign upload could leave a temporary file behind.
- `npm audit` reported 5 vulnerabilities (2 moderate, 3 high) that were not acted on.

**Open decisions:** the mastery target (5 or 3), how to repair the historical XP rows, whether to collect real M, T and Z webcam data next, and whether to remove the unused recognizer-switch code.

---

## 22. Testing

| Suite | Command | Last recorded result |
|---|---|---|
| Backend (pytest, about 30 modules) | `python -m pytest tests -q` | 461 passed |
| Frontend (Vitest) | `npx vitest run` | 27 files, 212 tests passed |
| Types | `npx tsc --noEmit` | clean |
| Production build | `npm run build` | succeeds |

Backend tests cover authentication, progress sync, analytics, admin, words, native signs (strict grading, XP awarded exactly once), landmark validation and mirroring, stability tracking and the chatbot. Frontend tests cover the recognizer components, the Vimeo check, the hand overlay and the native practice page.

**Limit of the testing.** Browser checks used the real dev frontend with the API mocked. There is no committed end-to-end suite, and live camera recognition was exercised through unit tests and manual use.

---

## 23. Setup and run

Windows scripts exist in `scripts/windows/`, and `SETUP_WINDOWS.md` has the details. A typical flow from the repository root:

```powershell
# 1. Backend (Python 3.11 virtual environment)
venv311\Scripts\python.exe -m alembic upgrade head
venv311\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --env-file .env
#    To use the landmark recognizer, set in .env:
#    ASL_QUEST_EXPERIMENTAL_LANDMARK_MODEL=models/experiments/landmark_L2_s0.joblib

# 2. Frontend (second terminal)
cd frontend
npm install
npm run dev        # http://localhost:5173 , API docs at http://127.0.0.1:8000/docs
```

Required files outside Git: `models/best_model.pth`, `models/hand_landmarker.task`, `models/pretrained/asl_citizen_i3d/ASL_citizen_I3D_weights.pt`, and the native head `outputs/asl_citizen_native_10/experiments/i3d_frozen/checkpoints/head_best.pt`. After starting, call `GET /health` to see whether the models loaded. The native model loads lazily, so its errors appear only when a clip is submitted.

**Machine note.** On a 16 GB laptop, run one heavy job at a time (backend, browser tests and training together can run out of memory).

---

## 24. Configuration reference

**Backend**
| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | local SQLite file | database connection |
| `ASL_QUEST_SECRET_KEY` | development placeholder | JWT signing key (set a private value in production) |
| `ASL_QUEST_TOKEN_EXPIRE_MINUTES` | 10080 | login token lifetime |
| `ASL_QUEST_CORS_ORIGINS` | localhost:5173 and the Vercel site | allowed browser origins (setting this replaces the defaults) |
| `ASL_QUEST_BOOTSTRAP_ADMIN_EMAIL` | empty | email promoted to admin |
| `ASL_QUEST_HAND_DETECTION_CONFIDENCE` | 0.35 | server MediaPipe threshold (clamped 0.05 to 0.95) |
| `ASL_QUEST_RECOGNIZER` | `resnet` | default `/predict` recognizer |
| `ASL_QUEST_EXPERIMENTAL_LANDMARK_MODEL` | off | enables `/predict_landmarks` and the landmark recognizer in the UI |
| `ASL_QUEST_DEV_COLLECT`, `ASL_QUEST_DEV_DIAGNOSTICS` | off | development-only file-writing tools |
| `OPENROUTER_API_KEY`, `CHATBOT_PROVIDER`, `CHATBOT_MODEL`, `CHATBOT_LOCAL_BASE_URL`, `CHATBOT_LOCAL_MODEL` | see section 18 | AURA providers |
| `PORT` | unset | bind port for hosting platforms |

**Frontend (compile-time)**
| Variable | Purpose |
|---|---|
| `VITE_API_BASE_URL` | backend address in production (unset in development, which uses the `/api` proxy) |
| `VITE_DEMO_MODE` | frontend-only demo profile |
| `VITE_MEDIAPIPE_WASM_BASE`, `VITE_HAND_LANDMARKER_MODEL_URL` | self-hosted MediaPipe files |

---

## 25. Deployment

The configured plan is a static frontend on **Vercel**, the FastAPI backend and a managed **PostgreSQL** database on **Render**. The frontend needs `VITE_API_BASE_URL` at build time. The backend needs `DATABASE_URL`, a strong secret key, and the frontend's address in the allowed origins. Large model files must be placed on the host because they are not in Git, and the host needs enough memory for PyTorch and the video model.

Before serving a new version run `alembic upgrade head`, then check `/health`, CORS and one real prediction. Keep database backups. Keep data collection and diagnostics off.

This describes the intended deployment. Per the project documentation, nothing was deployed or pushed in the latest working sessions, so confirm what is live before claiming a public URL.

---

## 26. Future scope

- Collect real webcam data (especially M, N, T and Z) from several people and rooms, and complete the held-out evaluation protocol.
- Measure native-sign accuracy on real webcam clips, and expand beyond 10 signs.
- Calibrate confidence scores and pick the acceptance threshold from measured data.
- Recognize J and Z using motion across several frames instead of one still pose.
- Fix the mastery formula, add upload limits and rate limiting, and add an end-to-end test suite.
- Support sentences or continuous signing, and add more learning content and languages.

---

## 27. Likely review questions and answers

**Why did you use both ResNet18 and landmarks?**
ResNet18 reads Kaggle-style close-ups well (94.1%) but dropped to 52.1% on real webcam frames. Landmarks keep only hand geometry and reached 84.0% on the same pilot, so landmarks are the default and ResNet18 is the fallback.

**Why is the I3D backbone frozen?**
There are only 149 training clips. Training a small head on good pretrained features is stable, whereas fine-tuning millions of parameters would overfit.

**Is the 96.9% accuracy reliable?**
It is measured on 129 dataset clips from the same community-recorded dataset. It does not show webcam performance, which has not been measured.

**Why can't it recognize J and Z properly?**
They involve motion, and the alphabet recognizer looks at one still pose.

**Is the confidence score a probability?**
No. It is a softmax score. For the landmark model, 0.80–0.90 meant about 66% actual accuracy.

**What data do you store?**
Results of each attempt, XP and progress, account details and chat history. Frames and clips are processed and discarded, and only landmark points leave the browser in the default mode.

**What is the difference between machine learning and deep learning in your system?**
ResNet18 and I3D are deep neural networks. The landmark classifier is logistic regression, which is classical machine learning. The stability tracker and recommendations are rules.

**What does gamification add?**
XP, levels, streaks, badges and challenges give learners a reason to return and practise, and the server records them so they can be reported.

**What would you improve first?**
Collect real webcam data from several people and rooms, and evaluate on a person and session the model has not seen.

---

## 28. Glossary

| Term | Meaning |
|---|---|
| ASL | American Sign Language |
| Fingerspelling | spelling words with the 26 letter handshapes |
| Native sign | a whole-word ASL sign, as opposed to spelling it |
| Gloss | the written label of a sign, such as `THANKYOU` |
| Landmark | one of the 21 points on a hand (0 is the wrist) |
| Canonicalization | normalizing landmarks (wrist origin, palm length of one, mirrored to one hand) |
| Handedness | MediaPipe's Left/Right label, assigned as if the image were a mirrored selfie |
| CNN | convolutional neural network, a network that learns image filters |
| ResNet18 | an 18-layer CNN with skip connections, used for letters |
| I3D | Inflated 3D ConvNet, a video network, used frozen as a feature extractor |
| Transfer learning | reusing a model trained on a large dataset |
| Linear head | one linear layer trained on frozen features |
| Softmax | converts scores into probabilities that sum to one |
| Top-1, top-k | the single best class, or the k best classes |
| Macro-F1 | F1 score averaged over classes so each counts equally |
| Domain shift | accuracy loss when live data differs from training data |
| Calibration | how well confidence matches real accuracy |
| Held-out | data not used for training or model choice (here also a new person and session) |
| JWT | signed token that identifies a user |
| ORM | maps database tables to Python classes |
| Migration | a recorded schema change (Alembic) |
| XP ledger | the `xp_events` table recording every XP award |
| AURA, SIGNO | the project assistant, and the in-app companion character |

---

## 29. Points to verify before you present

1. **Dataset split.** The repository records 70/15/15 (test set of 11,700). An earlier slide said 80/20. Make the slide match what you actually trained on (check `outputs/splits.json`).
2. **Native grading rule.** The documents disagree (top-3 versus strict top-1). Confirm what the running code does.
3. **Collected webcam images in Git.** An audit reported many image files under `outputs/verify_webcam/` tracked in the repository. If they show a face or room, remove them from the repository and check whether the repository is public. This audit was not independently verified here.
4. **Chat data.** AURA conversations are stored and sent to OpenRouter by default. Do not claim that no data leaves the server.
5. **Deployment.** Confirm whether the Vercel and Render deployment is live before showing a public URL.
6. **Webcam evidence.** The 84.0% figure is a 94-frame pilot. Say so when you quote it.

---

## 30. References

1. K. He, X. Zhang, S. Ren and J. Sun, "Deep Residual Learning for Image Recognition," *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 770–778.
2. F. Zhang et al., "MediaPipe Hands: On-device Real-time Hand Tracking," 2020.
3. J. Carreira and A. Zisserman, "Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset," *CVPR*, 2017. (I3D; verify details before citing.)
4. A. Desai et al., "ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition," *NeurIPS Datasets and Benchmarks*, 2023. (Verify details before citing.)
5. Kaggle, "ASL Alphabet" dataset (cite the dataset page and author).
6. F. M. Najib, "Sign language interpretation using machine learning and artificial intelligence," *Neural Computing and Applications*, vol. 37, pp. 841–857, 2025.
7. Barrister R., Ambeth Kumar V. D. and Ashok Kumar V. D., "Mizo hand sign language detection using a multi-scale transformer-based hybrid feature extractor and fusion network," 2026. (Check author names against the paper.)

*End of document. If behaviour changes, update the affected section and keep the labels honest.*
