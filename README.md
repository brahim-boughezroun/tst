# Hand Rift Web

A browser version of the OpenCV + MediaPipe hand-rift project. It uses the user's webcam, tracks two hands with MediaPipe Hand Landmarker, detects a thumb/index pinch on both hands, charges the effect, and opens a neon rift when the user pulls their hands apart.

## Run locally

Requirements: a recent Node.js installation and a browser with webcam support.

```bash
npm install
npm run dev
```

Open the local URL printed by Vite, click **Start camera**, and allow camera permission.

## Build

```bash
npm run build
```

The production site will be generated in `dist/`.

## Deploy to Vercel

1. Push this folder to a GitHub repository.
2. Import the repository in Vercel.
3. Vercel should detect Vite automatically.
4. Build command: `npm run build`
5. Output directory: `dist`
6. Deploy.

Camera access requires a secure context on normal public websites. Vercel gives the deployed site HTTPS automatically.

## Gesture

1. Show both hands.
2. Pinch thumb + index finger on both hands.
3. Keep pinching until the charge reaches 100%.
4. Pull your hands apart to open the rift.
5. Press **Reset rift** or the `R` key to start over.

## Privacy

The webcam frames are processed in the browser. This app does not include a backend or upload camera frames to an application server.

## Main files

- `src/main.ts` — MediaPipe setup, webcam loop, hand smoothing, pinch detection, state machine, and rift rendering.
- `src/style.css` — interface and responsive design.
- `index.html` — website layout.
