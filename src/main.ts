import './style.css';
import {
  FilesetResolver,
  HandLandmarker,
  type HandLandmarkerResult,
  type NormalizedLandmark,
} from '@mediapipe/tasks-vision';

const CFG = {
  smoothing: 0.30,
  pinchStartThreshold: 0.38,
  pinchReleaseThreshold: 0.52,
  activationHoldFrames: 18,
  pullStartPixels: 18,
  fullOpenPixels: 300,
  lineWidth: 4,
  nodeRadius: 6,
  glowIntensity: 0.85,
  pulseSpeed: 0.08,
  hueSpeed: 0.8,
};

const CONNECTIONS: Array<[number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];

const FINGERTIPS = new Set([4, 8, 12, 16, 20]);
const HAND_LABELS = ['Left', 'Right'] as const;
type HandLabel = typeof HAND_LABELS[number];
type Point3 = { x: number; y: number; z: number };
type PixelPoint = { x: number; y: number };
type RiftState = 'IDLE' | 'CHARGING' | 'RIFT_READY' | 'OPENING' | 'PORTAL_ACTIVE';

const STAR_FIELD = Array.from({ length: 150 }, () => ({
  x: Math.random(),
  y: Math.random(),
  size: 1 + Math.floor(Math.random() * 3),
  hue: Math.random() * 360,
}));

const video = document.querySelector<HTMLVideoElement>('#webcam')!;
const canvas = document.querySelector<HTMLCanvasElement>('#canvas')!;
const ctx = canvas.getContext('2d', { alpha: false })!;
const startOverlay = document.querySelector<HTMLDivElement>('#startOverlay')!;
const startButton = document.querySelector<HTMLButtonElement>('#startButton')!;
const resetButton = document.querySelector<HTMLButtonElement>('#resetButton')!;
const statusChip = document.querySelector<HTMLDivElement>('#statusChip')!;
const errorText = document.querySelector<HTMLParagraphElement>('#errorText')!;

let handLandmarker: HandLandmarker | null = null;
let mediaStream: MediaStream | null = null;
let animationId = 0;
let lastVideoTime = -1;
let frameCount = 0;
let hue = 195;
let riftState: RiftState = 'IDLE';
let activationProgress = 0;
let riftOpenProgress = 0;
let riftStartDistance: number | null = null;
let savedLeftPoint: PixelPoint | null = null;
let savedRightPoint: PixelPoint | null = null;
let readyFlashFrames = 0;
let smoothedHands: Partial<Record<HandLabel, Point3[]>> = {};
let pinchStates: Record<HandLabel, boolean> = { Left: false, Right: false };

function hsl(h: number, s: number, l: number, alpha = 1): string {
  return `hsla(${((h % 360) + 360) % 360}, ${s}%, ${l}%, ${alpha})`;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function distance2d(a: { x: number; y: number }, b: { x: number; y: number }): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function midpoint(a: PixelPoint, b: PixelPoint): PixelPoint {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

function toPixel(point: Point3, width: number, height: number): PixelPoint {
  return {
    x: (1 - point.x) * width,
    y: point.y * height,
  };
}

function smoothHand(label: HandLabel, current: Point3[]): Point3[] {
  const previous = smoothedHands[label];
  if (!previous || previous.length !== current.length) {
    smoothedHands[label] = current.map((p) => ({ ...p }));
    return current;
  }

  const smoothed = current.map((point, index) => ({
    x: lerp(previous[index].x, point.x, CFG.smoothing),
    y: lerp(previous[index].y, point.y, CFG.smoothing),
    z: lerp(previous[index].z, point.z, CFG.smoothing),
  }));
  smoothedHands[label] = smoothed;
  return smoothed;
}

function calculatePinchRatio(hand: Point3[]): number {
  const thumbTip = hand[4];
  const indexTip = hand[8];
  const wrist = hand[0];
  const middleMcp = hand[9];
  const pinchDistance = distance2d(thumbTip, indexTip);
  const palmSize = distance2d(wrist, middleMcp);
  return palmSize < 0.0001 ? 999 : pinchDistance / palmSize;
}

function updatePinchState(label: HandLabel, ratio: number): boolean {
  if (pinchStates[label]) {
    pinchStates[label] = ratio < CFG.pinchReleaseThreshold;
  } else {
    pinchStates[label] = ratio < CFG.pinchStartThreshold;
  }
  return pinchStates[label];
}

function resetRift(): void {
  riftState = 'IDLE';
  activationProgress = 0;
  riftOpenProgress = 0;
  riftStartDistance = null;
  savedLeftPoint = null;
  savedRightPoint = null;
  readyFlashFrames = 0;
}

function drawMirroredVideo(): void {
  ctx.save();
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  ctx.restore();

  const vignette = ctx.createRadialGradient(
    canvas.width / 2,
    canvas.height / 2,
    Math.min(canvas.width, canvas.height) * 0.16,
    canvas.width / 2,
    canvas.height / 2,
    Math.max(canvas.width, canvas.height) * 0.72,
  );
  vignette.addColorStop(0, 'rgba(0,0,0,0)');
  vignette.addColorStop(1, 'rgba(0,0,0,.42)');
  ctx.fillStyle = vignette;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function drawHandSkeleton(hand: Point3[], baseHue: number): void {
  ctx.save();
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  for (const [a, b] of CONNECTIONS) {
    const p1 = toPixel(hand[a], canvas.width, canvas.height);
    const p2 = toPixel(hand[b], canvas.width, canvas.height);
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.strokeStyle = hsl(baseHue + a * 5, 100, 58, 0.92);
    ctx.lineWidth = CFG.lineWidth;
    ctx.shadowColor = hsl(baseHue + a * 5, 100, 60);
    ctx.shadowBlur = 14;
    ctx.stroke();
  }
  ctx.restore();
}

function drawHandNodes(hand: Point3[], baseHue: number): void {
  const pulse = (Math.sin(frameCount * CFG.pulseSpeed) + 1) * 0.5;
  ctx.save();
  for (let i = 0; i < 21; i++) {
    const p = toPixel(hand[i], canvas.width, canvas.height);
    let radius = CFG.nodeRadius + (FINGERTIPS.has(i) ? 3 : 0) + pulse * 2;
    radius *= Math.max(0.75, canvas.width / 1280);

    ctx.beginPath();
    ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
    ctx.fillStyle = hsl(baseHue + i * 6, 100, 62);
    ctx.shadowColor = hsl(baseHue + i * 6, 100, 60);
    ctx.shadowBlur = 18;
    ctx.fill();

    ctx.shadowBlur = 0;
    ctx.beginPath();
    ctx.arc(p.x, p.y, Math.max(1.5, radius / 3), 0, Math.PI * 2);
    ctx.fillStyle = '#fff';
    ctx.fill();
  }
  ctx.restore();
}

function drawPinchIndicator(hand: Point3[], baseHue: number, isPinching: boolean): PixelPoint {
  const thumb = toPixel(hand[4], canvas.width, canvas.height);
  const index = toPixel(hand[8], canvas.width, canvas.height);
  const center = midpoint(thumb, index);

  ctx.save();
  if (isPinching) {
    ctx.beginPath();
    ctx.moveTo(thumb.x, thumb.y);
    ctx.lineTo(index.x, index.y);
    ctx.strokeStyle = hsl(baseHue, 100, 72);
    ctx.lineWidth = 5;
    ctx.shadowColor = hsl(baseHue, 100, 60);
    ctx.shadowBlur = 22;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(center.x, center.y, 18, 0, Math.PI * 2);
    ctx.strokeStyle = hsl(baseHue, 100, 72);
    ctx.lineWidth = 3;
    ctx.stroke();

    ctx.shadowBlur = 0;
    ctx.beginPath();
    ctx.arc(center.x, center.y, 6, 0, Math.PI * 2);
    ctx.fillStyle = '#fff';
    ctx.fill();
  } else {
    ctx.beginPath();
    ctx.arc(center.x, center.y, 10, 0, Math.PI * 2);
    ctx.strokeStyle = hsl(baseHue, 80, 55, 0.75);
    ctx.lineWidth = 2;
    ctx.stroke();
  }
  ctx.restore();
  return center;
}

function drawActivationEnergy(left: PixelPoint, right: PixelPoint, progress: number): void {
  const center = midpoint(left, right);
  ctx.save();
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(left.x, left.y);
  ctx.lineTo(right.x, right.y);
  ctx.strokeStyle = hsl(hue + progress * 100, 100, 60, 0.85);
  ctx.lineWidth = 3;
  ctx.shadowColor = hsl(hue + progress * 100, 100, 60);
  ctx.shadowBlur = 18 + 14 * progress;
  ctx.stroke();

  const lineLength = distance2d(left, right);
  const numberOfNodes = Math.max(6, Math.floor(lineLength / 35));
  for (let i = 0; i <= numberOfNodes; i++) {
    const t = i / numberOfNodes;
    const x = left.x + (right.x - left.x) * t;
    const y = left.y + (right.y - left.y) * t + Math.sin(frameCount * 0.30 + i * 0.85) * 8 * progress;
    const radius = 2 + progress * 4;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fillStyle = hsl(hue + i * 12, 100, 67);
    ctx.fill();
  }

  const pulse = (Math.sin(frameCount * 0.15) + 1) * 0.5;
  const radius = 8 + progress * 28 + pulse * 6 * progress;
  ctx.beginPath();
  ctx.arc(center.x, center.y, radius, 0, Math.PI * 2);
  ctx.strokeStyle = hsl(hue + progress * 80, 100, 58);
  ctx.lineWidth = 3;
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(center.x, center.y, Math.max(3, radius * 0.35), 0, Math.PI * 2);
  ctx.fillStyle = '#fff';
  ctx.shadowBlur = 26;
  ctx.fill();
  ctx.restore();
}

function buildRiftPolygon(left: PixelPoint, right: PixelPoint, progress: number): PixelPoint[] {
  const dx = right.x - left.x;
  const dy = right.y - left.y;
  const length = Math.max(1, Math.hypot(dx, dy));
  const normalX = -dy / length;
  const normalY = dx / length;
  const pointCount = 42;
  const scale = Math.max(0.72, canvas.width / 1280);
  const maximumHalfWidth = (2 + progress * 46) * scale;
  const upper: PixelPoint[] = [];
  const lower: PixelPoint[] = [];

  for (let i = 0; i <= pointCount; i++) {
    const t = i / pointCount;
    const baseX = left.x + dx * t;
    const baseY = left.y + dy * t;
    const envelope = Math.pow(Math.sin(Math.PI * t), 0.72);
    const movingNoise = Math.sin(t * Math.PI * 9 + frameCount * 0.18)
      + 0.55 * Math.sin(t * Math.PI * 19 - frameCount * 0.11);
    const halfWidth = maximumHalfWidth * envelope;
    const edgeNoise = movingNoise * (2 + progress * 5) * scale * envelope;
    const centerNoise = movingNoise * 2.5 * scale * envelope;

    upper.push({
      x: baseX + normalX * (halfWidth + edgeNoise + centerNoise),
      y: baseY + normalY * (halfWidth + edgeNoise + centerNoise),
    });
    lower.push({
      x: baseX + normalX * (-halfWidth + edgeNoise + centerNoise),
      y: baseY + normalY * (-halfWidth + edgeNoise + centerNoise),
    });
  }
  return [...upper, ...lower.reverse()];
}

function polygonPath(points: PixelPoint[]): Path2D {
  const path = new Path2D();
  if (!points.length) return path;
  path.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i++) path.lineTo(points[i].x, points[i].y);
  path.closePath();
  return path;
}

function drawRift(left: PixelPoint, right: PixelPoint, progress: number): void {
  if (progress <= 0) return;
  const polygon = buildRiftPolygon(left, right, progress);
  const path = polygonPath(polygon);
  const xs = polygon.map((p) => p.x);
  const ys = polygon.map((p) => p.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const width = Math.max(1, maxX - minX);
  const height = Math.max(1, maxY - minY);

  ctx.save();
  ctx.clip(path);
  const inside = ctx.createLinearGradient(minX, minY, maxX, maxY);
  inside.addColorStop(0, 'rgba(12,4,30,.98)');
  inside.addColorStop(.45, 'rgba(39,5,47,.98)');
  inside.addColorStop(1, 'rgba(4,10,34,.98)');
  ctx.fillStyle = inside;
  ctx.fillRect(minX - 20, minY - 20, width + 40, height + 40);

  for (let band = 0; band < 8; band++) {
    const bandY = minY + (((band / 8) + frameCount * 0.004) % 1) * height;
    ctx.beginPath();
    ctx.moveTo(minX, bandY);
    ctx.lineTo(maxX, bandY);
    ctx.strokeStyle = hsl(hue + band * 24, 100, 40, 0.42);
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  for (const star of STAR_FIELD) {
    const x = minX + star.x * width;
    const y = minY + star.y * height;
    const brightness = 65 + 20 * ((Math.sin(frameCount * 0.08 + star.hue) + 1) * 0.5);
    ctx.beginPath();
    ctx.arc(x, y, star.size, 0, Math.PI * 2);
    ctx.fillStyle = hsl(star.hue + hue, 80, brightness, 0.95);
    ctx.fill();
  }
  ctx.restore();

  ctx.save();
  ctx.lineJoin = 'round';
  ctx.shadowColor = hsl(hue, 100, 58);
  ctx.shadowBlur = 30;
  ctx.strokeStyle = hsl(hue, 100, 52, 0.75);
  ctx.lineWidth = 16;
  ctx.stroke(path);
  ctx.shadowBlur = 20;
  ctx.strokeStyle = hsl(hue + 45, 100, 67, 0.95);
  ctx.lineWidth = 7;
  ctx.stroke(path);
  ctx.shadowBlur = 10;
  ctx.strokeStyle = 'rgba(255,255,255,.95)';
  ctx.lineWidth = 2;
  ctx.stroke(path);

  for (const point of [left, right]) {
    const pulse = (Math.sin(frameCount * 0.2) + 1) * 0.5;
    const radius = 13 + pulse * 7;
    ctx.beginPath();
    ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
    ctx.strokeStyle = hsl(hue, 100, 56);
    ctx.lineWidth = 4;
    ctx.shadowBlur = 24;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(point.x, point.y, 5, 0, Math.PI * 2);
    ctx.fillStyle = '#fff';
    ctx.fill();
  }
  ctx.restore();
}

function drawInterface(detected: Set<HandLabel>): void {
  let status = '';
  let progress = 0;

  if (riftState === 'IDLE') {
    status = detected.size < 2 ? 'SHOW BOTH HANDS' : 'PINCH THUMB + INDEX ON BOTH HANDS';
  } else if (riftState === 'CHARGING') {
    status = `CHARGING ${Math.round(activationProgress * 100)}%`;
    progress = activationProgress;
  } else if (riftState === 'RIFT_READY') {
    status = '100% — KEEP PINCHING AND PULL APART';
    progress = 1;
  } else if (riftState === 'OPENING') {
    status = `OPENING RIFT ${Math.round(riftOpenProgress * 100)}%`;
    progress = riftOpenProgress;
  } else {
    status = 'RIFT OPEN — PRESS RESET';
    progress = 1;
  }

  const scale = Math.max(0.72, canvas.width / 1280);
  const x = 28 * scale;
  const y = 44 * scale;
  const barWidth = Math.min(360 * scale, canvas.width * 0.42);
  const barHeight = 13 * scale;

  ctx.save();
  ctx.font = `700 ${Math.max(14, 20 * scale)}px Inter, system-ui, sans-serif`;
  ctx.fillStyle = '#fff';
  ctx.shadowColor = 'rgba(0,0,0,.6)';
  ctx.shadowBlur = 10;
  ctx.fillText(status, x, y);
  ctx.shadowBlur = 0;

  ctx.strokeStyle = 'rgba(255,255,255,.30)';
  ctx.lineWidth = 2;
  ctx.strokeRect(x, y + 18 * scale, barWidth, barHeight);
  if (progress > 0) {
    ctx.fillStyle = hsl(hue, 100, 60, 0.95);
    ctx.shadowColor = hsl(hue, 100, 60);
    ctx.shadowBlur = 12;
    ctx.fillRect(x, y + 18 * scale, barWidth * Math.min(1, Math.max(0, progress)), barHeight);
  }
  ctx.restore();
}

function updateRiftState(bothPinching: boolean, centers: Partial<Record<HandLabel, PixelPoint>>): void {
  const left = centers.Left;
  const right = centers.Right;
  const widthScale = Math.max(0.65, canvas.width / 1280);
  const pullStart = CFG.pullStartPixels * widthScale;
  const fullOpen = CFG.fullOpenPixels * widthScale;

  if (riftState === 'IDLE') {
    activationProgress = 0;
    riftOpenProgress = 0;
    if (bothPinching) riftState = 'CHARGING';
    return;
  }

  if (riftState === 'CHARGING') {
    if (!bothPinching || !left || !right) {
      resetRift();
      return;
    }
    activationProgress = Math.min(1, activationProgress + 1 / CFG.activationHoldFrames);
    if (activationProgress >= 1) {
      riftState = 'RIFT_READY';
      savedLeftPoint = { ...left };
      savedRightPoint = { ...right };
      riftStartDistance = distance2d(savedLeftPoint, savedRightPoint);
      readyFlashFrames = 8;
    }
    return;
  }

  if (riftState === 'RIFT_READY') {
    if (!bothPinching || !left || !right || riftStartDistance === null) {
      resetRift();
      return;
    }
    savedLeftPoint = { ...left };
    savedRightPoint = { ...right };
    const pullDistance = distance2d(savedLeftPoint, savedRightPoint) - riftStartDistance;
    if (pullDistance > pullStart) riftState = 'OPENING';
    return;
  }

  if (riftState === 'OPENING') {
    if (!bothPinching || !left || !right || riftStartDistance === null) {
      resetRift();
      return;
    }
    savedLeftPoint = { ...left };
    savedRightPoint = { ...right };
    const pullDistance = Math.max(0, distance2d(savedLeftPoint, savedRightPoint) - riftStartDistance);
    riftOpenProgress = Math.min(1, Math.max(0, pullDistance / fullOpen));
    if (riftOpenProgress >= 1) {
      riftState = 'PORTAL_ACTIVE';
      readyFlashFrames = 6;
    }
    return;
  }

  riftOpenProgress = 1;
  if (bothPinching && left && right) {
    savedLeftPoint = { ...left };
    savedRightPoint = { ...right };
  }
}

function normalizeHandedness(result: HandLandmarkerResult, index: number): HandLabel | null {
  const category = result.handedness?.[index]?.[0]?.categoryName;
  if (category === 'Left' || category === 'Right') return category;
  return null;
}

function renderResults(result: HandLandmarkerResult): void {
  const detected = new Set<HandLabel>();
  const centers: Partial<Record<HandLabel, PixelPoint>> = {};

  for (let i = 0; i < result.landmarks.length; i++) {
    const label = normalizeHandedness(result, i);
    if (!label || detected.has(label)) continue;
    detected.add(label);

    const hand = smoothHand(
      label,
      result.landmarks[i].map((landmark: NormalizedLandmark) => ({
        x: landmark.x,
        y: landmark.y,
        z: landmark.z,
      })),
    );

    const ratio = calculatePinchRatio(hand);
    const isPinching = updatePinchState(label, ratio);
    const baseHue = label === 'Left' ? hue : hue + 140;
    drawHandSkeleton(hand, baseHue);
    drawHandNodes(hand, baseHue);
    centers[label] = drawPinchIndicator(hand, baseHue, isPinching);
  }

  for (const label of HAND_LABELS) {
    if (!detected.has(label)) {
      pinchStates[label] = false;
      delete smoothedHands[label];
    }
  }

  const bothHandsVisible = detected.has('Left') && detected.has('Right');
  const bothPinching = bothHandsVisible && pinchStates.Left && pinchStates.Right;
  updateRiftState(bothPinching, centers);

  if ((riftState === 'CHARGING' || riftState === 'RIFT_READY') && bothPinching && centers.Left && centers.Right) {
    drawActivationEnergy(centers.Left, centers.Right, activationProgress);
  }

  if ((riftState === 'OPENING' || riftState === 'PORTAL_ACTIVE') && savedLeftPoint && savedRightPoint) {
    drawRift(savedLeftPoint, savedRightPoint, riftOpenProgress);
  }

  if (readyFlashFrames > 0) {
    const strength = Math.min(0.55, (readyFlashFrames / 8) * 0.55);
    ctx.fillStyle = `rgba(255,255,255,${strength})`;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    readyFlashFrames -= 1;
  }

  drawInterface(detected);
}

function resizeCanvasToVideo(): void {
  if (!video.videoWidth || !video.videoHeight) return;
  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }
}

function renderLoop(): void {
  if (!handLandmarker || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    animationId = requestAnimationFrame(renderLoop);
    return;
  }

  resizeCanvasToVideo();
  drawMirroredVideo();
  frameCount += 1;
  hue = (hue + CFG.hueSpeed) % 360;

  if (video.currentTime !== lastVideoTime) {
    const result = handLandmarker.detectForVideo(video, performance.now());
    renderResults(result);
    lastVideoTime = video.currentTime;
  }

  animationId = requestAnimationFrame(renderLoop);
}

async function createLandmarker(): Promise<HandLandmarker> {
  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm',
  );

  const options = {
    baseOptions: {
      modelAssetPath:
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
      delegate: 'GPU' as const,
    },
    runningMode: 'VIDEO' as const,
    numHands: 2,
    minHandDetectionConfidence: 0.65,
    minHandPresenceConfidence: 0.60,
    minTrackingConfidence: 0.60,
  };

  try {
    return await HandLandmarker.createFromOptions(vision, options);
  } catch (gpuError) {
    console.warn('GPU delegate unavailable, retrying with CPU.', gpuError);
    return HandLandmarker.createFromOptions(vision, {
      ...options,
      baseOptions: {
        ...options.baseOptions,
        delegate: 'CPU',
      },
    });
  }
}

async function startCamera(): Promise<void> {
  startButton.disabled = true;
  startButton.textContent = 'Loading…';
  errorText.textContent = '';

  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('This browser does not support camera access. Try a recent Chrome, Edge, Firefox, or Safari.');
    }

    handLandmarker ??= await createLandmarker();
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: 'user',
      },
      audio: false,
    });

    video.srcObject = mediaStream;
    await video.play();
    resizeCanvasToVideo();

    startOverlay.classList.add('hidden');
    statusChip.textContent = 'LIVE';
    statusChip.classList.add('live');
    resetButton.disabled = false;
    resetRift();
    cancelAnimationFrame(animationId);
    animationId = requestAnimationFrame(renderLoop);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Could not start the camera.';
    errorText.textContent = message;
    startButton.disabled = false;
    startButton.textContent = 'Try again';
  }
}

startButton.addEventListener('click', () => void startCamera());
resetButton.addEventListener('click', resetRift);
window.addEventListener('keydown', (event) => {
  if (event.key.toLowerCase() === 'r') resetRift();
});
window.addEventListener('beforeunload', () => {
  cancelAnimationFrame(animationId);
  mediaStream?.getTracks().forEach((track) => track.stop());
  handLandmarker?.close();
});
