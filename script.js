// DOM elements
const video = document.getElementById('webcam');
const canvas = document.getElementById('output');
const ctx = canvas.getContext('2d');
const startBtn = document.getElementById('startBtn');
const resetBtn = document.getElementById('resetBtn');
const detectionWarning = document.getElementById('detection-warning');
const repCounter = document.getElementById('rep-counter');
const correctCounter = document.getElementById('correct-counter');
const incorrectCounter = document.getElementById('incorrect-counter');
const feedback = document.getElementById('feedback');
const depthBar = document.getElementById('depth-bar');
const depthIndicator = document.getElementById('depth-indicator');
const exitFullscreenBtn = document.getElementById('exitFullscreenBtn');
const videoContainer = document.querySelector('.video-container');
const counterContainer = document.getElementById('counter-container');

// Fullscreen mode flag
let isFullscreenMode = false;

// Model and detection variables
let detector;
let model;
let rafId;

// Thresholds from Python script
const STANDING_ANGLE = 170.0;
const GOOD_SQUAT_LOWER_ANGLE = 45.0;
const GOOD_SQUAT_UPPER_ANGLE = 95.0;
const MIN_REQUIRED_LOWER_ANGLE = 95.0;
const MIN_REQUIRED_UPPER_ANGLE = 140.0;
const TOO_DEEP_ANGLE = 35.0;
const KNEE_CAVING_THRESHOLD = 1.1;
const CORRECT_MIN_RATIO = 0.85;
const CORRECT_MAX_RATIO = 1.61;
const KNEE_HISTORY_LENGTH = 10;
const KNEE_CAVING_FRAME_THRESHOLD = 3;
const MAX_REPS_PER_SESSION = 12;
const MIN_ANGLE_CHANGE_FOR_REP = 30;

// Rep counting and form tracking
let repState = 'up';
let repCounted = false;
let incorrectRepCounted = false;
let counter = 0;
let correctReps = 0;
let incorrectReps = 0;
let tooDeepCount = 0;
let notDeepEnoughCount = 0;
let kneeCavingCount = 0;
let currentRepTooDeep = false;
let currentRepNotDeepEnough = false;
let currentRepKneesCaved = false;
let minAngleInCurrentRep = 180;
let maxAngleInCurrentRep = 0;
let minKneeRatioInCurrentRep = 2.0;
let kneeCavingConsecutiveFrames = 0;
let kneeRatioHistory = [];
let sessionComplete = false;
let sessionCompletedTime = 0;

// Detection loss tracking
let detectionLostFrames = 0;
const MAX_DETECTION_LOST_FRAMES = 90;
let detectionFound = false;

// Smoothing for landmarks and angles
let landmarkHistory = {};
const SMOOTHING_FACTOR = 0.2;
let smoothedAngles = { knee: [] };
const ANGLE_SMOOTHING_WINDOW = 10;

// Feedback persistence
let feedbackHistory = new Map();
const FEEDBACK_PERSISTENCE = 5;
const FEEDBACK_THRESHOLD = 3;
let feedbackStates = {
  'too_deep': { count: 0, active: false },
  'not_deep_enough': { count: 0, active: false },
  'knees_caving': { count: 0, active: false },
  'good_form': { count: 0, active: false }
};

// Session data for visualization
let sessionData = {
  frame_count: [],
  knee_angles: [],
  knee_ratios: [],
  rep_markers: [],
  rep_knee_angles: [],
  rep_feedback: []
};

// Keypoint colors and connections
const keypointColors = Array(17).fill('#FF0000');
const connectedKeypoints = [
  ['left_hip', 'right_hip'],
  ['left_hip', 'left_knee'], ['left_knee', 'left_ankle'],
  ['right_hip', 'right_knee'], ['right_knee', 'right_ankle']
];

const keypointMap = {
  nose: 0, left_eye: 1, right_eye: 2, left_ear: 3, right_ear: 4,
  left_shoulder: 5, right_shoulder: 6, left_elbow: 7, right_elbow: 8,
  left_wrist: 9, right_wrist: 10, left_hip: 11, right_hip: 12,
  left_knee: 13, right_knee: 14, left_ankle: 15, right_ankle: 16
};

// Helper Functions
function smoothLandmarks(keypoints) {
  const smoothedKeypoints = [...keypoints];
  for (let i = 0; i < keypoints.length; i++) {
    const keypoint = keypoints[i];
    const id = i;
    if (keypoint.score > 0.2) {
      const position = { x: keypoint.x, y: keypoint.y };
      if (landmarkHistory[id]) {
        const smoothedX = SMOOTHING_FACTOR * position.x + (1 - SMOOTHING_FACTOR) * landmarkHistory[id].x;
        const smoothedY = SMOOTHING_FACTOR * position.y + (1 - SMOOTHING_FACTOR) * landmarkHistory[id].y;
        smoothedKeypoints[i] = { ...keypoint, x: smoothedX, y: smoothedY };
        landmarkHistory[id] = { x: smoothedX, y: smoothedY };
      } else {
        landmarkHistory[id] = position;
      }
    }
  }
  return smoothedKeypoints;
}

function calculateAngle(a, b, c) {
  const ab = { x: b.x - a.x, y: b.y - a.y };
  const cb = { x: b.x - c.x, y: b.y - c.y };
  const dot = ab.x * cb.x + ab.y * cb.y;
  const magAB = Math.sqrt(ab.x * ab.x + ab.y * ab.y);
  const magCB = Math.sqrt(cb.x * cb.x + cb.y * cb.y);
  const angleRad = Math.acos(dot / (magAB * magCB));
  return angleRad * (180 / Math.PI);
}

function smoothAngle(newValue, angleType) {
  smoothedAngles[angleType].push(newValue);
  if (smoothedAngles[angleType].length > ANGLE_SMOOTHING_WINDOW) {
    smoothedAngles[angleType].shift();
  }
  return smoothedAngles[angleType].reduce((sum, val) => sum + val, 0) / smoothedAngles[angleType].length;
}

function updateFeedbackState(type, condition, text, color) {
  if (condition) {
    feedbackStates[type].count = Math.min(feedbackStates[type].count + 1, FEEDBACK_THRESHOLD + 5);
    if (feedbackStates[type].count >= FEEDBACK_THRESHOLD && !feedbackStates[type].active) {
      feedbackStates[type].active = true;
      feedbackHistory.set(text, { color: color, count: FEEDBACK_PERSISTENCE });
    } else if (feedbackStates[type].active) {
      feedbackHistory.set(text, { color: color, count: FEEDBACK_PERSISTENCE });
    }
  } else {
    feedbackStates[type].count = Math.max(0, feedbackStates[type].count - 1);
    if (feedbackStates[type].count === 0 && feedbackStates[type].active) {
      feedbackStates[type].active = false;
      feedbackHistory.delete(text);
    }
  }
}

function drawFeedback(messages) {
  requestAnimationFrame(() => {
    const expiredMessages = [];
    feedbackHistory.forEach((item, text) => {
      item.count--;
      if (item.count <= 0) expiredMessages.push(text);
    });
    expiredMessages.forEach(text => feedbackHistory.delete(text));
    if (feedbackHistory.size === 0) {
      feedback.classList.add('hidden');
      feedback.innerHTML = '';
      return;
    }
    feedback.classList.remove('hidden');
    const fragment = document.createDocumentFragment();
    feedbackHistory.forEach((item, text) => {
      const opacity = Math.max(0.3, item.count / FEEDBACK_PERSISTENCE);
      let icon = 'fa-info-circle';
      if (text.includes('Good')) icon = 'fa-check-circle';
      else if (text.includes('Knees')) icon = 'fa-exclamation-triangle';
      else if (text.includes('Deep')) icon = 'fa-arrow-down';
      const div = document.createElement('div');
      div.style.color = item.color;
      div.style.opacity = opacity;
      const iconEl = document.createElement('i');
      iconEl.className = `fas ${icon}`;
      div.appendChild(iconEl);
      div.appendChild(document.createTextNode(` ${text}`));
      fragment.appendChild(div);
    });
    feedback.innerHTML = '';
    feedback.appendChild(fragment);
  });
}

function drawArrow(ctx, startPoint, endPoint, color, thickness = 3) {
  ctx.beginPath();
  ctx.moveTo(startPoint.x, startPoint.y);
  ctx.lineTo(endPoint.x, endPoint.y);
  ctx.strokeStyle = color;
  ctx.lineWidth = thickness;
  ctx.stroke();
  const angle = Math.atan2(endPoint.y - startPoint.y, endPoint.x - startPoint.x);
  const headLength = 15;
  ctx.beginPath();
  ctx.moveTo(endPoint.x, endPoint.y);
  ctx.lineTo(endPoint.x - headLength * Math.cos(angle - Math.PI / 6), endPoint.y - headLength * Math.sin(angle - Math.PI / 6));
  ctx.lineTo(endPoint.x - headLength * Math.cos(angle + Math.PI / 6), endPoint.y - headLength * Math.sin(angle + Math.PI / 6));
  ctx.lineTo(endPoint.x, endPoint.y);
  ctx.fillStyle = color;
  ctx.fill();
}

async function setupDetector() {
  const modelType = poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING;
  model = poseDetection.SupportedModels.MoveNet;
  detector = await poseDetection.createDetector(model, { modelType, enableSmoothing: true });
  console.log('MoveNet model loaded successfully');
}

async function setupCamera() {
  const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
  const constraints = {
    video: {
      facingMode: 'user',
      width: { ideal: isMobile ? 1280 : 1920 },
      height: { ideal: isMobile ? 720 : 1080 }
    }
  };
  try {
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = stream;
    return new Promise((resolve) => {
      video.onloadedmetadata = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        video.style.transform = '';
        canvas.style.transform = '';
        video.style.objectFit = 'cover';
        canvas.style.objectFit = 'cover';
        document.body.classList.add('camera-initialized');
        console.log(`Camera dimensions: ${canvas.width}x${canvas.height}`);
        resolve(video);
      };
    });
  } catch (error) {
    console.error('Error accessing webcam:', error);
    alert('Error accessing webcam. Please ensure a webcam is connected and permissions are granted.');
    throw error;
  }
}

const TARGET_FPS = 30;
let lastDetectionTime = 0;

async function detectPose() {
  const now = performance.now();
  if (now - lastDetectionTime < 1000 / TARGET_FPS) {
    rafId = requestAnimationFrame(detectPose);
    return;
  }
  lastDetectionTime = now;
  if (!detector || !video.readyState) {
    video.onloadeddata = () => { rafId = requestAnimationFrame(detectPose); };
    return;
  }
  detectionFound = false;
  if (video.readyState < 2) {
    await new Promise((resolve) => { video.onloadeddata = () => resolve(video); });
  }
  const poses = await detector.estimatePoses(video);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  if (poses && poses.length > 0) {
    const pose = poses[0];
    if (pose.keypoints && pose.keypoints.length > 0) {
      const smoothedKeypoints = smoothLandmarks(pose.keypoints);
      processDetection(smoothedKeypoints);
      drawPose({ ...pose, keypoints: smoothedKeypoints });
    }
  }
  if (!detectionFound) {
    handleDetectionLoss();
  } else {
    detectionWarning.classList.add('hidden');
    detectionLostFrames = 0;
  }
  rafId = requestAnimationFrame(detectPose);
}

function processDetection(keypoints) {
  sessionData.frame_count.push(sessionData.frame_count.length);
  const requiredLandmarks = [11, 12, 13, 14, 15, 16];
  const visibleLandmarks = requiredLandmarks.filter(idx => keypoints[idx].score > 0.2);
  if (visibleLandmarks.length >= requiredLandmarks.length - 1) {
    detectionFound = true;
    const leftHip = keypoints[11];
    const rightHip = keypoints[12];
    const leftKnee = keypoints[13];
    const rightKnee = keypoints[14];
    const leftAnkle = keypoints[15];
    const rightAnkle = keypoints[16];
    const rawKneeAngle = calculateAngle(rightHip, rightKnee, rightAnkle);
    const kneeAngle = smoothAngle(rawKneeAngle, 'knee');
    console.log(`Raw Knee Angle: ${rawKneeAngle.toFixed(1)}°, Smoothed Knee Angle: ${kneeAngle.toFixed(1)}°`);
    if (kneeAngle < minAngleInCurrentRep) {
      minAngleInCurrentRep = kneeAngle;
    }
    if (kneeAngle > maxAngleInCurrentRep) {
      maxAngleInCurrentRep = kneeAngle;
    }
    const per = Math.min(100, Math.max(0, (kneeAngle - TOO_DEEP_ANGLE) / (STANDING_ANGLE - TOO_DEEP_ANGLE) * 100));
    depthBar.classList.remove('hidden');
    const barHeight = 300;
    const filledHeight = per * barHeight / 100;
    depthIndicator.style.bottom = `${filledHeight}px`;
    let status, barColor;
    if (kneeAngle > MIN_REQUIRED_UPPER_ANGLE) {
      status = 'TOO SHALLOW';
      barColor = '#FFA500';
      currentRepNotDeepEnough = true;
    } else if (kneeAngle >= MIN_REQUIRED_LOWER_ANGLE && kneeAngle <= MIN_REQUIRED_UPPER_ANGLE) {
      status = 'DEEPER!';
      barColor = '#FF8C00';
      currentRepNotDeepEnough = true;
    } else if (kneeAngle >= GOOD_SQUAT_LOWER_ANGLE && kneeAngle <= GOOD_SQUAT_UPPER_ANGLE) {
      status = 'GOOD!';
      barColor = '#4CAF50';
      currentRepNotDeepEnough = false;
      currentRepTooDeep = false;
    } else if (kneeAngle < TOO_DEEP_ANGLE) {
      status = 'TOO DEEP';
      barColor = '#FF0000';
      currentRepTooDeep = true;
    } else {
      status = 'GOOD';
      barColor = '#FFFF00';
      currentRepNotDeepEnough = false;
    }
    let kneeAnkleRatio = 0;
    if (leftKnee.score > 0.2 && rightKnee.score > 0.2 && leftAnkle.score > 0.2 && rightAnkle.score > 0.2) {
      const kneeWidth = Math.abs(rightKnee.x - leftKnee.x);
      const ankleWidth = Math.abs(rightAnkle.x - leftAnkle.x);
      kneeAnkleRatio = ankleWidth > 0 ? kneeWidth / ankleWidth : 2.0;
      console.log(`Knee Width: ${kneeWidth.toFixed(1)}, Ankle Width: ${ankleWidth.toFixed(1)}, Raw Ratio: ${kneeAnkleRatio.toFixed(2)}`);
      if (kneeAnkleRatio > 0) {
        kneeRatioHistory.push(kneeAnkleRatio);
        if (kneeRatioHistory.length > KNEE_HISTORY_LENGTH) {
          kneeRatioHistory.shift();
        }
        const smoothedKneeRatio = kneeRatioHistory.reduce((sum, val) => sum + val, 0) / kneeRatioHistory.length;
        console.log(`Smoothed Knee Ratio: ${smoothedKneeRatio.toFixed(2)}`);
        if (smoothedKneeRatio < minKneeRatioInCurrentRep) {
          minKneeRatioInCurrentRep = smoothedKneeRatio;
        }
        let kneeLineColor = '#4CAF50';
        let kneeStatus = 'GOOD';
        if (kneeAngle <= MIN_REQUIRED_UPPER_ANGLE && kneeAngle < (STANDING_ANGLE - 10)) {
          if (smoothedKneeRatio < KNEE_CAVING_THRESHOLD) {
            kneeLineColor = '#FF0000';
            kneeStatus = 'BAD';
            kneeCavingConsecutiveFrames++;
            console.log(`Knee Caving Check: Angle=${kneeAngle.toFixed(1)}°, Smoothed Ratio=${smoothedKneeRatio.toFixed(2)}, Frames=${kneeCavingConsecutiveFrames}`);
            if (kneeCavingConsecutiveFrames >= KNEE_CAVING_FRAME_THRESHOLD) {
              currentRepKneesCaved = true;
              console.log('Knees caved detected!');
            }
          } else {
            kneeLineColor = '#4CAF50';
            kneeStatus = 'GOOD';
            kneeCavingConsecutiveFrames = Math.max(0, kneeCavingConsecutiveFrames - 1);
            if (kneeCavingConsecutiveFrames < KNEE_CAVING_FRAME_THRESHOLD) {
              currentRepKneesCaved = false;
              console.log('Knee caving reset');
            }
          }
        } else {
          kneeCavingConsecutiveFrames = 0;
          currentRepKneesCaved = false;
          console.log('Knee caving counter reset (not in middle of rep or standing)');
        }
        ctx.beginPath();
        ctx.moveTo(leftKnee.x, leftKnee.y);
        ctx.lineTo(rightKnee.x, rightKnee.y);
        ctx.strokeStyle = kneeLineColor;
        ctx.lineWidth = currentRepKneesCaved ? 6 : 4;
        ctx.stroke();
        if (smoothedKneeRatio < KNEE_CAVING_THRESHOLD && kneeAngle <= MIN_REQUIRED_UPPER_ANGLE && kneeAngle < (STANDING_ANGLE - 10)) {
          const idealKneeDistance = ankleWidth * CORRECT_MIN_RATIO * 1.3;
          const leftIdealX = rightKnee.x - idealKneeDistance;
          ctx.beginPath();
          ctx.moveTo(leftIdealX, leftKnee.y);
          ctx.lineTo(rightKnee.x, rightKnee.y);
          ctx.strokeStyle = '#4CAF50';
          ctx.lineWidth = 2;
          ctx.stroke();
          if (currentRepKneesCaved) {
            const midY = (leftKnee.y + rightKnee.y) / 2;
            drawArrow(ctx, { x: leftKnee.x + 10, y: midY }, { x: leftKnee.x - 20, y: midY }, '#FF0000', 4);
            drawArrow(ctx, { x: rightKnee.x - 10, y: midY }, { x: rightKnee.x + 20, y: midY }, '#FF0000', 4);
          }
        }
        sessionData.knee_ratios.push(smoothedKneeRatio);
      }
    }
    sessionData.knee_angles.push(kneeAngle);
    const feedbackMessages = [];
    updateFeedbackState('too_deep', currentRepTooDeep, 'Squat too deep!', '#FF0000');
    updateFeedbackState('not_deep_enough', currentRepNotDeepEnough, 'Squat deeper!', '#FFA500');
    updateFeedbackState('knees_caving', currentRepKneesCaved && kneeAngle <= MIN_REQUIRED_UPPER_ANGLE && kneeAngle < (STANDING_ANGLE - 10), 'Push knees outward!', '#FF0000');
    updateFeedbackState('good_form', !currentRepTooDeep && !currentRepNotDeepEnough && !currentRepKneesCaved && kneeAngle <= GOOD_SQUAT_UPPER_ANGLE, 'Good form!', '#4CAF50');
    if (currentRepKneesCaved && kneeAngle <= MIN_REQUIRED_UPPER_ANGLE && kneeAngle < (STANDING_ANGLE - 10)) {
      ctx.font = '24px Arial';
      ctx.fillStyle = '#FF0000';
      ctx.fillText('KNEES CAVING IN!', canvas.width / 2 - 100, 50);
    }
    const angleChange = maxAngleInCurrentRep - minAngleInCurrentRep;
    if (!repCounted && kneeAngle < (GOOD_SQUAT_UPPER_ANGLE - 10) && angleChange >= MIN_ANGLE_CHANGE_FOR_REP && !sessionComplete) {
      let isIncorrect = false;
      let incorrectReason = '';
      if (kneeAngle < TOO_DEEP_ANGLE) {
        isIncorrect = true;
        incorrectReason = 'Too deep';
        tooDeepCount++;
      } else if (kneeAngle > MIN_REQUIRED_UPPER_ANGLE) {
        isIncorrect = true;
        incorrectReason = 'Not deep enough';
        notDeepEnoughCount++;
      } else if (currentRepKneesCaved) {
        isIncorrect = true;
        incorrectReason = 'Knees caving inward';
        kneeCavingCount++;
      }
      if (!isIncorrect) {
        counter++;
        correctReps++;
        repCounted = true;
        sessionData.rep_markers.push(sessionData.frame_count.length);
        sessionData.rep_knee_angles.push(minAngleInCurrentRep);
        sessionData.rep_feedback.push('Good form');
        console.log(`Good Rep Counted: Total=${counter}, Correct=${correctReps}`);
      } else {
        incorrectReps++;
        incorrectRepCounted = true;
        sessionData.rep_markers.push(sessionData.frame_count.length);
        sessionData.rep_knee_angles.push(minAngleInCurrentRep);
        sessionData.rep_feedback.push(incorrectReason);
        console.log(`Incorrect Rep Counted: Total=${counter}, Incorrect=${incorrectReps}, Reason=${incorrectReason}`);
      }
      repCounter.textContent = `Total: ${counter}`;
      correctCounter.textContent = `Good: ${correctReps}`;
      incorrectCounter.textContent = `Bad: ${incorrectReps}`;
      if (counter >= MAX_REPS_PER_SESSION && !sessionComplete) {
        sessionComplete = true;
        sessionCompletedTime = Date.now();
        feedbackHistory.set('Session complete! Take rest.', { color: '#FF0000', count: FEEDBACK_PERSISTENCE * 2 });
        console.log('Session complete!');
      }
      minAngleInCurrentRep = 180;
      maxAngleInCurrentRep = 0;
      minKneeRatioInCurrentRep = 2.0;
    }
    if (repCounted && kneeAngle > (STANDING_ANGLE - 10)) {
      repCounted = false;
      currentRepKneesCaved = false;
      currentRepTooDeep = false;
      currentRepNotDeepEnough = false;
      console.log('Rep counting reset: Ready for next rep');
    }
    if (incorrectRepCounted && kneeAngle > (STANDING_ANGLE - 10)) {
      incorrectRepCounted = false;
    }
    drawFeedback(feedbackMessages);
  } else {
    console.log(`Detection weak: Visible landmarks = ${visibleLandmarks.length}/${requiredLandmarks.length}`);
    detectionFound = false;
  }
}

function handleDetectionLoss() {
  detectionLostFrames++;
  if (detectionLostFrames >= MAX_DETECTION_LOST_FRAMES) {
    detectionWarning.classList.remove('hidden');
    console.log('Detection lost for too long, showing warning...');
  }
}

function drawPose(pose) {
  if (!pose || !pose.keypoints) return;
  const keypoints = pose.keypoints;
  for (let i = 0; i < keypoints.length; i++) {
    const keypoint = keypoints[i];
    if (keypoint.score > 0.2) {
      ctx.fillStyle = keypointColors[i];
      ctx.fillRect(keypoint.x - 4, keypoint.y - 4, 8, 8);
    }
  }
  ctx.lineWidth = 4;
  connectedKeypoints.forEach(([keypointA, keypointB]) => {
    const indexA = keypointMap[keypointA];
    const indexB = keypointMap[keypointB];
    const keypointA_obj = keypoints[indexA];
    const keypointB_obj = keypoints[indexB];
    if (keypointA_obj.score > 0.2 && keypointB_obj.score > 0.2) {
      ctx.strokeStyle = '#00FF00';
      ctx.beginPath();
      ctx.moveTo(keypointA_obj.x, keypointA_obj.y);
      ctx.lineTo(keypointB_obj.x, keypointB_obj.y);
      ctx.stroke();
    }
  });
}

function resetTrainer() {
  counter = 0;
  correctReps = 0;
  incorrectReps = 0;
  tooDeepCount = 0;
  notDeepEnoughCount = 0;
  kneeCavingCount = 0;
  repCounted = false;
  incorrectRepCounted = false;
  sessionComplete = false;
  sessionCompletedTime = 0;
  minAngleInCurrentRep = 180;
  maxAngleInCurrentRep = 0;
  minKneeRatioInCurrentRep = 2.0;
  kneeCavingConsecutiveFrames = 0;
  kneeRatioHistory = [];
  detectionLostFrames = 0;
  feedbackHistory.clear();
  feedback.classList.add('hidden');
  repCounter.textContent = `Total: ${counter}`;
  correctCounter.textContent = `Good: ${correctReps}`;
  incorrectCounter.textContent = `Bad: ${incorrectReps}`;
  counterContainer.classList.remove('hidden');
  depthBar.classList.remove('hidden');
  sessionData = {
    frame_count: [],
    knee_angles: [],
    knee_ratios: [],
    rep_markers: [],
    rep_knee_angles: [],
    rep_feedback: []
  };
  console.log('Trainer reset');
}

function enterFullscreenMode() {
  if (isFullscreenMode) return;
  document.body.classList.add('fullscreen-mode');
  exitFullscreenBtn.classList.remove('hidden');
  isFullscreenMode = true;
  if (videoContainer.requestFullscreen) {
    videoContainer.requestFullscreen().catch(err => console.log('Error enabling fullscreen:', err));
  }
}

function exitFullscreenMode() {
  if (!isFullscreenMode) return;
  document.body.classList.remove('fullscreen-mode');
  document.body.classList.remove('camera-active');
  exitFullscreenBtn.classList.add('hidden');
  isFullscreenMode = false;
  if (document.fullscreenElement) {
    document.exitFullscreen().catch(err => console.log('Error exiting fullscreen:', err));
  }
  if (video.srcObject) {
    video.srcObject.getTracks().forEach(track => track.stop());
    video.srcObject = null;
  }
  startBtn.disabled = false;
  startBtn.textContent = 'Start Camera';
}

async function startApp() {
  try {
    startBtn.disabled = true;
    startBtn.textContent = 'Loading...';
    if (!detector) await setupDetector();
    if (!video.srcObject) await setupCamera();
    document.body.classList.add('camera-active');
    enterFullscreenMode();
    video.play();
    detectPose();
    startBtn.textContent = "Running";
    resetBtn.disabled = false;
    resetTrainer();
    console.log('Squat Trainer started successfully');
  } catch (error) {
    console.error('Error starting the application:', error);
    startBtn.disabled = false;
    startBtn.textContent = 'Start Camera';
    alert('Error starting the application. Please ensure a webcam is connected and permissions are granted.');
  }
}

startBtn.addEventListener('click', startApp);
resetBtn.addEventListener('click', resetTrainer);
exitFullscreenBtn.addEventListener('click', exitFullscreenMode);

function calculatePerformanceScore() {
  const repScore = counter > 0 ? (correctReps / counter) * 100 : 0;
  const formIssuesPenalty = (tooDeepCount + notDeepEnoughCount + kneeCavingCount) * 5;
  return Math.round(Math.max(0, Math.min(100, repScore - formIssuesPenalty)));
}

function createResultsObject() {
  const performanceScore = calculatePerformanceScore();
  const currentDate = new Date();
  return {
    workoutName: "Squat",
    date: currentDate.toLocaleDateString(),
    time: currentDate.toLocaleTimeString(),
    sets: 1,
    reps: { total: counter, correct: correctReps, incorrect: incorrectReps },
    formIssues: {
      tooDeep: tooDeepCount,
      notDeepEnough: notDeepEnoughCount,
      kneesCaving: kneeCavingCount
    },
    performanceScore: performanceScore,
    sessionId: Date.now().toString(36) + Math.random().toString(36).substr(2),
    sessionData: sessionData
  };
}

function saveSessionData(resultsObject) {
  try {
    let sessionHistory = JSON.parse(localStorage.getItem('exerciseSessionHistory')) || [];
    sessionHistory.push(resultsObject);
    if (sessionHistory.length > 20) sessionHistory = sessionHistory.slice(-20);
    localStorage.setItem('exerciseSessionHistory', JSON.stringify(sessionHistory));
    localStorage.setItem('lastExerciseSession', JSON.stringify(resultsObject));
    console.log('Session data saved successfully');
    return true;
  } catch (error) {
    console.error('Error saving session data:', error);
    return false;
  }
}

let downloadInProgress = false;

function generateResultsFile() {
  if (downloadInProgress) return;
  downloadInProgress = true;
  const resultsObject = createResultsObject();
  saveSessionData(resultsObject);
  const jsonData = JSON.stringify(resultsObject, null, 2);
  const blob = new Blob([jsonData], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `squat_results_${resultsObject.sessionId}.json`;
  a.style.display = 'none';
  document.body.appendChild(a);
  setTimeout(() => {
    a.click();
    setTimeout(() => {
      document.body.removeChild(a);
      URL.revokeObjectURL(a.href);
      downloadInProgress = false;
    }, 1000);
  }, 500);
}

window.addEventListener('beforeunload', () => {
  if (rafId) cancelAnimationFrame(rafId);
  if (video.srcObject) video.srcObject.getTracks().forEach(track => track.stop());
  generateResultsFile();
});

console.log('Squat Trainer initialized. Press Start Camera to begin.');
