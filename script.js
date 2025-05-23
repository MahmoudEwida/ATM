// DOM elements
const video = document.getElementById('webcam');
const canvas = document.getElementById('output');
const ctx = canvas.getContext('2d');
const startBtn = document.getElementById('startBtn');
const resetBtn = document.getElementById('resetBtn');
const countdown = document.getElementById('countdown');
const detectionWarning = document.getElementById('detection-warning');
const positionBar = document.getElementById('position-bar');
const positionProgress = document.getElementById('position-progress');
const repCounter = document.getElementById('rep-counter');
const feedback = document.getElementById('feedback');
const exitFullscreenBtn = document.getElementById('exitFullscreenBtn');
const videoContainer = document.querySelector('.video-container');
const container = document.querySelector('.container');

// Fullscreen mode flag
let isFullscreenMode = false;

// Model and detection variables
let detector;
let model;
let rafId;

// Thresholds for good form - Based on the Python trainer
const ELBOW_ANGLE_MIN = 85.0;  // Threshold for bent arms
const ELBOW_ANGLE_MAX = 160.0;  // Threshold for extended arms
const BACK_ANGLE_MIN = 143.6;  // Minimum back angle (more tilted forward)
const BACK_ANGLE_MAX = 170.0;  // Maximum back angle (more upright)
const ELBOW_POSITION_MINOR_THRESHOLD = 10;  // Minor elbow forward lean (acceptable)
const ELBOW_POSITION_MAJOR_THRESHOLD = 30;  // Major elbow forward lean (form issue)

// Elbow swinging detection thresholds
const ELBOW_STABILITY_THRESHOLD = 15.0;  // Maximum acceptable elbow movement
const BAD_ELBOW_STABILITY_THRESHOLD = 30.0;  // Threshold for major elbow swinging

// Variables for rep detection
let repState = 'up';  // Track state of the movement: 'up' = arms bent, 'down' = arms extended
const ELBOW_ANGLE_UP_THRESHOLD = ELBOW_ANGLE_MIN + 15;  // Angle below this is considered 'up' position
const ELBOW_ANGLE_DOWN_THRESHOLD = ELBOW_ANGLE_MAX - 30;  // Angle above this is considered 'down' position
let repCounted = false;  // Flag to prevent double counting
let counter = 0;  // Count of total reps
let correctReps = 0;  // Count of correctly performed reps
let incorrectReps = 0;  // Count of incorrectly performed reps
let lastRepTime = 0;  // Time when last rep was counted
const MIN_TIME_BETWEEN_REPS = 500;  // Minimum milliseconds between reps

// Get DOM elements for rep counters
const counterContainer = document.getElementById('counter-container');
const correctCounter = document.getElementById('correct-counter');
const incorrectCounter = document.getElementById('incorrect-counter');

// Exercise state management
let exerciseState = 'waiting';  // 'waiting' -> 'ready' -> 'counting'
let positionFrames = 0;  // Number of consecutive frames user has been in position
const POSITION_THRESHOLD = 30;  // Need this many frames in position to start counting
let countdownStarted = false;
let countdownStartTime = 0;
const COUNTDOWN_DURATION = 3;  // seconds

// Detection loss tracking
let detectionLostFrames = 0;
const MAX_DETECTION_LOST_FRAMES = 30;  // Reset trainer after this many frames with lost detection
let detectionFound = false;  // Flag to track if detection was found in the current frame

// Track form issues
let formIssues = {
    'elbow_not_extended': 0,
    'back_too_straight': 0,
    'back_too_tilted': 0,
    'elbow_leaning_forward': 0,
    'elbow_swinging': 0
};

// Form issues and rep tracking are defined earlier in the code

// Data collection for visualizations
let sessionData = {
    'frame_count': [],
    'elbow_angles': [],
    'back_angles': [],
    'elbow_vertical_angles': [],
    'elbow_x_positions': [],
    'rep_markers': [],
    'rep_elbow_angles': [],
    'rep_back_angles': [],
    'rep_elbow_vertical_angles': [],
    'rep_elbow_stability': []
};

// Store previous positions for tracking movement
let prevPositions = {
    'elbow_x': null,
    'elbow_y': null
};

// Add rolling window for elbow position tracking (for swinging detection)
let elbowPositionsWindow = [];  // Track last 30 frames
const ELBOW_WINDOW_SIZE = 30;
let elbowMovementWindow = [];   // Track last 20 movement calculations
const MOVEMENT_WINDOW_SIZE = 20;

// Add smoothing for landmark positions
let landmarkHistory = {};
const SMOOTHING_FACTOR = 0.5;  // Higher value = more smoothing (0.5 provides strong smoothing)

// Add smoothing for form feedback to prevent flickering
let feedbackHistory = new Map();
const FEEDBACK_PERSISTENCE = 20; // Increased - frames to keep feedback visible once triggered
const FEEDBACK_THRESHOLD = 8; // Number of consecutive frames needed to trigger a feedback message

// Feedback state tracking to prevent flickering
let feedbackStates = {
  'back_too_straight': { count: 0, active: false },
  'back_too_tilted': { count: 0, active: false },
  'elbow_swinging': { count: 0, active: false },
  'elbow_leaning_forward': { count: 0, active: false },
  'get_in_position': { count: 0, active: false }
};

// Color for keypoints (all red)
const keypointColors = Array(17).fill('#FF0000');

// Connection lines for the skeleton
const connectedKeypoints = [
  ['nose', 'left_eye'], ['nose', 'right_eye'], 
  ['left_eye', 'left_ear'], ['right_eye', 'right_ear'],
  ['left_shoulder', 'right_shoulder'], ['left_shoulder', 'left_hip'], 
  ['right_shoulder', 'right_hip'], ['left_hip', 'right_hip'],
  ['left_shoulder', 'left_elbow'], ['left_elbow', 'left_wrist'], 
  ['right_shoulder', 'right_elbow'], ['right_elbow', 'right_wrist'],
  ['left_hip', 'left_knee'], ['left_knee', 'left_ankle'], 
  ['right_hip', 'right_knee'], ['right_knee', 'right_ankle']
];

// Mapping from keypoint name to index
const keypointMap = {
  nose: 0,
  left_eye: 1,
  right_eye: 2,
  left_ear: 3,
  right_ear: 4,
  left_shoulder: 5,
  right_shoulder: 6,
  left_elbow: 7,
  right_elbow: 8,
  left_wrist: 9,
  right_wrist: 10,
  left_hip: 11,
  right_hip: 12,
  left_knee: 13,
  right_knee: 14,
  left_ankle: 15,
  right_ankle: 16
};

// Loading overlay removed

// Helper Functions

// Function to smooth landmark positions
function smoothLandmarks(keypoints) {
  const smoothedKeypoints = [...keypoints];
  
  for (let i = 0; i < keypoints.length; i++) {
    const keypoint = keypoints[i];
    const id = i;
    
    if (keypoint.score > 0.3) {  // Only smooth visible keypoints
      const position = { x: keypoint.x, y: keypoint.y };
      
      if (landmarkHistory[id]) {
        // Apply exponential smoothing
        const smoothedX = SMOOTHING_FACTOR * position.x + (1 - SMOOTHING_FACTOR) * landmarkHistory[id].x;
        const smoothedY = SMOOTHING_FACTOR * position.y + (1 - SMOOTHING_FACTOR) * landmarkHistory[id].y;
        
        smoothedKeypoints[i] = {
          ...keypoint,
          x: smoothedX,
          y: smoothedY
        };
        
        landmarkHistory[id] = { x: smoothedX, y: smoothedY };
      } else {
        // First time seeing this landmark
        landmarkHistory[id] = position;
      }
    }
  }
  
  return smoothedKeypoints;
}

// Function to calculate elbow stability using rolling window
function calculateElbowStability(elbowXRel) {
  // Add current position to window
  elbowPositionsWindow.push(elbowXRel);
  
  // Keep window at max size
  if (elbowPositionsWindow.length > ELBOW_WINDOW_SIZE) {
    elbowPositionsWindow.shift();
  }
  
  // Need at least 3 points to calculate meaningful movement
  if (elbowPositionsWindow.length < 3) {
    return 0;
  }
  
  // Calculate movement over the window, but filter outliers for stability
  // Sort positions to find outliers
  const sortedPositions = [...elbowPositionsWindow].sort((a, b) => a - b);
  
  // Remove the top and bottom 10% of values if we have enough samples
  let filteredPositions = sortedPositions;
  if (sortedPositions.length > 10) {
    const cutoff = Math.floor(sortedPositions.length * 0.1);
    filteredPositions = sortedPositions.slice(cutoff, sortedPositions.length - cutoff);
  }
  
  // Calculate range of movement using the filtered values
  const maxPos = Math.max(...filteredPositions);
  const minPos = Math.min(...filteredPositions);
  const rangeOfMovement = maxPos - minPos;
  
  // Add to movement window for consistent detection
  elbowMovementWindow.push(rangeOfMovement);
  
  // Keep window at max size
  if (elbowMovementWindow.length > MOVEMENT_WINDOW_SIZE) {
    elbowMovementWindow.shift();
  }
  
  // Return the average movement over the last several frames
  return elbowMovementWindow.reduce((sum, val) => sum + val, 0) / elbowMovementWindow.length;
}

// Function to draw an arrow
function drawArrow(ctx, startPoint, endPoint, color, thickness = 3) {
  ctx.beginPath();
  ctx.moveTo(startPoint.x, startPoint.y);
  ctx.lineTo(endPoint.x, endPoint.y);
  ctx.strokeStyle = color;
  ctx.lineWidth = thickness;
  ctx.stroke();
  
  // Calculate the angle of the line
  const angle = Math.atan2(endPoint.y - startPoint.y, endPoint.x - startPoint.x);
  
  // Draw the arrowhead
  const headLength = 15;
  ctx.beginPath();
  ctx.moveTo(endPoint.x, endPoint.y);
  ctx.lineTo(
    endPoint.x - headLength * Math.cos(angle - Math.PI / 6),
    endPoint.y - headLength * Math.sin(angle - Math.PI / 6)
  );
  ctx.lineTo(
    endPoint.x - headLength * Math.cos(angle + Math.PI / 6),
    endPoint.y - headLength * Math.sin(angle + Math.PI / 6)
  );
  ctx.lineTo(endPoint.x, endPoint.y);
  ctx.fillStyle = color;
  ctx.fill();
}

// Function to calculate angle between three points
function calculateAngle(a, b, c) {
  const ab = { x: b.x - a.x, y: b.y - a.y };
  const cb = { x: b.x - c.x, y: b.y - c.y };
  
  // Calculate dot product
  const dot = ab.x * cb.x + ab.y * cb.y;
  
  // Calculate magnitudes
  const magAB = Math.sqrt(ab.x * ab.x + ab.y * ab.y);
  const magCB = Math.sqrt(cb.x * cb.x + cb.y * cb.y);
  
  // Calculate angle in radians and convert to degrees
  const angleRad = Math.acos(dot / (magAB * magCB));
  const angleDeg = angleRad * (180 / Math.PI);
  
  return angleDeg;
}

// Smooth angle values with a moving average
let smoothedAngles = {
  elbow: [],
  back: [],
  elbowVertical: []
};
const ANGLE_SMOOTHING_WINDOW = 10;

// Function to smooth an angle value using moving average
function smoothAngle(newValue, angleType) {
  smoothedAngles[angleType].push(newValue);
  if (smoothedAngles[angleType].length > ANGLE_SMOOTHING_WINDOW) {
    smoothedAngles[angleType].shift();
  }
  
  // Calculate average, excluding outliers if we have enough samples
  if (smoothedAngles[angleType].length >= 4) {
    // Sort values to find potential outliers
    const sorted = [...smoothedAngles[angleType]].sort((a, b) => a - b);
    // Remove the smallest and largest value (potentially outliers)
    const trimmed = sorted.slice(1, -1);
    // Calculate average of remaining values
    return trimmed.reduce((sum, val) => sum + val, 0) / trimmed.length;
  } else {
    // If we don't have enough samples, use simple average
    return smoothedAngles[angleType].reduce((sum, val) => sum + val, 0) / smoothedAngles[angleType].length;
  }
}

// Add or update a feedback message with state tracking
function updateFeedbackState(type, condition, text, color) {
  // Update state count based on condition
  if (condition) {
    feedbackStates[type].count = Math.min(feedbackStates[type].count + 1, FEEDBACK_THRESHOLD + 5);
    
    // Activate feedback when threshold is reached
    if (feedbackStates[type].count >= FEEDBACK_THRESHOLD && !feedbackStates[type].active) {
      feedbackStates[type].active = true;
      // Add to feedback history when newly activated
      feedbackHistory.set(text, { color: color, count: FEEDBACK_PERSISTENCE });
    } else if (feedbackStates[type].active) {
      // Refresh persistence if already active
      feedbackHistory.set(text, { color: color, count: FEEDBACK_PERSISTENCE });
    }
  } else {
    // Decrease count when condition is false
    feedbackStates[type].count = Math.max(0, feedbackStates[type].count - 1);
    
    // Deactivate when count falls to zero
    if (feedbackStates[type].count === 0 && feedbackStates[type].active) {
      feedbackStates[type].active = false;
    }
  }
}

// Function to draw feedback at bottom of screen with enhanced persistence (optimized)
function drawFeedback(messages) {
  // Batch DOM updates using requestAnimationFrame for better performance
  requestAnimationFrame(() => {
    // Update existing messages in history - decrease persistence counters
    const expiredMessages = [];
    feedbackHistory.forEach((item, text) => {
      item.count--;
      if (item.count <= 0) {
        expiredMessages.push(text);
      }
    });
    
    // Remove expired messages
    expiredMessages.forEach(text => {
      feedbackHistory.delete(text);
    });
    
    // Hide feedback if no messages
    if (feedbackHistory.size === 0) {
      feedback.classList.add('hidden');
      return;
    }
    
    feedback.classList.remove('hidden');
    
    // Simplified message construction
    if (feedbackHistory.size > 0) {
      // Create a document fragment for better performance
      const fragment = document.createDocumentFragment();
      
      // Convert map to HTML elements
      feedbackHistory.forEach((item, text) => {
        // Simplified opacity calculation
        const opacity = Math.max(0.3, item.count / FEEDBACK_PERSISTENCE);
        
        // Simplified icon selection
        let icon = 'fa-info-circle';
        if (text.includes('Good')) icon = 'fa-check-circle';
        else if (text.includes('Back')) icon = 'fa-exclamation-triangle';
        else if (text.includes('Elbow')) icon = 'fa-hand-paper';
        
        // Create div directly instead of string concatenation
        const div = document.createElement('div');
        div.style.color = item.color;
        div.style.opacity = opacity;
        
        const iconEl = document.createElement('i');
        iconEl.className = `fas ${icon}`;
        div.appendChild(iconEl);
        
        div.appendChild(document.createTextNode(` ${text}`));
        fragment.appendChild(div);
      });
      
      // Clear and update in one operation
      feedback.innerHTML = '';
      feedback.appendChild(fragment);
    }
  });
}

// Function to get rep quality evaluation
function getRepQuality(elbowAngle, backAngle, elbowVerticalAngle = null) {
  const issues = [];
  
  // Check elbow extension
  if (elbowAngle < ELBOW_ANGLE_MAX) {
    issues.push('Elbows not fully extended');
  }
  
  // Check back position
  if (backAngle < BACK_ANGLE_MIN) {
    issues.push('Back too tilted');
  } else if (backAngle > BACK_ANGLE_MAX) {
    issues.push('Back too straight');
  }
  
  // Check elbow vertical position - only flag major tilting issues
  if (elbowVerticalAngle !== null && elbowVerticalAngle > ELBOW_POSITION_MAJOR_THRESHOLD) {
    issues.push('Elbow leaning too far forward');
  }
  
  if (issues.length === 0) {
    return 'Good form!';
  } else {
    return issues.join(', ');
  }
}

// Function to reset the trainer
function resetTrainer() {
  exerciseState = 'waiting';
  repState = 'up';
  positionFrames = 0;
  countdownStarted = false;
  repCounted = false;
  
  // Don't reset counter unless explicitly requested
  // counter = 0;
  // correctReps = 0;
  // incorrectReps = 0;
  // repCounter.textContent = `Total: ${counter}`;
  // correctCounter.textContent = `Good: ${correctReps}`;
  // incorrectCounter.textContent = `Bad: ${incorrectReps}`;
  
  // Hide overlays
  countdown.classList.add('hidden');
  positionBar.classList.add('hidden');
  counterContainer.classList.add('hidden');
  feedback.classList.add('hidden');
  
  console.log('Trainer reset');
}

// Initialize the detector
async function setupDetector() {
  // Load the MoveNet model
  const modelType = poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING;
  model = poseDetection.SupportedModels.MoveNet;
  const modelConfig = {
    modelType,
    enableSmoothing: true
  };
  
  detector = await poseDetection.createDetector(model, modelConfig);
  console.log('MoveNet model loaded successfully');
}

// Setup camera
async function setupCamera() {
  const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
  
  // Configure camera to match native phone camera app
  const constraints = {
    video: {
      facingMode: 'user',
      width: { ideal: 1920 },  // Higher resolution for better quality
      height: { ideal: 1080 },
      zoom: 1.0             // No digital zoom by default
    }
  };
  
  // On mobile, request the highest quality possible
  if (isMobile) {
    constraints.video.width = { ideal: 1920, min: 1280 };
    constraints.video.height = { ideal: 1080, min: 720 };
  }
  
  console.log('Camera constraints:', constraints);

  try {
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = stream;
    
    return new Promise((resolve) => {
      video.onloadedmetadata = () => {
        // Get the actual camera track settings
        const videoTrack = stream.getVideoTracks()[0];
        const settings = videoTrack.getSettings();
        console.log('Camera settings:', settings);
        
        // Reset any previous transformations
        video.style.transform = '';
        canvas.style.transform = '';
        video.style.objectFit = 'cover';
        canvas.style.objectFit = 'cover';
        
        // Set canvas dimensions to match video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        // Fix EXIF orientation issues on mobile devices
        document.body.classList.add('camera-initialized');
        
        // Log camera dimensions for debugging
        console.log(`Camera dimensions: ${canvas.width}x${canvas.height}`);
        console.log(`Video dimensions: ${video.videoWidth}x${video.videoHeight}`);
        
        resolve(video);
      };
    });
  } catch (error) {
    console.error('Error accessing webcam:', error);
    alert('Error accessing webcam. Please make sure you have a webcam connected and have granted permission.');
    throw error;
  }
}

// Detect poses in the current video frame
const TARGET_FPS = 30; // Reduced from 60
let lastDetectionTime = 0;

async function detectPose() {
  const now = performance.now();
  if (now - lastDetectionTime < 1000/TARGET_FPS) {
    rafId = requestAnimationFrame(detectPose);
    return;
  }
  lastDetectionTime = now;
  
  if (!detector || !video.readyState) {
    // Make sure the video and detector are ready
    video.onloadeddata = () => {
      rafId = requestAnimationFrame(detectPose);
    };
    return;
  }

  // Reset detection found flag for this frame
  detectionFound = false;
  
  // Check if video is ready
  if (video.readyState < 2) {
    await new Promise((resolve) => {
      video.onloadeddata = () => {
        resolve(video);
      };
    });
  }

  // Get poses from detector
  const poses = await detector.estimatePoses(video);
  
  // Clear the canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  // Draw video as background
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  
  // Process pose if detected
  if (poses && poses.length > 0) {
    const pose = poses[0]; // Only use the first detected person
    
    if (pose.keypoints && pose.keypoints.length > 0) {
      // Smooth the keypoints
      const smoothedKeypoints = smoothLandmarks(pose.keypoints);
      
      // Handle detection found
      processDetection(smoothedKeypoints);
      
      // Draw the pose
      drawPose({...pose, keypoints: smoothedKeypoints});
    }
  }
  
  // Handle detection loss
  if (!detectionFound) {
    handleDetectionLoss();
  }
  
  // Continue detection
  rafId = requestAnimationFrame(detectPose);
}

// Process detection and track exercise
function processDetection(keypoints) {
  // Update frame counter
  sessionData.frame_count.push(sessionData.frame_count.length);
  
  // Update detection flag
  detectionFound = true;
  detectionLostFrames = 0; // Reset lost frame counter
  
  // Check if we have the minimum required landmarks for tricep pushdown analysis
  const requiredLandmarks = [5, 6, 7, 8, 9, 10, 11, 12]; // Shoulders, elbows, wrists, hips
  // Also track leg landmarks (knees, ankles) separately with a lower threshold
  const legLandmarks = [13, 14, 15, 16]; // Left/right knees and ankles
  const visibleLandmarks = requiredLandmarks.filter(idx => keypoints[idx].score > 0.3);
  
  if (visibleLandmarks.length >= requiredLandmarks.length - 2) { // Allow some missing landmarks
    // Frame counter for visualization
    const frameCount = sessionData.frame_count.length;
    sessionData.frame_count.push(frameCount);
    
    try {
      // Determine which side (left or right) is more visible
      const leftVisibility = keypoints[5].score + keypoints[7].score + keypoints[9].score;
      const rightVisibility = keypoints[6].score + keypoints[8].score + keypoints[10].score;
      
      // Use the side with higher visibility and update global variable
      useRightSide = rightVisibility >= leftVisibility;
      
      // Get reference points based on visibility
      let shoulder, elbow, wrist, hip, knee, ankle;
      
      if (useRightSide) {
        // Right side landmarks
        shoulder = keypoints[6]; // right_shoulder
        elbow = keypoints[8];    // right_elbow
        wrist = keypoints[10];   // right_wrist
        hip = keypoints[12];     // right_hip
        knee = keypoints[14];    // right_knee
        ankle = keypoints[16];   // right_ankle
      } else {
        // Left side landmarks
        shoulder = keypoints[5]; // left_shoulder
        elbow = keypoints[7];    // left_elbow
        wrist = keypoints[9];    // left_wrist
        hip = keypoints[11];     // left_hip
        knee = keypoints[13];    // left_knee
        ankle = keypoints[15];   // left_ankle
      }
      
      // Calculate angles
      // Elbow angle (shoulder-elbow-wrist)
      const rawElbowAngle = calculateAngle(
        { x: shoulder.x, y: shoulder.y },
        { x: elbow.x, y: elbow.y },
        { x: wrist.x, y: wrist.y }
      );
      const elbowAngle = smoothAngle(rawElbowAngle, 'elbow');
      
      // Back angle if we have hip and ankle data
      let rawBackAngle = 160; // Default if we can't calculate
      if (hip.score > 0.3 && ankle.score > 0.3) {
        rawBackAngle = calculateAngle(
          { x: shoulder.x, y: shoulder.y },
          { x: hip.x, y: hip.y },
          { x: ankle.x, y: ankle.y }
        );
      }
      const backAngle = smoothAngle(rawBackAngle, 'back');
      
      // Calculate elbow verticality (angle from vertical)
      // Create a vertical line from elbow
      const verticalPoint = { x: elbow.x, y: elbow.y - 100 }; // 100 pixels directly above elbow
      
      // Calculate the angle (shoulder-elbow-vertical)
      let rawElbowVerticalAngle = calculateAngle(
        { x: shoulder.x, y: shoulder.y },
        { x: elbow.x, y: elbow.y },
        verticalPoint
      );
      
      // Determine if the elbow is leaning forward or backward
      if ((useRightSide && shoulder.x < elbow.x) || (!useRightSide && shoulder.x > elbow.x)) {
        rawElbowVerticalAngle = 180 - rawElbowVerticalAngle;
      }
      
      // Apply smoothing to elbow vertical angle
      const elbowVerticalAngle = smoothAngle(rawElbowVerticalAngle, 'elbowVertical');
      
      // Calculate elbow swinging (horizontal movement)
      // Get the current elbow position relative to shoulder
      const elbowXRel = elbow.x - shoulder.x;
      
      // Track the relative horizontal movement of the elbow
      const elbowXMovement = calculateElbowStability(elbowXRel);
      
      // Store current positions for next frame comparison
      prevPositions.elbow_x = elbowXRel;
      prevPositions.elbow_y = elbow.y;
      
      // Store data for visualization
      sessionData.elbow_angles.push(elbowAngle);
      sessionData.back_angles.push(backAngle);
      sessionData.elbow_vertical_angles.push(elbowVerticalAngle);
      sessionData.elbow_x_positions.push(elbowXMovement);
      
      // Draw the analyzed arm and back with thicker lines for emphasis
      ctx.strokeStyle = '#00FF00'; // Green for arms
      ctx.lineWidth = 5;
      ctx.beginPath();
      ctx.moveTo(shoulder.x, shoulder.y);
      ctx.lineTo(elbow.x, elbow.y);
      ctx.lineTo(wrist.x, wrist.y);
      ctx.stroke();
      
      ctx.strokeStyle = '#FFC800'; // Orange for back
      if (hip.score > 0.3) {
        ctx.beginPath();
        ctx.moveTo(shoulder.x, shoulder.y);
        ctx.lineTo(hip.x, hip.y);
        ctx.stroke();
        
        // Lower threshold for legs to improve visibility (0.2 instead of 0.3)
        if (knee.score > 0.2 && ankle.score > 0.2) {
          ctx.beginPath();
          ctx.moveTo(hip.x, hip.y);
          ctx.lineTo(knee.x, knee.y);
          ctx.lineTo(ankle.x, ankle.y);
          ctx.stroke();
        }
      }
      
      // Initialize feedback messages list
      const feedbackMessages = [];
      
      // Exercise state management based on the Python implementation
      handleExerciseState(elbowAngle, backAngle, elbowVerticalAngle, elbowXMovement, feedbackMessages, shoulder, elbow, hip);
      
      // Draw all feedback messages
      drawFeedback(feedbackMessages);
      
    } catch (error) {
      console.error('Error processing pose:', error);
    }
  }
}

// Handle the exercise state machine
function handleExerciseState(elbowAngle, backAngle, elbowVerticalAngle, elbowXMovement, feedbackMessages, shoulder, elbow, hip) {
  if (exerciseState === 'waiting') {
    // Check if user is in starting position (arms bent)
    if (elbowAngle < ELBOW_ANGLE_UP_THRESHOLD + 20) { // More lenient threshold for starting position
      positionFrames++;
      
      // Add visual indicator that position is being detected
      positionBar.classList.remove('hidden');
      const progress = Math.min(1.0, positionFrames / POSITION_THRESHOLD);
      positionProgress.style.width = `${progress * 100}%`;
      
      if (positionFrames >= POSITION_THRESHOLD && !countdownStarted) {
        // User has been in position for enough frames, start countdown
        countdownStarted = true;
        countdownStartTime = Date.now();
        countdown.classList.remove('hidden');
      }
    } else {
      // Reset if user moves out of position
      positionFrames = Math.max(0, positionFrames - 2); // Decrease more slowly to be forgiving
      const progress = Math.min(1.0, positionFrames / POSITION_THRESHOLD);
      positionProgress.style.width = `${progress * 100}%`;
      countdownStarted = false;
      countdown.classList.add('hidden');
    }
    
    // Handle countdown display
    if (countdownStarted) {
      const elapsed = (Date.now() - countdownStartTime) / 1000;
      const secondsLeft = Math.max(0, Math.ceil(COUNTDOWN_DURATION - elapsed));
      
      if (secondsLeft > 0) {
        // Show countdown
        countdown.textContent = secondsLeft.toString();
      } else {
        // Countdown finished, start counting reps
        exerciseState = 'counting';
        countdown.classList.add('hidden');
        positionBar.classList.add('hidden');
        counterContainer.classList.remove('hidden');
        repCounter.textContent = `Total: ${counter}`;
        correctCounter.textContent = `Good: ${correctReps}`;
        incorrectCounter.textContent = `Bad: ${incorrectReps}`;
        console.log('Starting to count reps!');
      }
    } else {
      // Update 'Get in position' state
      updateFeedbackState('get_in_position', true, 'Get in position (arms bent)', '#FFFF00');
    }
  }
  
  // Handle form feedback and rep counting when in counting state
  if (exerciseState === 'counting') {
    // Apply state-based feedback instead of direct frame-by-frame feedback
    // This reduces flickering by requiring consistent detection before showing feedback
    
    // Check back angle and update state
    updateFeedbackState('back_too_straight', backAngle > BACK_ANGLE_MAX, 'Bend your back slightly forward', '#FF0000');
    updateFeedbackState('back_too_tilted', backAngle < BACK_ANGLE_MIN, 'Straighten your back a bit', '#FF0000');
    
    // Only draw feedback arrows when the state is active
    if (feedbackStates.back_too_straight.active && hip.score > 0.3) {
      // Back is too straight - draw downward arrow showing need to tilt forward
      const midback = { 
        x: (shoulder.x + hip.x) / 2, 
        y: (shoulder.y + hip.y) / 2 
      };
      // Point downward for a more intuitive 'tilt forward' gesture
      const downwardPoint = { 
        x: midback.x, 
        y: midback.y + 60 
      };
      // Draw arrow from middle of back pointing downward
      drawArrow(ctx, midback, downwardPoint, '#FF0000', 4);
      
      // Add 'tilt forward' text next to the arrow
      ctx.font = '16px Arial';
      ctx.fillStyle = '#FF0000';
      ctx.fillText('Tilt forward', midback.x + 10, midback.y + 30);
      
      formIssues.back_too_straight++;
    } else if (feedbackStates.back_too_tilted.active && hip.score > 0.3) {
      // Back is too tilted - draw upward arrow showing need to straighten up
      const midback = { 
        x: (shoulder.x + hip.x) / 2, 
        y: (shoulder.y + hip.y) / 2 
      };
      // Point upward for a more intuitive 'straighten up' gesture
      const upwardPoint = { 
        x: midback.x, 
        y: midback.y - 60 
      };
      // Draw arrow from middle of back pointing upward
      drawArrow(ctx, midback, upwardPoint, '#FF0000', 4);
      
      // Add 'straighten up' text next to the arrow
      ctx.font = '16px Arial';
      ctx.fillStyle = '#FF0000';
      ctx.fillText('Straighten up', midback.x + 10, midback.y - 20);
      
      formIssues.back_too_tilted++;
    }
    
    // Update elbow swinging state
    updateFeedbackState('elbow_swinging', 
      elbowXMovement > BAD_ELBOW_STABILITY_THRESHOLD, 
      'Stop swinging elbows!', 
      '#FF0000'
    );
    
    // Only show feedback and arrow when state is active
    if (feedbackStates.elbow_swinging.active) {
      // Major swinging - red arrow
      const sidePoint = { 
        x: elbow.x - 50, 
        y: elbow.y 
      };
      drawArrow(ctx, elbow, sidePoint, '#FF0000', 4);
      formIssues.elbow_swinging++;
    } else if (elbowXMovement > ELBOW_STABILITY_THRESHOLD) {
      // Minor swinging - track but don't show feedback
      formIssues.elbow_swinging++;
    }
    
    // Rep counting
    const currentTime = Date.now();
    const timeSinceLastRep = currentTime - lastRepTime;
    
    // State machine for rep counting
    if (repState === 'up' && elbowAngle > ELBOW_ANGLE_DOWN_THRESHOLD) {
      repState = 'down';
      console.log(`State change: up -> down (angle: ${elbowAngle.toFixed(1)})`);
      
      // Only count the rep once when transitioning from up to down
      if (!repCounted && timeSinceLastRep > MIN_TIME_BETWEEN_REPS) {
        repCounted = true;
        counter++;
        lastRepTime = currentTime;
        
        // Check if this was a correct rep (proper back angle and minimal elbow swinging)
        const isBackCorrect = backAngle >= BACK_ANGLE_MIN && backAngle <= BACK_ANGLE_MAX;
        const isElbowStable = elbowXMovement <= ELBOW_STABILITY_THRESHOLD;
        const isCorrectRep = isBackCorrect && isElbowStable;
        
        // Update the appropriate counter
        if (isCorrectRep) {
          correctReps++;
          correctCounter.textContent = `Good: ${correctReps}`;
          correctCounter.style.color = '#4CAF50';  // Highlight with brighter green
          setTimeout(() => { correctCounter.style.color = ''; }, 500);  // Reset after highlighting
        } else {
          incorrectReps++;
          incorrectCounter.textContent = `Bad: ${incorrectReps}`;
          incorrectCounter.style.color = '#FF5252';  // Highlight with brighter red
          setTimeout(() => { incorrectCounter.style.color = ''; }, 500);  // Reset after highlighting
          
          // Explain why it was incorrect
          if (!isBackCorrect) {
            if (backAngle > BACK_ANGLE_MAX) {
              feedbackHistory.set('Back too straight on last rep', { color: '#FFA500', count: FEEDBACK_PERSISTENCE });
            } else {
              feedbackHistory.set('Back too tilted on last rep', { color: '#FFA500', count: FEEDBACK_PERSISTENCE });
            }
          }
          if (!isElbowStable) {
            feedbackHistory.set('Too much elbow swinging on last rep', { color: '#FFA500', count: FEEDBACK_PERSISTENCE });
          }
        }
        
        // Update all counters
        repCounter.textContent = `Total: ${counter}`;
        console.log(`Rep ${counter} detected! Correct: ${correctReps}, Incorrect: ${incorrectReps}`);
        
        // Check if 12 repetitions have been completed
        if (counter >= 12) {
          // Generate text file with exercise results
          generateResultsFile();
          
          // Display completion message in the middle of the camera view
          const completionMessage = document.createElement('div');
          completionMessage.className = 'completion-message';
          completionMessage.innerHTML = '<h2>Exercise Complete!</h2><p>Results have been saved to a text file.</p>';
          document.querySelector('.video-container').appendChild(completionMessage);
          
          // Remove the completion message after 5 seconds
          setTimeout(() => {
            if (completionMessage.parentNode) {
              completionMessage.parentNode.removeChild(completionMessage);
            }
          }, 5000);
          
          // Stop the camera and AI after a short delay
          setTimeout(() => {
            // Stop the camera
            if (video.srcObject) {
              video.srcObject.getTracks().forEach(track => {
                track.stop();
              });
            }
            
            // Stop the animation frame
            if (rafId) {
              cancelAnimationFrame(rafId);
            }
            
            // Reset button states
            startBtn.disabled = false;
            startBtn.textContent = 'Start Camera';
          }, 5000);
        }
        
        // Store rep data
        sessionData.rep_markers.push(frameCount);
        sessionData.rep_elbow_angles.push(elbowAngle);
        sessionData.rep_back_angles.push(backAngle);
        sessionData.rep_elbow_vertical_angles.push(elbowVerticalAngle);
        sessionData.rep_elbow_stability.push(elbowXMovement);
        
        // Check for elbow swinging on this rep
        if (elbowXMovement > ELBOW_STABILITY_THRESHOLD) {
          formIssues.elbow_swinging++;
        }
      }
    } 
    // Detect state change from arms extended back to arms bent
    else if (repState === 'down' && elbowAngle < ELBOW_ANGLE_UP_THRESHOLD) {
      repState = 'up';
      repCounted = false; // Reset the rep counted flag
      console.log(`State change: down -> up (angle: ${elbowAngle.toFixed(1)})`);
    }
  }
}

// Handle detection loss with hysteresis to prevent flickering
let lastDetectionState = true; // Start assuming detection is present
let stableDetectionLostFrames = 0;

function handleDetectionLoss() {
  detectionLostFrames++;
  
  // Never show detection warning
  detectionWarning.classList.add('hidden');
  
  // Keep track of detectionLostFrames for other functionality
  if (detectionLostFrames > 10) {
    stableDetectionLostFrames++;
  } else {
    stableDetectionLostFrames = 0;
  }
  
  // Reset the trainer if detection is lost for too long
  if (detectionLostFrames >= MAX_DETECTION_LOST_FRAMES) {
    console.log('Detection lost for too long, resetting trainer...');
    resetTrainer();
  }
}

// Track which side is more visible
let useRightSide = true; // Default to right side

// Update the keypointColors for lower body to use light red
keypointColors[11] = '#ff6b6b'; // left_hip (light red)
keypointColors[12] = '#ff6b6b'; // right_hip (light red)
keypointColors[13] = '#ff6b6b'; // left_knee (light red)
keypointColors[14] = '#ff6b6b'; // right_knee (light red)
keypointColors[15] = '#ff6b6b'; // left_ankle (light red)
keypointColors[16] = '#ff6b6b'; // right_ankle (light red)

// Draw the pose skeleton and keypoints (optimized version)
function drawPose(pose) {
  if (!pose || !pose.keypoints) return;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  // Draw pose using TensorFlow.js pose detection
  const keypoints = pose.keypoints;
  
  // Determine which side is more visible (faster calculation)
  const leftVisibility = keypoints[5].score + keypoints[7].score + keypoints[9].score + keypoints[11].score;
  const rightVisibility = keypoints[6].score + keypoints[8].score + keypoints[10].score + keypoints[12].score;
  useRightSide = rightVisibility >= leftVisibility;
  
  // Draw keypoints with simplified rendering
  for (let i = 0; i < keypoints.length; i++) {
    const keypoint = keypoints[i];
    
    // Only draw keypoints with high confidence
    // Use a lower threshold for leg keypoints (indices 13-16 are knees and ankles)
    const isLegKeypoint = (i >= 13 && i <= 16);
    const confidenceThreshold = isLegKeypoint ? 0.2 : 0.3;
    
    if (keypoint.score > confidenceThreshold) {
      ctx.fillStyle = keypointColors[i];
      // Use rectangle instead of arc for better performance
      // Make leg keypoints slightly larger for better visibility
      const pointSize = isLegKeypoint ? 5 : 4;
      ctx.fillRect(keypoint.x - pointSize/2, keypoint.y - pointSize/2, pointSize, pointSize);
    }
  }
  
  // Simplified skeleton drawing
  ctx.lineWidth = 4;
  
  // Draw full body connections including lower body
  const bodyConnections = [
    // Upper body
    ['left_shoulder', 'right_shoulder'],
    ['left_shoulder', 'left_elbow'],
    ['right_shoulder', 'right_elbow'],
    ['left_elbow', 'left_wrist'],
    ['right_elbow', 'right_wrist'],
    // Torso
    ['left_shoulder', 'left_hip'],
    ['right_shoulder', 'right_hip'],
    ['left_hip', 'right_hip'],
    // Lower body
    ['left_hip', 'left_knee'],
    ['right_hip', 'right_knee'],
    ['left_knee', 'left_ankle'],
    ['right_knee', 'right_ankle']
  ];
  
  bodyConnections.forEach(([keypointA, keypointB]) => {
    const indexA = keypointMap[keypointA];
    const indexB = keypointMap[keypointB];
    
    const keypointA_obj = keypoints[indexA];
    const keypointB_obj = keypoints[indexB];
    
    // Check if this is a leg connection
    const isLegConnection = keypointA.includes('knee') || keypointB.includes('knee') || 
                         keypointA.includes('ankle') || keypointB.includes('ankle');
    // Use a lower threshold for leg connections
    const confidenceThreshold = isLegConnection ? 0.2 : 0.3;
    
    // Only draw if both keypoints have sufficient confidence
    if (keypointA_obj.score > confidenceThreshold && keypointB_obj.score > confidenceThreshold) {
      // Simplified side detection
      const isRightSide = keypointA.includes('right') || keypointB.includes('right');
      const isLeftSide = keypointA.includes('left') || keypointB.includes('left');
      const isLowerBody = keypointA.includes('knee') || keypointB.includes('knee') || 
                        keypointA.includes('ankle') || keypointB.includes('ankle') || 
                        (keypointA.includes('hip') && keypointB.includes('hip'));
      
      // Set color based on which side is more visible and body part
      if (isLowerBody) {
        // Light red for lower body
        ctx.strokeStyle = '#ff6b6b';
      } else if ((useRightSide && isRightSide && !isLeftSide) || 
          (!useRightSide && isLeftSide && !isRightSide)) {
        ctx.strokeStyle = '#00FF00'; // Green for more visible upper body side
      } else {
        ctx.strokeStyle = '#FFFFFF'; // White for less visible or central connections
      }
      
      // Direct line drawing without begin/end path for performance
      ctx.beginPath();
      ctx.moveTo(keypointA_obj.x, keypointA_obj.y);
      ctx.lineTo(keypointB_obj.x, keypointB_obj.y);
      ctx.stroke();
    }
  });
}

// Function to enter fullscreen mode for mobile
function enterFullscreenMode() {
  if (isFullscreenMode) return;
  
  document.body.classList.add('fullscreen-mode');
  exitFullscreenBtn.classList.remove('hidden');
  isFullscreenMode = true;

  // For iOS Safari - try to enter real fullscreen if possible
  if (videoContainer.requestFullscreen) {
    videoContainer.requestFullscreen().catch(err => {
      console.log('Error attempting to enable fullscreen:', err);
      // Continue with our custom fullscreen even if native fails
    });
  }
}

// Function to exit fullscreen mode
function exitFullscreenMode() {
  if (!isFullscreenMode) return;
  
  document.body.classList.remove('fullscreen-mode');
  document.body.classList.remove('camera-active');
  exitFullscreenBtn.classList.add('hidden');
  isFullscreenMode = false;

  // Exit real fullscreen if needed
  if (document.fullscreenElement) {
    document.exitFullscreen().catch(err => {
      console.log('Error attempting to exit fullscreen:', err);
    });
  }
  
  // Stop the camera stream
  if (video.srcObject) {
    video.srcObject.getTracks().forEach(track => {
      track.stop();
    });
    video.srcObject = null;
  }
  
  // Reset UI state
  startBtn.disabled = false;
  startBtn.textContent = 'Start Camera';
}

// Start the application
async function startApp() {
  try {
    startBtn.disabled = true;
    startBtn.textContent = 'Loading...';
    
    // Load the pose detection model if not already loaded
    if (!detector) {
      await setupDetector();
    }
    
    // Setup the camera if not already done
    if (!video.srcObject) {
      await setupCamera();
    }
    
    // Show camera and enter fullscreen mode for mobile
    document.body.classList.add('camera-active');
    enterFullscreenMode();
    
    // Start pose detection
    video.play();
    detectPose();
    
    // Update button states
    startBtn.textContent = 'Running';
    resetBtn.disabled = false;
    
    // Reset trainer to initial state
    resetTrainer();
    
    console.log('Tricep Pushdown Trainer started successfully');
  } catch (error) {
    console.error('Error starting the application:', error);
    startBtn.disabled = false;
    startBtn.textContent = 'Start Camera';
    alert('Error starting the application. Please make sure you have a webcam connected and have granted permission.');
  }
}

// Reset the training session
function resetTrainingSession() {
    counter = 0;
    correctReps = 0;
    incorrectReps = 0;
    repCounter.textContent = `Total: ${counter}`;
    correctCounter.textContent = `Good: ${correctReps}`;
    incorrectCounter.textContent = `Bad: ${incorrectReps}`;
    resetTrainer();
}

// Add event listeners
startBtn.addEventListener('click', startApp);
resetBtn.addEventListener('click', resetTrainingSession);
exitFullscreenBtn.addEventListener('click', exitFullscreenMode);

// Calculate performance score based on rep quality and form issues
function calculatePerformanceScore() {
  // Base the score on the ratio of correct reps to total reps
  let repScore = counter > 0 ? (correctReps / counter) * 100 : 0;
  
  // Calculate a form deduction based on the number of form issues
  const totalFormIssues = Object.values(formIssues).reduce((sum, count) => sum + count, 0);
  const formIssuesPenalty = totalFormIssues > 0 ? Math.min(25, totalFormIssues) : 0;
  
  // Final score calculation (max 100%)
  let finalScore = Math.max(0, Math.min(100, repScore - formIssuesPenalty));
  
  // Round to nearest whole number
  return Math.round(finalScore);
}

// Create a results object with all session data
function createResultsObject() {
  const performanceScore = calculatePerformanceScore();
  const currentDate = new Date();
  
  return {
    workoutName: "Tricep Pushdown",
    date: currentDate.toLocaleDateString(),
    time: currentDate.toLocaleTimeString(),
    sets: 1, // Default to 1 set in this implementation
    reps: {
      total: counter,
      correct: correctReps,
      incorrect: incorrectReps
    },
    formIssues: {
      elbowNotExtended: formIssues.elbow_not_extended,
      backTooStraight: formIssues.back_too_straight,
      backTooTilted: formIssues.back_too_tilted,
      elbowLeaningForward: formIssues.elbow_leaning_forward,
      elbowSwinging: formIssues.elbow_swinging
    },
    performanceScore: performanceScore,
    sessionId: Date.now().toString(36) + Math.random().toString(36).substr(2)
  };
}

// Store session data in localStorage
function saveSessionData(resultsObject) {
  try {
    // Get existing session history or initialize new array
    let sessionHistory = JSON.parse(localStorage.getItem('exerciseSessionHistory')) || [];
    
    // Add new session data
    sessionHistory.push(resultsObject);
    
    // Save back to localStorage (limit to last 20 sessions)
    if (sessionHistory.length > 20) {
      sessionHistory = sessionHistory.slice(-20);
    }
    
    localStorage.setItem('exerciseSessionHistory', JSON.stringify(sessionHistory));
    localStorage.setItem('lastExerciseSession', JSON.stringify(resultsObject));
    
    console.log('Session data saved successfully');
    return true;
  } catch (error) {
    console.error('Error saving session data:', error);
    return false;
  }
}

// Flag to prevent multiple downloads
let downloadInProgress = false;

// Function to generate and download results file
function generateResultsFile() {
  // Prevent multiple downloads
  if (downloadInProgress) {
    console.log('Download already in progress');
    return;
  }
  
  downloadInProgress = true;
  console.log('Generating results file...');
  
  // Get complete results data
  const resultsObject = createResultsObject();
  
  // Save to localStorage
  saveSessionData(resultsObject);
  
  // Create formatted content for the text file
  const content = `Exercise Results:
------------------------
Workout: ${resultsObject.workoutName}
Date: ${resultsObject.date} at ${resultsObject.time}
Sets: ${resultsObject.sets}
Total Repetitions: ${resultsObject.reps.total}
Correct Repetitions: ${resultsObject.reps.correct}
Incorrect Repetitions: ${resultsObject.reps.incorrect}

Performance Score: ${resultsObject.performanceScore}%

Form Issues:
------------------------
Elbow Not Extended: ${resultsObject.formIssues.elbowNotExtended} times
Back Too Straight: ${resultsObject.formIssues.backTooStraight} times
Back Too Tilted: ${resultsObject.formIssues.backTooTilted} times
Elbow Leaning Forward: ${resultsObject.formIssues.elbowLeaningForward} times
Elbow Swinging: ${resultsObject.formIssues.elbowSwinging} times`;
  
  try {
    // Create a Blob with the JSON data
    const jsonData = JSON.stringify(resultsObject, null, 2); // Pretty print with 2 spaces
    const blob = new Blob([jsonData], { type: 'application/json' });
    
    // Create a download link and trigger the download
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `exercise_results_${resultsObject.sessionId}.json`;
    a.style.display = 'none';
    document.body.appendChild(a);
    
    // Force the download
    setTimeout(() => {
      console.log('Triggering download...');
      a.click();
      
      // Clean up
      setTimeout(() => {
        document.body.removeChild(a);
        URL.revokeObjectURL(a.href);
        console.log('Download complete');
        downloadInProgress = false;
      }, 1000);
    }, 500);
  } catch (error) {
    console.error('Error generating results file:', error);
    alert('Error generating results file. Please check the console for details.');
    downloadInProgress = false;
  }
}

// Clean up on page unload
window.addEventListener('beforeunload', () => {
  if (rafId) {
    cancelAnimationFrame(rafId);
  }
  
  if (video.srcObject) {
    video.srcObject.getTracks().forEach(track => {
      track.stop();
    });
  }
});

// Log initial state
console.log('Tricep Pushdown Trainer initialized. Press Start Camera to begin.');
