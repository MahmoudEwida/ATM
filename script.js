const video = document.getElementById('webcam');
const canvas = document.getElementById('output');
const ctx = canvas.getContext('2d');
const startBtn = document.getElementById('startBtn');
const resetBtn = document.getElementById('resetBtn');
const detectionWarning = document.getElementById('detection-warning');
const repCounter = document.getElementById('rep-counter');
const badRepsCounter = document.getElementById('bad-reps-counter');
const feedback = document.getElementById('feedback');
const exitFullscreenBtn = document.getElementById('exitFullscreenBtn');
const videoContainer = document.querySelector('.video-container');
const counterContainer = document.getElementById('counter-container');
const mainBackBtn = document.getElementById('mainBackBtn');

// Thresholds based on Python implementation
const ELBOW_ANGLE_MIN = 70.0;  // Minimum elbow angle (fully curled)
const ELBOW_ANGLE_MAX = 160.0;  // Maximum elbow angle (fully extended)
const ELBOW_POSITION_MAJOR_THRESHOLD = 30.0;
const ELBOW_STABILITY_THRESHOLD = 25.0;  // Increased from 15.0 to be more lenient
const BAD_ELBOW_STABILITY_THRESHOLD = 45.0;  // Increased from 35.0 to be more lenient
const ELBOW_ANGLE_UP_THRESHOLD = ELBOW_ANGLE_MIN + 10;  // Angle below this is considered "up" position
const ELBOW_ANGLE_DOWN_THRESHOLD = ELBOW_ANGLE_MAX - 20;  // Angle above this is considered "down" position

// State management
let exerciseState = 'waiting';
let repState = 'down';  // 'down' means arms extended, 'up' means arms curled
let repCounted = false;
let counter = 0;
let badReps = {
    total: 0,
    elbow_swinging: 0
};
let currentRepHasError = false;
let positionFrames = 0;
const POSITION_THRESHOLD = 30;
let countdownStarted = false;
let countdownStartTime = 0;
const COUNTDOWN_DURATION = 3000;
let lastRepTime = 0;
const MIN_TIME_BETWEEN_REPS = 500;
let detectionLostFrames = 0;
const MAX_DETECTION_LOST_FRAMES = 30;
let detectionFound = false;
let isFullscreenMode = false; // Added to fix undefined error

// Smoothing and tracking
let landmarkHistory = {};
const SMOOTHING_FACTOR = 0.5;  // Higher = more smoothing
let prevPositions = { elbow_x: null, elbow_y: null };
const elbowPositionsWindow = [];
const elbowMovementWindow = [];
const ELBOW_POS_WINDOW_SIZE = 15;  // Track last 15 frames
const MOVEMENT_WINDOW_SIZE = 10;  // Track last 10 frames of movement
const MIN_CONSECUTIVE_FRAMES = 10;
const MIN_MOVEMENT_THRESHOLD = 25;

// Feedback persistence
let feedbackHistory = new Map();
const FEEDBACK_PERSISTENCE = 60;  // Increased from 10 to 60 frames for longer display

// Session data
let sessionData = {
    frame_count: [],
    elbow_angles: [],
    elbow_x_positions: [],
    rep_markers: [],
    rep_elbow_angles: [],
    rep_elbow_stability: [],
    rep_feedback: [],
    saved: false
};

// Keypoint mappings
const keypointMap = {
    nose: 0, left_eye: 1, right_eye: 2, left_ear: 3, right_ear: 4,
    left_shoulder: 5, right_shoulder: 6, left_elbow: 7, right_elbow: 8,
    left_wrist: 9, right_wrist: 10, left_hip: 11, right_hip: 12,
    left_knee: 13, right_knee: 14, left_ankle: 15, right_ankle: 16
};
const keypointColors = Array(17).fill('#FF0000');
const connectedKeypoints = [
    ['left_shoulder', 'right_shoulder'],
    ['left_shoulder', 'left_elbow'], ['left_elbow', 'left_wrist'],
    ['right_shoulder', 'right_elbow'], ['right_elbow', 'right_wrist'],
    ['left_shoulder', 'left_hip'], ['left_hip', 'left_knee'], ['left_knee', 'left_ankle'],
    ['right_shoulder', 'right_hip'], ['right_hip', 'right_knee'], ['right_knee', 'right_ankle']
];

// Change the MAX_REPS_PER_SESSION constant
const MAX_REPS_PER_SESSION = 10;

// ====================== CUSTOMIZATION INSTRUCTIONS ======================
// To change where the BACK button leads:
// 1. Replace the URL below with your desired destination
// 2. Examples:
//    - For a local page: '../your-page.html'
//    - For the squat trainer: '../squat final/index.html'
//    - For an external website: 'https://your-website.com'
// ====================================================================
const previousPageUrl = '';

if (mainBackBtn) {
  mainBackBtn.addEventListener('click', () => {
    if (video?.srcObject) {
      video.srcObject.getTracks().forEach((track) => track.stop());
    }
    window.location.href = '/workout';
  });
}

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
    const angleRad = Math.acos(Math.min(1, Math.max(-1, dot / (magAB * magCB))));
    return angleRad * (180 / Math.PI);
}

function calculateElbowStability(elbowXRel) {
    // Add current position to window
    elbowPositionsWindow.push(elbowXRel);
    
    if (elbowPositionsWindow.length > ELBOW_POS_WINDOW_SIZE) {
        elbowPositionsWindow.shift();
    }
    
    if (elbowPositionsWindow.length < 3) {
        return 0;
    }

    // Calculate moving average with more weight on recent positions
    const movingAverage = elbowPositionsWindow.reduce((sum, val) => sum + val, 0) / elbowPositionsWindow.length;
    
    // Calculate variance with more weight on recent movements
    let weightedVariance = 0;
    let totalWeight = 0;
    
    elbowPositionsWindow.forEach((pos, index) => {
        const weight = Math.pow(index + 1, 1.5);  // Reduced weight power from 2 to 1.5 for smoother detection
        const diff = pos - movingAverage;
        weightedVariance += (diff * diff) * weight;
        totalWeight += weight;
    });
    
    weightedVariance = weightedVariance / totalWeight;
    
    // Increase minimum variance threshold to ignore small movements
    if (weightedVariance < 35) {  // Increased from 25 to reduce sensitivity to small movements
        return 0;
    }
    
    // Calculate range of movement
    const maxPos = Math.max(...elbowPositionsWindow);
    const minPos = Math.min(...elbowPositionsWindow);
    const range = maxPos - minPos;
    
    // Increase minimum movement threshold
    if (range < 30) {  // Increased from MIN_MOVEMENT_THRESHOLD to be more lenient
        return 0;
    }
    
    // Add to movement window
    elbowMovementWindow.push(range);
    if (elbowMovementWindow.length > MOVEMENT_WINDOW_SIZE) {
        elbowMovementWindow.shift();
    }
    
    // Calculate weighted average with more bias towards stability
    let weightedSum = 0;
    let weightSum = 0;
    
    elbowMovementWindow.forEach((value, index) => {
        const weight = Math.pow(index + 1, 1.5);  // Reduced weight power for smoother detection
        weightedSum += value * weight;
        weightSum += weight;
    });
    
    const weightedAverage = weightedSum / weightSum;
    
    // Apply stronger dampening
    return weightedAverage * 0.5;  // Increased dampening from 0.6 to 0.5 (50% reduction)
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

function drawFeedback() {
    // Remove expired messages
    const expiredMessages = [];
    feedbackHistory.forEach((item, text) => {
        item.count--;
        if (item.count <= 0) expiredMessages.push(text);
    });
    expiredMessages.forEach(text => feedbackHistory.delete(text));

    // If no messages, hide feedback area
    if (feedbackHistory.size === 0) {
        feedback.classList.add('hidden');
        feedback.innerHTML = '';
        return;
    }

    // Show feedback area and create messages
    feedback.classList.remove('hidden');
    feedback.innerHTML = '';  // Clear existing messages

    feedbackHistory.forEach((item, text) => {
        const div = document.createElement('div');
        div.style.color = item.color;
        div.style.opacity = Math.max(0.5, item.count / FEEDBACK_PERSISTENCE);  // Maintain minimum opacity
        div.textContent = text;
        feedback.appendChild(div);
    });
}

function drawLoadingBar(progress) {
    ctx.save();
    ctx.fillStyle = '#4CAF50';
    const barHeight = 20;
    const barWidth = canvas.width * progress;
    ctx.fillRect(0, canvas.height - barHeight, barWidth, barHeight);
    ctx.restore();
}

async function calibratePose() {
    if (!detector || !video.readyState || video.readyState < 2) {
        return false;
    }
    try {
        const poses = await detector.estimatePoses(video);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        if (poses && poses.length > 0) {
            const pose = poses[0];
            if (pose.keypoints && pose.keypoints.length > 0) {
                const keypoints = pose.keypoints;
                
                // Calculate shoulder rotation to check if user is sideways
                const shoulderAngle = calculateShoulderRotation(keypoints);
                const rotatedEnough = shoulderAngle > 20;

                const elbowAngle = calculateAngle(
                    keypoints[keypointMap[useRightSide ? 'right_shoulder' : 'left_shoulder']],
                    keypoints[keypointMap[useRightSide ? 'right_elbow' : 'left_elbow']],
                    keypoints[keypointMap[useRightSide ? 'right_wrist' : 'left_wrist']]
                );

                // Check if arm is straight (high elbow angle) and user is sideways
                if (elbowAngle > ELBOW_ANGLE_DOWN_THRESHOLD && rotatedEnough) {
                    console.log('Calibration successful: User is in starting position (sideways with arm straight)');
                    return true;
                } else {
                    feedbackHistory.set('Turn sideways and extend arm down', { color: '#FFA500', count: FEEDBACK_PERSISTENCE });
                    drawFeedback();
                }
            }
        }
    } catch (error) {
        console.error('Error during calibration:', error);
    }
    return false;
}

async function animateLoadingBar() {
    const duration = 1000;
    const startTime = performance.now();
    function animate(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const easedProgress = 1 - (1 - progress) * (1 - progress);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        drawLoadingBar(easedProgress);
        drawFeedback();
        if (progress < 1) {
            rafId = requestAnimationFrame(animate);
        } else {
            isCalibrating = false;
            startBtn.textContent = 'Running';
            detectPose();
        }
    }
    rafId = requestAnimationFrame(animate);
}

async function setupDetector() {
    try {
        if (typeof poseDetection === 'undefined') {
            throw new Error('Pose detection library not loaded');
        }
        const modelType = poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING;
        detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, {
            modelType,
            enableSmoothing: true
        });
        console.log('MoveNet model loaded successfully');
    } catch (error) {
        console.error('Error setting up detector:', error);
        throw error;
    }
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
                video.style.transform = 'scaleX(-1)';
                canvas.style.transform = 'scaleX(-1)';
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
let useRightSide = true;
let isCalibrating = false;
let detector;
let rafId;

async function detectPose() {
    const now = performance.now();
    if (now - lastDetectionTime < 1000 / TARGET_FPS) {
        rafId = requestAnimationFrame(detectPose);
        return;
    }
    lastDetectionTime = now;
    if (!detector || !video.readyState || video.readyState < 2) {
        rafId = requestAnimationFrame(detectPose);
        return;
    }
    detectionFound = false;
    try {
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
    } catch (error) {
        console.error('Error detecting pose:', error);
    }
    rafId = requestAnimationFrame(detectPose);
}

function calculateShoulderRotation(keypoints) {
    const leftShoulder = keypoints[keypointMap['left_shoulder']];
    const rightShoulder = keypoints[keypointMap['right_shoulder']];
    
    // Calculate the angle of shoulder line relative to horizontal
    const dx = rightShoulder.x - leftShoulder.x;
    const dy = rightShoulder.y - leftShoulder.y;
    const angle = Math.abs(Math.atan2(dy, dx) * (180 / Math.PI));
    
    return angle;
}

function getMostVisibleSide(keypoints) {
    const leftVisibility = (
        keypoints[keypointMap['left_shoulder']].score +
        keypoints[keypointMap['left_elbow']].score +
        keypoints[keypointMap['left_wrist']].score +
        keypoints[keypointMap['left_hip']].score
    );
    
    const rightVisibility = (
        keypoints[keypointMap['right_shoulder']].score +
        keypoints[keypointMap['right_elbow']].score +
        keypoints[keypointMap['right_wrist']].score +
        keypoints[keypointMap['right_hip']].score
    );
    
    return {
        isRightSide: rightVisibility >= leftVisibility,
        score: Math.max(leftVisibility, rightVisibility) / 4  // Average score (4 points)
    };
}

function processDetection(keypoints) {
    sessionData.frame_count.push(sessionData.frame_count.length);
    
    // Determine which side is more visible
    const { isRightSide, score } = getMostVisibleSide(keypoints);
    useRightSide = isRightSide;
    
    // Required landmarks for one side
    const requiredLandmarks = useRightSide ? 
        [keypointMap['right_shoulder'], keypointMap['right_elbow'], keypointMap['right_wrist'], keypointMap['right_hip']] :
        [keypointMap['left_shoulder'], keypointMap['left_elbow'], keypointMap['left_wrist'], keypointMap['left_hip']];
    
    // Check if all required landmarks are visible enough
    const allLandmarksVisible = requiredLandmarks.every(idx => keypoints[idx].score > 0.2);
    
    if (allLandmarksVisible && score > 0.3) {  // Ensure overall visibility is good enough
        detectionFound = true;

        const shoulder = keypoints[keypointMap[useRightSide ? 'right_shoulder' : 'left_shoulder']];
        const elbow = keypoints[keypointMap[useRightSide ? 'right_elbow' : 'left_elbow']];
        const wrist = keypoints[keypointMap[useRightSide ? 'right_wrist' : 'left_wrist']];
        const hip = keypoints[keypointMap[useRightSide ? 'right_hip' : 'left_hip']];

        const elbowAngle = calculateAngle(shoulder, elbow, wrist);
        const elbowXRel = elbow.x - shoulder.x;
        const elbowXMovement = calculateElbowStability(elbowXRel);

        sessionData.elbow_angles.push(elbowAngle);
        sessionData.elbow_x_positions.push(elbowXMovement);

        if (exerciseState === 'waiting') {
            // Calculate shoulder rotation to ensure user is sideways
            const shoulderAngle = calculateShoulderRotation(keypoints);
            const rotatedEnough = shoulderAngle > 20;  // Same threshold as Python version

            if (rotatedEnough && elbowAngle > ELBOW_ANGLE_DOWN_THRESHOLD) {
                positionFrames++;
                const progress = Math.min(1, positionFrames / POSITION_THRESHOLD);
                
                // Draw progress bar with improved visibility
                const barWidth = 200;
                const barHeight = 20;
                const barX = canvas.width / 2 - barWidth / 2;
                const barY = 100;
                
                // Draw background
                ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
                ctx.fillRect(barX - 2, barY - 2, barWidth + 4, barHeight + 4);
                
                // Draw progress
                ctx.fillStyle = '#4CAF50';
                ctx.fillRect(barX, barY, barWidth * progress, barHeight);
                
                if (positionFrames >= POSITION_THRESHOLD && !countdownStarted) {
                    countdownStarted = true;
                    countdownStartTime = performance.now();
                }
            } else {
                positionFrames = Math.max(0, positionFrames - 2);
                countdownStarted = false;
                feedbackHistory.set('Turn sideways and extend arm down', { color: '#FFA500', count: FEEDBACK_PERSISTENCE });
            }

            if (countdownStarted) {
                const elapsed = performance.now() - countdownStartTime;
                const secondsLeft = Math.max(0, Math.ceil((COUNTDOWN_DURATION - elapsed) / 1000));
                if (secondsLeft > 0) {
                    // Save the current canvas state
                    ctx.save();
                    
                    // Translate to center, scale to flip, translate back
                    ctx.translate(canvas.width, 0);
                    ctx.scale(-1, 1);
                    
                    // Draw countdown with improved visibility
                    ctx.font = 'bold 100px Arial';
                    ctx.fillStyle = '#FFFFFF';
                    ctx.textAlign = 'center';
                    ctx.fillText(secondsLeft, canvas.width / 2, canvas.height / 2);
                    
                    ctx.font = 'bold 30px Arial';
                    ctx.fillStyle = '#4CAF50';
                    ctx.fillText('Get ready!', canvas.width / 2, canvas.height / 2 - 60);
                    
                    // Restore the canvas state
                    ctx.restore();
                } else {
                    exerciseState = 'counting';
                    console.log('Starting to count reps!');
                }
            }
        }

        if (exerciseState === 'counting') {
            let currentRepErrors = [];
            let hasError = false;
            let errorThisFrame = false;
            
            // Check elbow stability (swinging)
            if (elbowXMovement > ELBOW_STABILITY_THRESHOLD) {
                if (elbowXMovement > BAD_ELBOW_STABILITY_THRESHOLD) {
                    currentRepErrors.push("Elbow swinging detected");
                    hasError = true;
                    errorThisFrame = true;
                }
            }

            // State machine for rep counting
            if (repState === 'down' && elbowAngle < ELBOW_ANGLE_UP_THRESHOLD) {
                repState = 'up';
                repCounted = false;
                currentRepHasError = false;
                currentRepErrors = [];
                hasError = false;
            } else if (repState === 'up' && elbowAngle > ELBOW_ANGLE_DOWN_THRESHOLD) {
                repState = 'down';
                
                if (!repCounted && (performance.now() - lastRepTime) > MIN_TIME_BETWEEN_REPS) {
                    repCounted = true;
                    counter++;
                    lastRepTime = performance.now();
                    
                    if (hasError && !currentRepHasError) {
                        badReps.total++;
                        currentRepHasError = true;
                        if (currentRepErrors.includes("Elbow swinging detected")) {
                            badReps.elbow_swinging++;
                            // Clear previous feedback and show new feedback
                            feedbackHistory.clear();
                            feedbackHistory.set('Keep your elbow steady! Avoid swinging.', { color: '#FF0000', count: FEEDBACK_PERSISTENCE });
                        }
                        badRepsCounter.textContent = `Bad Reps: ${badReps.total}`;
                    } else if (!hasError) {
                        // Clear previous feedback and show good form_feedback
                        feedbackHistory.clear();
                        feedbackHistory.set('Good form!', { color: '#4CAF50', count: FEEDBACK_PERSISTENCE });
                    }
                    
                    repCounter.textContent = `Total: ${counter}`;
                    
                    // Store rep data
                    sessionData.rep_markers.push(sessionData.frame_count.length);
                    sessionData.rep_elbow_angles.push(elbowAngle);
                    sessionData.rep_elbow_stability.push(elbowXMovement);
                    sessionData.rep_feedback.push(currentRepErrors.length > 0 ? currentRepErrors.join(", ") : "Good form");

                    if (counter === MAX_REPS_PER_SESSION) {
                        console.log("Session complete! Counter:", counter, "Bad reps:", badReps);
                        if (rafId) {
                            cancelAnimationFrame(rafId);
                            rafId = null;
                        }
                        
                        // First stop the camera and exit fullscreen
                        if (video.srcObject) {
                            video.srcObject.getTracks().forEach(track => track.stop());
                        }
                        
                        // Save session data
                        if (!sessionData.saved) {
                            generateResultsFile();
                            sessionData.saved = true;
                        }
                        
                        // Exit fullscreen first
                        exitFullscreenMode();
                        
                        // Show completion message after a short delay
                        setTimeout(() => {
                            const completionData = {
                                reps: {
                                    total: counter,
                                    correct: counter - badReps.total,
                                    incorrect: badReps.total
                                },
                                formIssues: {
                                    elbowSwinging: badReps.elbow_swinging || 0
                                },
                                performanceScore: Math.round(((counter - badReps.total) / counter) * 100) || 0
                            };
                            showSessionComplete(completionData);
                        }, 500);
                        
                        exerciseState = 'completed';
                        return;
                    }
                }
            }

            // Always draw feedback if there are messages
            if (feedbackHistory.size > 0) {
                drawFeedback();
            }
        }
    } else {
        detectionFound = false;
        // Only show position feedback if no other feedback is showing
        if (feedbackHistory.size === 0) {
            feedbackHistory.set('Move closer to camera or adjust position', { color: '#FF0000', count: FEEDBACK_PERSISTENCE });
        }
    }
}

function handleDetectionLoss() {
    detectionLostFrames++;
    if (detectionLostFrames >= MAX_DETECTION_LOST_FRAMES) {
        detectionWarning.classList.remove('hidden');
        console.log('Detection lost for too long, resetting trainer...');
        exerciseState = 'waiting';
        repState = 'down';
        positionFrames = 0;
        countdownStarted = false;
        repCounted = false;
    }
}

function drawPose(pose) {
    if (!pose || !pose.keypoints) return;
    const keypoints = pose.keypoints;
    
    // Determine which side is more visible
    const leftVisibility = keypoints[keypointMap['left_shoulder']].score + 
                         keypoints[keypointMap['left_elbow']].score + 
                         keypoints[keypointMap['left_wrist']].score;
    const rightVisibility = keypoints[keypointMap['right_shoulder']].score + 
                          keypoints[keypointMap['right_elbow']].score + 
                          keypoints[keypointMap['right_wrist']].score;
    useRightSide = rightVisibility >= leftVisibility;
    
    // Get the landmarks for the side we're tracking
    const shoulder = keypoints[keypointMap[useRightSide ? 'right_shoulder' : 'left_shoulder']];
    const elbow = keypoints[keypointMap[useRightSide ? 'right_elbow' : 'left_elbow']];
    const wrist = keypoints[keypointMap[useRightSide ? 'right_wrist' : 'left_wrist']];
    const hip = keypoints[keypointMap[useRightSide ? 'right_hip' : 'left_hip']];
    
    // Only draw if we have good visibility of the tracked side
    if (shoulder.score > 0.2 && elbow.score > 0.2 && wrist.score > 0.2) {
        // Draw arm with thicker, more visible lines
        ctx.lineWidth = 6;
        ctx.strokeStyle = '#00FF00';
        
        // Draw upper arm
        ctx.beginPath();
        ctx.moveTo(shoulder.x, shoulder.y);
        ctx.lineTo(elbow.x, elbow.y);
        ctx.stroke();
        
        // Draw forearm
        ctx.beginPath();
        ctx.moveTo(elbow.x, elbow.y);
        ctx.lineTo(wrist.x, wrist.y);
        ctx.stroke();
        
        // Draw torso line if hip is visible
        if (hip.score > 0.2) {
            ctx.beginPath();
            ctx.moveTo(shoulder.x, shoulder.y);
            ctx.lineTo(hip.x, hip.y);
            ctx.stroke();
        }
        
        // Draw keypoints with larger, more visible circles
        const points = [shoulder, elbow, wrist];
        if (hip.score > 0.2) points.push(hip);
        
        points.forEach(point => {
            // Outer circle (white)
            ctx.beginPath();
            ctx.arc(point.x, point.y, 6, 0, 2 * Math.PI);
            ctx.fillStyle = '#FFFFFF';
            ctx.fill();
            
            // Inner circle (red)
            ctx.beginPath();
            ctx.arc(point.x, point.y, 4, 0, 2 * Math.PI);
            ctx.fillStyle = '#FF0000';
            ctx.fill();
        });
    }
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
        if (!navigator.mediaDevices) {
            throw new Error('Webcam access is not supported in this browser');
        }
        await tf.setBackend('webgl');
        await tf.ready();
        if (!detector) await setupDetector();
        if (!video.srcObject) await setupCamera();
        document.body.classList.add('camera-active');
        enterFullscreenMode();
        video.play();
        isCalibrating = true;
        feedbackHistory.set('Turn sideways and extend arm down', { color: '#FFA500', count: FEEDBACK_PERSISTENCE });
        async function checkCalibration() {
            if (await calibratePose()) {
                await animateLoadingBar();
            } else {
                rafId = requestAnimationFrame(checkCalibration);
            }
        }
        setTimeout(() => {
            rafId = requestAnimationFrame(checkCalibration);
        }, 500);
        console.log('Bicep Curl Trainer started, awaiting calibration');
    } catch (error) {
        console.error('Error starting the application:', error);
        startBtn.disabled = false;
        startBtn.textContent = 'Start Camera';
        let errorMessage = 'Error starting the application: ' + error.message;
        if (error.message.includes('permission')) {
            errorMessage += '\nPlease grant webcam permissions in your browser settings and try again.';
        } else if (error.message.includes('webcam')) {
            errorMessage += '\nPlease ensure a webcam is connected and not in use by another application.';
        }
        alert(errorMessage);
    }
}

function showSessionComplete(sessionData) {
    // Hide the video container and controls
    document.querySelector('.video-container').style.display = 'none';
    document.querySelector('.controls').style.display = 'none';

    // Show the session complete screen
    const sessionComplete = document.getElementById('sessionComplete');
    sessionComplete.classList.remove('hidden');

    // Update statistics
    document.getElementById('final-total-reps').textContent = counter;
    document.getElementById('final-good-form').textContent = counter - badReps.total;
    document.getElementById('final-bad-form').textContent = badReps.total;
    document.getElementById('elbow-swinging-count').textContent = badReps.elbow_swinging;
    
    // Calculate and display performance score
    const performanceScore = Math.round(((counter - badReps.total) / counter) * 100) || 0;
    document.getElementById('final-score').textContent = performanceScore + '%';

    // Add event listener to next button if not already added
    const nextBtn = document.getElementById('nextBtn');
    if (nextBtn) {
        nextBtn.addEventListener('click', () => {
            window.location.href = '/workoutFeedback';
        });
    }
}

function generateResultsFile() {
    const currentDate = new Date();
    const dateOptions = { 
        year: 'numeric', 
        month: '2-digit', 
        day: '2-digit',
        timeZone: 'UTC'
    };
    const timeOptions = {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: true,
        timeZone: 'UTC'
    };

    const workoutData = {
        workoutName: "Bicep Curl",
        date: currentDate.toLocaleDateString('en-US', dateOptions),
        time: currentDate.toLocaleTimeString('en-US', timeOptions),
        sets: 1,
        reps: {
            total: counter,
            correct: counter - badReps.total,
            incorrect: badReps.total
        },
        formIssues: {
            elbowSwinging: badReps.elbow_swinging || 0
        },
        performanceScore: Math.round(((counter - badReps.total) / counter) * 100),
        sessionId: Math.random().toString(36).substring(2, 15),
        sessionData: {
            rep_feedback: sessionData.rep_feedback,
            saved: false
        }
    };

    console.log('Saving workout data:', workoutData);

    // Save to localStorage
    try {
        const existingData = localStorage.getItem('workoutBackup') || '[]';
        const backupData = JSON.parse(existingData);
        backupData.push(workoutData);
        localStorage.setItem('workoutBackup', JSON.stringify(backupData));
        console.log('Saved workout data to local backup');
        sessionData.saved = true;
    } catch (backupError) {
        console.error('Failed to save backup:', backupError);
    }
}

startBtn.addEventListener('click', startApp);
exitFullscreenBtn.addEventListener('click', exitFullscreenMode);

window.addEventListener('beforeunload', () => {
    if (rafId) cancelAnimationFrame(rafId);
    if (video.srcObject) video.srcObject.getTracks().forEach(track => track.stop());
});

console.log('Bicep Curl Trainer initialized. Press Start Camera to begin.');
