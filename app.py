import cv2
import mediapipe as mp
import numpy as np
import Quartz.CoreGraphics as CG
import math

# --- Precision Tuning ---
CAM_W, CAM_H = 640, 480
# The smaller this box, the less you have to move your hand to cover the screen
ZONE_X1, ZONE_X2 = 140, 500 
ZONE_Y1, ZONE_Y2 = 100, 380

# Pinch Sensitivity
CLICK_THRESHOLD = 32   # Distance to trigger "Press"
RELEASE_THRESHOLD = 45 # Distance to trigger "Release" (slightly larger to prevent flickering)
SMOOTHING = 2          # 1 = Instant, 3 = Stable. 2 is ideal for Chess.

# --- Native Mac Mouse Logic ---
def post_mouse_event(x, y, event_type, button=CG.kCGMouseButtonLeft):
    event = CG.CGEventCreateMouseEvent(None, event_type, (x, y), button)
    CG.CGEventPost(CG.kCGHIDEventTap, event)

# --- Setup MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)

cap = cv2.VideoCapture(0)
screen_w = CG.CGDisplayPixelsWide(CG.CGMainDisplayID())
screen_h = CG.CGDisplayPixelsHigh(CG.CGMainDisplayID())

ploc_x, ploc_y = 0, 0
is_dragging = False

print("DRAG-AND-DROP MODE ACTIVE")

while cap.isOpened():
    success, img = cap.read()
    if not success: break
    
    img = cv2.flip(img, 1)
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Draw active zone for visual reference
    cv2.rectangle(img, (ZONE_X1, ZONE_Y1), (ZONE_X2, ZONE_Y2), (255, 0, 255), 2)
    
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        idx = hand.landmark[8] # Index tip
        tmb = hand.landmark[4] # Thumb tip
        
        # Convert to pixels
        ix, iy = int(idx.x * CAM_W), int(idx.y * CAM_H)
        tx, ty = int(tmb.x * CAM_W), int(tmb.y * CAM_H)
        
        # 1. Map hand position to screen
        sx = np.interp(ix, (ZONE_X1, ZONE_X2), (0, screen_w))
        sy = np.interp(iy, (ZONE_Y1, ZONE_Y2), (0, screen_h))
        
        # 2. Smoothing
        cx = ploc_x + (sx - ploc_x) / SMOOTHING
        cy = ploc_y + (sy - ploc_y) / SMOOTHING
        
        # 3. Handle Movement and Dragging
        dist = math.hypot(ix - tx, iy - ty)
        
        if dist < CLICK_THRESHOLD:
            if not is_dragging:
                post_mouse_event(cx, cy, CG.kCGEventLeftMouseDown)
                is_dragging = True
            else:
                # While pinched, we use the "MouseDrag" event
                post_mouse_event(cx, cy, CG.kCGEventLeftMouseDragged)
        elif dist > RELEASE_THRESHOLD:
            if is_dragging:
                post_mouse_event(cx, cy, CG.kCGEventLeftMouseUp)
                is_dragging = False
            else:
                # Not pinching, just move the cursor normally
                post_mouse_event(cx, cy, CG.kCGEventMouseMoved)
        
        ploc_x, ploc_y = cx, cy

        # Visual feedback on camera
        color = (0, 255, 0) if is_dragging else (255, 0, 0)
        cv2.circle(img, (ix, iy), 10, color, cv2.FILLED)

    cv2.imshow("Chess Hand Controller", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# Safety release
if is_dragging: post_mouse_event(ploc_x, ploc_y, CG.kCGEventLeftMouseUp)
cap.release()
cv2.destroyAllWindows()