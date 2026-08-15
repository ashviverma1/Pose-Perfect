"""
POSE-PERFECT

A project created to allow users to correct the specific poses pre-modeled to ensure they are perfecting them!

The poses already created are:

  - Finger heart (thumb + index of each hand crossing near each other)
        -> "You are in love with the game"
  - Thumbs up / thumbs down (one hand)
        -> "You are saying good job!" / "You are saying bad job!"
  - Clap (both wrists close together)
        -> "You are proud"
  - Wave (one hand raised and moving side-to-side)
        -> "You are saying hello!"
  - Dab (one arm bent across the face, other arm extended out straight,
    full body visible)
        -> "You are using the dab"
  - Peace sign, both hands (index + middle up, ring + pinky curled, on each hand)
        -> "You are keeping the peace"
  - Praying hands (palms pressed together, fingers pointing up)
        -> "You are praying for patience"
  - Rock horns, either hand (index + pinky up, middle + ring curled)
        -> "You are a rockstar"
  - Point to the sky (one arm raised straight overhead)
        -> "You got the moves"
  - Namaste (palms together at chest height)
        -> "You are saying namaste"
  - Superhero pose (hands on hips, feet planted wide)
        -> "You are a superhero"
  - Tree pose (one foot resting near the opposite knee, arms overhead, full body visible)
        -> "You found his zen"
  - Lunge (one knee bent deeply forward, other leg extended back)
        -> "You are lunging into action"
  - Thinking pose (hand near the chin, elbow bent, like "The Thinker")
        -> "You are deep in thought"

  q  -> quit
"""

import math
import time
import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ----------------------------------------------------------------------
# Tunable thresholds
# ----------------------------------------------------------------------
HEART_DIST_THRESHOLD = 0.08      # usual distance between fingertips for finger-heart
CLAP_DIST_THRESHOLD = 0.08       # usual distance between wrists for clap
PRAYER_DIST_THRESHOLD = 0.06     # usual distance between matching fingertips for praying hands
MESSAGE_HOLD_SECONDS = 1.5       # how long a caption stays on screen after being triggered
COOLDOWN_SECONDS = 1.0           # minimum time between re-triggering the SAME message


def distance(a, b):
    #Euclidean distance between two mediapipe landmarks (normalized coords).
    return math.hypot(a.x - b.x, a.y - b.y)


def all_visible(landmarks, min_visibility=0.5):
    #True if every given landmark has visibility above the threshold (or no visibility attr at all).
    return all(getattr(p, "visibility", 1.0) >= min_visibility for p in landmarks)


def angle_at_point(a, b, c):
    #Angle at point b, formed by points a-b-c.
    ab = (a.x - b.x, a.y - b.y)
    cb = (c.x - b.x, c.y - b.y)
    dot = ab[0] * cb[0] + ab[1] * cb[1]
    mag_ab = math.hypot(*ab)
    mag_cb = math.hypot(*cb)
    if mag_ab * mag_cb == 0:
        return 0
    cos_angle = max(-1.0, min(1.0, dot / (mag_ab * mag_cb)))
    return math.degrees(math.acos(cos_angle))


def detect_thumbs(hand_landmarks):
    """Return 'up', 'down', or None for a single hand."""
    lm = hand_landmarks.landmark
    wrist = lm[0]
    thumb_tip = lm[4]
    thumb_mcp = lm[2]

    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]

    # Other four fingers should be curled (tip below/near pip -> folded into fist)
    curled = all(lm[t].y > lm[p].y - 0.02 for t, p in zip(finger_tips, finger_pips))
    if not curled:
        return None

    # Thumb pointing clearly up or down relative to its own base joint
    if thumb_tip.y < thumb_mcp.y - 0.08 and thumb_tip.y < wrist.y - 0.08:
        return "up"
    if thumb_tip.y > thumb_mcp.y + 0.08 and thumb_tip.y > wrist.y + 0.08:
        return "down"
    return None


def detect_finger_heart(left_hand, right_hand):
    #Finger-heart: thumb of one hand crosses near index of the other, vice versa.
    if left_hand is None or right_hand is None:
        return False
    l_thumb, l_index = left_hand.landmark[4], left_hand.landmark[8]
    r_thumb, r_index = right_hand.landmark[4], right_hand.landmark[8]

    cross1 = distance(l_thumb, r_index) < HEART_DIST_THRESHOLD
    cross2 = distance(r_thumb, l_index) < HEART_DIST_THRESHOLD
    return cross1 and cross2


def detect_clap(left_hand, right_hand):
    if left_hand is None or right_hand is None:
        return False
    l_wrist = left_hand.landmark[0]
    r_wrist = right_hand.landmark[0]
    return distance(l_wrist, r_wrist) < CLAP_DIST_THRESHOLD


def detect_hello_wave(left_hand, right_hand, pose_landmarks, prev_wrist_positions):
    #Rough hello-wave heuristic: one hand is raised near shoulder height and moves sideways.
    if pose_landmarks is None:
        return False

    lm = pose_landmarks.landmark
    left_shoulder, right_shoulder = lm[11], lm[12]
    left_hip, right_hip = lm[23], lm[24]

    def is_waving(hand_landmarks, shoulder, hip, prev_position):
        if hand_landmarks is None or prev_position is None:
            return False

        wrist = hand_landmarks.landmark[0]
        wrist_y = wrist.y
        shoulder_y = shoulder.y
        hip_y = hip.y

        if not (wrist_y < shoulder_y + 0.08 and wrist_y > hip_y - 0.05):
            return False

        horizontal_move = wrist.x - prev_position[0]
        vertical_change = abs(wrist_y - prev_position[1])
        return abs(horizontal_move) > 0.12 and vertical_change < 0.12

    if is_waving(left_hand, left_shoulder, left_hip, prev_wrist_positions.get("left")):
        return True
    if is_waving(right_hand, right_shoulder, right_hip, prev_wrist_positions.get("right")):
        return True
    return False


def detect_peace_sign(hand_landmarks):
    #Single hand: index + middle extended, ring + pinky curled.
    lm = hand_landmarks.landmark
    index_up = lm[8].y < lm[6].y - 0.03
    middle_up = lm[12].y < lm[10].y - 0.03
    ring_curled = lm[16].y > lm[14].y - 0.02
    pinky_curled = lm[20].y > lm[18].y - 0.02
    return index_up and middle_up and ring_curled and pinky_curled


def detect_both_hands_peace(left_hand, right_hand):
    if left_hand is None or right_hand is None:
        return False
    return detect_peace_sign(left_hand) and detect_peace_sign(right_hand)


def detect_praying_hands(left_hand, right_hand):
    """Palms pressed together, fingers pointing up: matching fingertips close together,
    and fingertips above their own wrist (hands held upright, not just resting near each other)."""
    if left_hand is None or right_hand is None:
        return False
    l_lm, r_lm = left_hand.landmark, right_hand.landmark

    fingertip_ids = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
    dists = [distance(l_lm[i], r_lm[i]) for i in fingertip_ids]
    close_together = all(d < PRAYER_DIST_THRESHOLD for d in dists)

    l_wrist, r_wrist = l_lm[0], r_lm[0]
    fingers_up = (l_lm[12].y < l_wrist.y - 0.05) and (r_lm[12].y < r_wrist.y - 0.05)

    return close_together and fingers_up


def detect_rock_horns(hand_landmarks):
    """Single hand: index + pinky extended, middle + ring curled (classic 'rock on' sign)."""
    lm = hand_landmarks.landmark
    index_up = lm[8].y < lm[6].y - 0.03
    pinky_up = lm[20].y < lm[18].y - 0.03
    middle_curled = lm[12].y > lm[10].y - 0.02
    ring_curled = lm[16].y > lm[14].y - 0.02
    return index_up and pinky_up and middle_curled and ring_curled


def is_prayer_near_face(pose_landmarks, left_hand, right_hand):
    """Distinguish 'namaste' (hands at chest) from 'praying for patience' (hands near the face).
    Falls back to treating it as namaste if pose isn't visible."""
    if pose_landmarks is None or left_hand is None or right_hand is None:
        return False
    nose_y = pose_landmarks.landmark[0].y
    wrist_y = (left_hand.landmark[0].y + right_hand.landmark[0].y) / 2
    return wrist_y < nose_y + 0.05


def detect_point_to_sky(pose_landmarks):
    """One arm raised straight up above the head, the other arm not raised."""
    if pose_landmarks is None:
        return False
    lm = pose_landmarks.landmark
    NOSE = lm[0]
    L_SHOULDER, R_SHOULDER = lm[11], lm[12]
    L_ELBOW, R_ELBOW = lm[13], lm[14]
    L_WRIST, R_WRIST = lm[15], lm[16]

    if not all_visible([NOSE, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST]):
        return False

    def arm_raised_straight(shoulder, elbow, wrist):
        return wrist.y < NOSE.y - 0.05 and angle_at_point(shoulder, elbow, wrist) > 140

    left_raised = arm_raised_straight(L_SHOULDER, L_ELBOW, L_WRIST)
    right_raised = arm_raised_straight(R_SHOULDER, R_ELBOW, R_WRIST)

    return left_raised != right_raised  # exactly one arm raised, not both


def detect_superhero_pose(pose_landmarks):
    #Hands on hips, feet planted wide apart.
    if pose_landmarks is None:
        return False
    lm = pose_landmarks.landmark
    L_SHOULDER, R_SHOULDER = lm[11], lm[12]
    L_WRIST, R_WRIST = lm[15], lm[16]
    L_HIP, R_HIP = lm[23], lm[24]
    L_ANKLE, R_ANKLE = lm[27], lm[28]

    needed = [L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST, L_HIP, R_HIP, L_ANKLE, R_ANKLE]
    if not all_visible(needed):
        return False

    hands_on_hips = distance(L_WRIST, L_HIP) < 0.12 and distance(R_WRIST, R_HIP) < 0.12
    shoulder_width = distance(L_SHOULDER, R_SHOULDER)
    stance_width = distance(L_ANKLE, R_ANKLE)
    feet_apart = stance_width > shoulder_width * 1.2

    return hands_on_hips and feet_apart


def detect_tree_pose(pose_landmarks):
    #One foot lifted and resting near the opposite knee, arms raised overhead.
    if pose_landmarks is None:
        return False
    lm = pose_landmarks.landmark
    NOSE = lm[0]
    L_WRIST, R_WRIST = lm[15], lm[16]
    L_KNEE, R_KNEE = lm[25], lm[26]
    L_ANKLE, R_ANKLE = lm[27], lm[28]

    needed = [NOSE, L_WRIST, R_WRIST, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE]
    if not all_visible(needed):
        return False

    arms_up = L_WRIST.y < NOSE.y and R_WRIST.y < NOSE.y

    def foot_lifted_near_opposite_knee(lifted_ankle, opposite_knee, standing_ankle):
        lifted = lifted_ankle.y < standing_ankle.y - 0.1
        near_knee = abs(lifted_ankle.x - opposite_knee.x) < 0.12
        return lifted and near_knee

    left_leg_tree = foot_lifted_near_opposite_knee(L_ANKLE, R_KNEE, R_ANKLE)
    right_leg_tree = foot_lifted_near_opposite_knee(R_ANKLE, L_KNEE, L_ANKLE)

    return arms_up and (left_leg_tree or right_leg_tree)


def detect_lunge(pose_landmarks):
    #One knee bent deeply forward, other leg extended back, feet spread in a wide stride.
    if pose_landmarks is None:
        return False
    lm = pose_landmarks.landmark
    L_HIP, R_HIP = lm[23], lm[24]
    L_KNEE, R_KNEE = lm[25], lm[26]
    L_ANKLE, R_ANKLE = lm[27], lm[28]

    needed = [L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE]
    if not all_visible(needed):
        return False

    left_knee_angle = angle_at_point(L_HIP, L_KNEE, L_ANKLE)
    right_knee_angle = angle_at_point(R_HIP, R_KNEE, R_ANKLE)

    one_bent_one_straight = (
        (left_knee_angle < 110 and right_knee_angle > 150)
        or (right_knee_angle < 110 and left_knee_angle > 150)
    )

    stride_width = distance(L_ANKLE, R_ANKLE)
    hip_width = distance(L_HIP, R_HIP)
    wide_stride = stride_width > hip_width * 1.5

    return one_bent_one_straight and wide_stride


def detect_dab(pose_landmarks):
    """
      - one elbow bent sharply (wrist near/above shoulder height, close to head)
      - the other arm extended fairly straight, raised up and away from the body
    Requires full upper body to be visible otherwise you're getting last warned
    """
    if pose_landmarks is None:
        return False
    lm = pose_landmarks.landmark

    L_SHOULDER, R_SHOULDER = lm[11], lm[12]
    L_ELBOW, R_ELBOW = lm[13], lm[14]
    L_WRIST, R_WRIST = lm[15], lm[16]
    NOSE = lm[0]

    # visibility check (helps ensure "whole body / upper body" is actually in frame) otherwise its wraps
    needed = [L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, NOSE]
    if not all_visible(needed):
        return False

    left_elbow_angle = angle_at_point(L_SHOULDER, L_ELBOW, L_WRIST)
    right_elbow_angle = angle_at_point(R_SHOULDER, R_ELBOW, R_WRIST)

    def is_bent_and_near_face(wrist, elbow_angle):
        return elbow_angle < 90 and distance(wrist, NOSE) < 0.25

    def is_extended_and_raised(wrist, shoulder, elbow_angle):
        return elbow_angle > 140 and wrist.y < shoulder.y

    left_bent = is_bent_and_near_face(L_WRIST, left_elbow_angle)
    right_extended = is_extended_and_raised(R_WRIST, R_SHOULDER, right_elbow_angle)

    right_bent = is_bent_and_near_face(R_WRIST, right_elbow_angle)
    left_extended = is_extended_and_raised(L_WRIST, L_SHOULDER, left_elbow_angle)

    return (left_bent and right_extended) or (right_bent and left_extended)


def detect_thinking_pose(pose_landmarks):
    #One hand resting near the chin with the elbow bent like the monkey hmm
    if pose_landmarks is None:
        return False
    lm = pose_landmarks.landmark
    MOUTH_LEFT, MOUTH_RIGHT = lm[9], lm[10]
    L_SHOULDER, R_SHOULDER = lm[11], lm[12]
    L_ELBOW, R_ELBOW = lm[13], lm[14]
    L_WRIST, R_WRIST = lm[15], lm[16]

    needed = [MOUTH_LEFT, MOUTH_RIGHT, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST]
    if not all_visible(needed):
        return False

    chin_x = (MOUTH_LEFT.x + MOUTH_RIGHT.x) / 2
    chin_y = (MOUTH_LEFT.y + MOUTH_RIGHT.y) / 2 + 0.04  # approximate chin, just below the mouth

    def hand_propping_chin(wrist, elbow, shoulder):
        near_chin = math.hypot(wrist.x - chin_x, wrist.y - chin_y) < 0.12
        elbow_bent = angle_at_point(shoulder, elbow, wrist) < 100
        return near_chin and elbow_bent

    left_thinking = hand_propping_chin(L_WRIST, L_ELBOW, L_SHOULDER)
    right_thinking = hand_propping_chin(R_WRIST, R_ELBOW, R_SHOULDER)

    return left_thinking or right_thinking


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. Check your camera index/permissions.") #wraps
        return

    current_message = ""
    message_expires_at = 0.0
    last_triggered = {}  # message -> timestamp, for cooldown
    prev_wrist_positions = {"left": None, "right": None}

    def trigger(message):
        nonlocal current_message, message_expires_at
        now = time.time()
        if now - last_triggered.get(message, 0) < COOLDOWN_SECONDS and current_message == message:
            # still refresh the hold time so it doesn't flicker off
            message_expires_at = now + MESSAGE_HOLD_SECONDS
            return
        last_triggered[message] = now
        current_message = message
        message_expires_at = now + MESSAGE_HOLD_SECONDS

    with mp_holistic.Holistic(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        model_complexity=1,
    ) as holistic:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # mirror for a natural selfie-view
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True

            left_hand = results.left_hand_landmarks
            right_hand = results.right_hand_landmarks
            pose = results.pose_landmarks

            #just scroll lil down and take a lil guess how many if statements you boutta see
            if detect_dab(pose):
                trigger("Dad is using the dab")
            elif detect_tree_pose(pose):
                trigger("Dad found his zen")
            elif detect_superhero_pose(pose):
                trigger("Dad is a superhero")
            elif detect_lunge(pose):
                trigger("Dad is lunging into action")
            elif detect_thinking_pose(pose):
                trigger("Dad is deep in thought")
            elif detect_point_to_sky(pose):
                trigger("Dad's got the moves")
            elif detect_praying_hands(left_hand, right_hand):
                if is_prayer_near_face(pose, left_hand, right_hand):
                    trigger("Dad is praying for patience")
                else:
                    trigger("Dad says namaste")
            elif detect_hello_wave(left_hand, right_hand, pose, prev_wrist_positions):
                trigger("Dad is saying hello!")
            elif detect_finger_heart(left_hand, right_hand):
                trigger("Dad is in love with the game")
            elif detect_both_hands_peace(left_hand, right_hand):
                trigger("Dad is keeping the peace")
            elif detect_clap(left_hand, right_hand):
                trigger("Dad is proud")
            else:
                rock_horns = any(
                    detect_rock_horns(hand) for hand in (left_hand, right_hand) if hand is not None
                )
                thumb_result = None
                for hand in (left_hand, right_hand):
                    if hand is not None:
                        thumb_result = detect_thumbs(hand)
                        if thumb_result:
                            break
                if rock_horns:
                    trigger("Dad is a rockstar")
                elif thumb_result == "up":
                    trigger("Dad says good job!")
                elif thumb_result == "down":
                    trigger("Dad says bad job!")
            if left_hand is not None:
                prev_wrist_positions["left"] = (left_hand.landmark[0].x, left_hand.landmark[0].y)
            if right_hand is not None:
                prev_wrist_positions["right"] = (right_hand.landmark[0].x, right_hand.landmark[0].y)

            #Draw landmarks
            if pose:
                mp_drawing.draw_landmarks(
                    frame, pose, mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )
            if left_hand:
                mp_drawing.draw_landmarks(frame, left_hand, mp_holistic.HAND_CONNECTIONS)
            if right_hand:
                mp_drawing.draw_landmarks(frame, right_hand, mp_holistic.HAND_CONNECTIONS)

            #Draw caption
            if current_message and time.time() < message_expires_at:
                h, w = frame.shape[:2]
                text = current_message
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1.1
                thickness = 3
                (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
                x = (w - tw) // 2
                y = h - 40
                # background box for readability
                cv2.rectangle(frame, (x - 15, y - th - 15), (x + tw + 15, y + 15), (0, 0, 0), -1)
                cv2.putText(frame, text, (x, y), font, scale, (0, 255, 255), thickness, cv2.LINE_AA)

            cv2.imshow("Dad Pose Detector - press q to quit", frame)
            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
