# Birthday-Project

This project uses MediaPipe Holistic to detect body and hand gestures from a live webcam feed and display a fun caption on screen. It is meant to be a playful pose detector that recognizes actions such as waving, clapping, praying, pointing, and other expressive gestures.

## What the code does

The script in [pose-detector.py](pose-detector.py) captures video from your webcam, processes each frame with MediaPipe Holistic, and analyzes body-pose and hand-landmark data. Based on the detected pose or gesture, it displays a matching message such as:

- "Dad is saying hello!"
- "Dad is proud"
- "Dad says good job!"
- "Dad is using the dab"

## How it works

1. The webcam feed is read frame by frame.
2. Each frame is converted to RGB and processed by MediaPipe Holistic.
3. The script extracts pose landmarks and hand landmarks.
4. Several heuristic functions check for specific gestures.
5. When a gesture is detected, the script shows a caption on the screen.

## Features

The current version can recognize a variety of simple gestures, including:

- Wave hello
- Clap
- Finger heart
- Peace sign
- Praying hands
- Rock horns
- Point to the sky
- Superhero pose
- Tree pose
- Lunge
- Dab
- Thinking pose

## Requirements

Install the required packages:

```bash
pip install opencv-python mediapipe
```

## Run the project

From the project folder, run:

```bash
python pose-detector.py
```

Press q to quit the webcam window.

## Notes

The gesture detection is heuristic-based, which means it uses simple rules and thresholds rather than a trained machine-learning classifier. This makes it easy to tweak if you want to improve accuracy for your own movements.

