# Object Tracker

Object Tracker is a Python-based application that utilizes deep learning models (YOLO and ResNet) for detecting, tracking, and re-identifying objects in video streams. The tool is designed to handle scenarios where objects might go out of frame or become obstructed and provides re-identification of the target once it's visible again.

## Features

- **Real-Time Object Detection**: Uses the YOLOv8 model for fast and accurate object detection.
- **Object Re-Identification**: Integrates ResNet-based feature extraction to re-identify objects even after occlusion or being out of frame.
- **Custom Target Tracking**: Allows users to select a specific object to track by clicking on it.
- **Unique Object IDs**: Assigns and displays unique IDs for each detected object.
- **User-Friendly Interface**: Includes a reset button and visual indicators for selected targets.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/bhanuteja3264/object_tracker.git
   cd object_tracker
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the YOLOv8 model:
   - Place the `yolov8n.pt` model file in the root directory. (You can download it from the [Ultralytics YOLOv8 repository](https://github.com/ultralytics/ultralytics)).

## Usage

1. Run the tracker:
   ```bash
   python app.py
   ```

2. Select a target:
   - Click on an object in the video feed to start tracking it.
   - The selected object will be highlighted, and its ID will be displayed.

3. Reset tracking:
   - Click the "RESET" button in the top-left corner to clear all targets and start fresh.

## Project Structure

- `app.py`: The main application script.
- `requirements.txt`: List of required Python packages.
- `README.md`: Project documentation (this file).

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a clear description of your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the object detection model.
- [PyTorch](https://pytorch.org/) for the deep learning framework.
