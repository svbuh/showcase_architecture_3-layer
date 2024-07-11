
# Image Classification with CNN

This project implements an image classification system using Convolutional Neural Networks (CNNs). The system is built with PyTorch and Flask, allowing for both training models on the CIFAR-10 dataset and serving a web interface for image classification.

## Project Structure

- `models/`: Contains the model definitions for both Simple and Advanced CNNs.
  - `simple_cnn_model.py`: Defines the `SimpleCNN` model.
  - `advanced_cnn_model.py`: Defines the `AdvancedCNN` model.
- `controllers/`: Contains the Flask application.
  - `prediction_controller.py`: Defines the Flask app and routes.
- `services/`: Contains the prediction service.
  - `prediction_service.py`: Provides methods for image preprocessing and prediction.
- `templates/`: Contains the HTML templates for the web interface.
  - `index.html`: Main page for image upload and URL input.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/svbuh/showcase_architecture_3-layer.git
    cd showcase_architecture_3-layer
    ```

2. Create a virtual environment:
    ```sh
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Training the Model

To train the model, run the `train_model.py` script. You can choose between the `SimpleCNN` and `AdvancedCNN` models by uncommenting the desired model in the script.

```sh
python train_model.py
```

This script will:
1. Load the CIFAR-10 dataset.
2. Train the selected model.
3. Save the trained model to a `.pth` file.
4. Evaluate the model on the test set and print the accuracy.

## Running the Flask App

1. Ensure the Flask app configuration points to the correct model path in `services/prediction_service.py`.
2. Run the Flask app:
    ```sh
    python controllers/prediction_controller.py
    ```
3. Open your browser and navigate to `http://127.0.0.1:5000/` to access the web interface.

## Usage

### Web Interface

The web interface allows users to classify images either by uploading a file or by providing an image URL.

1. **Provide an Image URL**:
   - Enter the image URL in the provided input field.
   - Click on the "Classify" button to see the predicted class.

### Command Line

The `train_model.py` script can be run to train the model and evaluate it on the CIFAR-10 test set.

## Dependencies

- `torch==2.3.1`
- `torchvision==0.18.1`
- `numpy==1.26.4`
- `Pillow==10.3.0`
- `matplotlib==3.9.0`
- `Werkzeug==3.0.3`
- `Flask==3.0.3`
- `requests==2.32.3`

## Acknowledgements

- The CIFAR-10 dataset is used for training and evaluation.
- The project uses pre-trained models from `torchvision.models`.
