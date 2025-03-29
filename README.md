# NeuralNetLife

## Project Description
**NeuralNetLife** is a pioneering project that leverages neural networks to predict life insurance costs. Utilizing Python, Docker, and Google Cloud technologies, this solution aims to provide an accurate and accessible platform for insurance cost prediction.

## Installation Instructions
Follow these steps to get **NeuralNetLife** up and running on your system:

1. **Prerequisites**:
   - Ensure that Docker is installed on your machine. For installation instructions, visit [Docker's official site](https://www.docker.com/products/docker-desktop).

2. **Build and Run the Docker Container**:
   - Navigate to the project directory and build the Docker container using the provided Dockerfile located in `.devcontainer`:
     ```bash
     docker build -t neuralnetlife .devcontainer
     ```
   - Once the build is complete, run the Docker container:
     ```bash
     docker run -p 8501:8501 neuralnetlife
     ```
   - Start the application by executing:
     ```bash
     streamlit run streamlit_app.py
     ```
   - This command will start the website locally on your machine, accessible via a web browser at `http://localhost:8501`.

## Usage
To use **NeuralNetLife**, you have two main options:

- **Hosting the Website**:
  - Simply run:
    ```bash
    streamlit run streamlit_app.py
    ```
  - This command will start the website, where you can interact with the neural network predictions.

- **Using Pretrained Neural Networks**:
  - If you want to utilize the pretrained models directly, refer to the `neural_network.py` file. This script provides functionality to load and interact with the pretrained neural networks.

## Contributing
Contributions to **NeuralNetLife** are welcome! Current contributors to the project are:
- Jack
- Tony
- Jiaxin

We appreciate your interest in improving **NeuralNetLife** and look forward to your creative and innovative enhancements.
