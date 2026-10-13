# Production-Ready MNIST MLOps Pipeline

A full-stack machine learning inference service built with PyTorch, ONNX Runtime, FastAPI, and Docker. This pipeline trains a custom Convolutional Neural Network (CNN), exports it to an optimized ONNX format for high-speed CPU inference, serves it via an asynchronous REST API, and provides an interactive HTML5 canvas UI.

## 🚀 Key Features
* **Custom CNN Architecture:** Achieves high test accuracy on the MNIST dataset using batch normalization, dropout, and multi-layer convolutions.
* **Model Optimization:** Exports PyTorch model weights to **ONNX Runtime** for optimized, low-latency inference.
* **Production Microservice:** Asynchronous REST API powered by **FastAPI** with pre-processing pipelines for raw image inputs.
* **Containerization:** Fully packaged with **Docker** for seamless deployment across any environment.
* **Interactive UI:** Built-in web client featuring an HTML5 drawing canvas to test predictions in real time.

---

## 🛠️ Tech Stack
* **Deep Learning:** PyTorch, Torchvision
* **Inference Engine:** ONNX Runtime
* **Backend Framework:** FastAPI, Uvicorn
* **Deployment:** Docker
* **Frontend:** HTML5 Canvas, JavaScript

---

## 📂 Project Structure
```text
mnist-production-pipeline/
├── app/
│   ├── main.py          # FastAPI application & inference route
│   └── static/          # HTML5 canvas UI client
├── models/
│   └── mnist_cnn.onnx   # Optimized ONNX model weights
├── src/
│   ├── model.py         # Custom CNN architecture definition
│   └── train.py         # Training, evaluation, and ONNX export script
├── Dockerfile           # Container configuration
├── requirement.txt      # Project dependencies
└── README.md