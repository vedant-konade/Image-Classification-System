CIFAR-10 Image Classification Project

This project is a complete implementation of an image classifier trained on the CIFAR-10 dataset using PyTorch.
The goal was to build the entire pipeline from scratch — data loading, model training, evaluation, and a small user interface for trying out predictions.

1. Dataset

The CIFAR-10 dataset contains 60,000 images in ten categories.

Images were normalised and augmented using random crop and horizontal flipping.

Training, validation and test splits were applied according to the assignment requirements.

2. Model

A custom CNN was designed with:

Three convolution blocks

Batch Normalisation

ReLU activations

Max-pooling

Dropout for regularisation

Training used the Adam optimiser with early stopping and a learning-rate scheduler.
The final model reached about 84.8% test accuracy.

3. Evaluation

The following were generated:

Accuracy and loss curves

Classification report including precision, recall and F1-score

Confusion matrix

Sample prediction outputs

These files are placed in the outputs/ directory.

4. Streamlit App

A small web interface was created using Streamlit.
It allows the user to upload an image and view:

The predicted class

Confidence score

Top-five predicted classes

The UI includes a custom design with simple CSS styling.

Run using:

streamlit run app/streamlit_app.py

5. How to Run Training
python -m src.train


Evaluate using:

python -m src.evaluate

6. Project Structure
imgclf/
├─ src/
│  ├─ train.py  
│  ├─ evaluate.py  
│  ├─ predict.py  
│  └─ utils.py  
├─ models/  
│  └─ cnn.py  
├─ app/  
│  └─ streamlit_app.py  
├─ outputs/  
│  └─ model + plots  
└─ requirements.txt  


  
7. Notes

The project is modular, easy to modify, and can be extended with transfer learning if required.

