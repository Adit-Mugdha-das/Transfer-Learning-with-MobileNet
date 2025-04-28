# Transfer Learning with MobileNet

This project applies **transfer learning** using the **MobileNet** architecture to classify images efficiently with a pre-trained convolutional neural network.  
It is part of **Week 2 (Course 4: Convolutional Neural Networks)** from the **Deep Learning Specialization** by **Andrew Ng** on Coursera.

##  Description

In this lab, I leveraged a pre-trained **MobileNet** model as a feature extractor and fine-tuned it for a new image classification task.  
MobileNet is designed for lightweight and efficient neural networks, making it ideal for mobile and embedded vision applications.

The workflow included:
- Loading a pre-trained MobileNet model
- Modifying the final classification layers
- Freezing pre-trained layers to preserve learned representations
- Fine-tuning only the top layers
- Evaluating performance on a custom dataset

##  Important Notes

To keep the repository lightweight and fit GitHub's upload limits:
- The `.ipynb_checkpoints/` and `__pycache__/` folders have been **removed**.
- The **`imagenet_base_model/` folder's MobileNet `.h5` files** have been **removed** to reduce size.
- Only a lightweight JSON file (`imagenet_class_index.json`) is optionally kept for class label mapping.
- The MobileNet model weights will be **automatically downloaded** during notebook execution via TensorFlow/Keras if not found locally.
-  **The `dataset/` folder containing training images was also removed** to further reduce repository size. 

>  If you want to manually download MobileNet pre-trained weights, you can find them here:  
>  [Keras Applications - MobileNet V1](https://keras.io/api/applications/mobilenet/#mobilenetv1-function)

>  To run the notebook, you can either:
> - Use your own small image dataset structured in folders by class (e.g., `alpaca/`, `not alpaca/`)
> - Or modify the code to download a sample dataset automatically

##  Key Concepts Covered

- Transfer Learning
- Feature Extraction
- Fine-Tuning Pre-trained Models
- Efficient Architectures for Mobile and Edge Devices
- MobileNet Model Application for Image Classification

##  Files Included

- `Transfer_learning_with_MobileNet_v1.ipynb` — Main Jupyter notebook
- `test_utils.py` — Helper utility functions
- `images/` — Supporting small images
- `imagenet_base_model/` — (Optional) Class label mapping JSON

##  Tools Used

- Python 3
- TensorFlow / Keras
- NumPy
- Matplotlib
- Jupyter Notebook

##  Course Info

This project is part of:
> [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)  
> Instructor: **Andrew Ng**  
> Course 4: Convolutional Neural Networks  
> Week 2: Transfer Learning and MobileNet

##  License

This repository is created for educational and portfolio purposes only.  
Please do not use it for direct assignment submission.

---

 Feel free to star this repository if you found the work useful or inspiring!
