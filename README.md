# DEVELOPMENT-OF-A-MACHINE-LEARNING-MODEL-FOR-EYE-DISEASE-CLASSIFICATION-USING-CNN
I designed this model to accurately detect and classify three eye diseases using Convolutional Neural Networks (CNNs), with a web interface for image upload, diagnosis, and result interpretation.
## Overview.  
The Eye Disease Image Classification System is a deep learning-based diagnostic tool developed to automatically detect and classify three eye conditions from retinal fundus images. Powered by Convolutional Neural Networks (CNNs), the system is trained on a diverse dataset that includes diabetic retinopathy, glaucoma, and cataract.
To enhance accuracy and generalization while minimizing computational cost, the model incorporates data augmentation techniques. Designed for real-time usage and deployed via Flask, the system is lightweight and optimized to run on standard hardware, making it suitable for low-resource clinical environments.
This project supports early disease detection, improves diagnostic confidence, and contributes to accessible and interpretable AI-powered medical imaging.

##  Dataset and Preprocessing
- Type: Retinal fundus images
- Number of Classes: 3
- Source: Publicly available on Kaggle [Eye Diseases Classification Dataset - Kaggle](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)
- Preprocessing: Resizing, normalization, and image augmentation (rotation, flipping, contrast adjustment, etc.)
- Dataset Split: Typically 80% training, 20% testing

## Model Architecture
The eye disease classification model is a lightweight Convolutional Neural Network (CNN) built using depthwise separable convolutions (SeparableConv2D) to reduce computational cost while maintaining high accuracy.  
- The architecture includes three convolutional blocks with increasing filters (16 → 32 → 64), each followed by batch normalization and max pooling for efficient feature extraction and downsampling.  
- The extracted features are flattened and passed through a dense layer with 64 neurons activated by ReLU, followed by a dropout layer (rate: 0.4) to prevent overfitting.
- The final output layer uses a softmax activation with 3 output neurons, corresponding to the three eye disease classes in the dataset.

I compiled the model using the Adam optimizer and trained using sparse_categorical_crossentropy as the loss function. It was trained for 20 epochs with a batch size of 32, including validation for performance monitoring.

## Performance Evaluation.
<img width="790" height="72" alt="Screenshot 2025-07-13 122247" src="https://github.com/user-attachments/assets/02ca4a84-7136-49c1-b412-a7444e1023f6" />  
<img width="764" height="354" alt="Screenshot 2025-07-13 122504" src="https://github.com/user-attachments/assets/b0c41599-7484-44a7-b70f-070dcdcb6417" />  

### Classification Report.    

<img width="671" height="210" alt="image" src="https://github.com/user-attachments/assets/d5313950-ad57-493a-bbb5-54cbbc6049a7" />   

### Confusion Matrix.   

<img width="518" height="356" alt="Screenshot 2025-07-13 122403" src="https://github.com/user-attachments/assets/d7840732-9781-4fec-91a6-7ac7cc5ac954" />

## Main Menu.  
The main menu serves as the central navigation hub for the eye disease image classification system, providing users with easy access to the platform's core functionalities. I designed the interface to be user-friendly, enabling seamless interaction by allowing users to upload eye images for classification and view diagnostic results.  

<img width="879" height="579" alt="Screenshot 2025-03-26 141358" src="https://github.com/user-attachments/assets/a4ad6c6e-d479-4e69-ba3b-850aa49795da" />  

## Presentation of Results.  

<img width="819" height="603" alt="Screenshot 2025-03-26 141954" src="https://github.com/user-attachments/assets/2ac0949e-9770-41d0-8c37-9727e551c005" />  
<img width="797" height="602" alt="Screenshot 2025-03-26 141849" src="https://github.com/user-attachments/assets/0bd60abe-3eb3-4279-8246-82f86b346af1" />    
<img width="799" height="603" alt="Screenshot 2025-07-14 101712" src="https://github.com/user-attachments/assets/d45b4307-50b8-456b-b628-467aa187a25e" />    

## Contribution.
This project contributes to the advancement of artificial intelligence in medical imaging by developing a CNN-based model that accurately classifies eye diseases such as cataract, diabetic retinopathy, and glaucoma. Achieving an accuracy of 88%, the model supports early detection and enhances diagnostic reliability. It incorporates effective preprocessing techniques like contrast adjustment, noise reduction, and data augmentation to improve model performance. Built using the CRISP-DM methodology, the project follows a structured pipeline from data preparation to deployment. It reduces reliance on manual screening, improves accessibility in resource-limited environments, and demonstrates potential for integration into clinical and telemedicine platforms. Furthermore, it promotes transparency and trust through explainable AI methods, offering a scalable and interpretable solution for AI-powered healthcare diagnostics.  
## Future Research.
Future work can focus on integrating multi-modal data such as OCT scans, patient history, and angiography with fundus images to improve diagnostic accuracy. Lightweight CNN architectures optimized for mobile and edge deployment could enhance accessibility in remote settings. Federated learning may offer a secure, collaborative model of training without sharing patient data. Transfer learning and domain adaptation can boost performance with limited labeled data. Incorporating explainable AI (XAI) will improve model transparency, while tracking disease progression over time can support early intervention for chronic conditions like glaucoma and diabetic retinopathy.  

## Model Download
You can download the trained CNN model for eye disease prediction using the link below:

[Download Eye_disease_model (.h5)](https://drive.google.com/drive/folders/1Bim1I_FBKmEg5bWEczxgaG6DDHK4s_qY?usp=sharing)


Please Note: The model is provided for academic, research, or live deployment on standard hardware. Permission may be required to access the file.  

## How to Run the Flask App.
Follow the steps below to run the eye disease prediction model using the Flask web framework:

- Clone the Repository.
- Install dependencies. Make sure you have Python installed (Python 3.7+ is recommended), then install all required packages.
- Download the dataset publicly available on Kaggle [Eye Diseases Classification Dataset - Kaggle](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)
- Download the trained .h5 model from Google Drive: [Download Eye_disease_model (.h5)](https://drive.google.com/drive/folders/1Bim1I_FBKmEg5bWEczxgaG6DDHK4s_qY?usp=sharing)
- After download, place the Eye_disease_model.h5 file in the root directory of the project (same folder as app.py).
- Run the Flask app using python app.py in your terminal.
- Open a browser, and type this address http://127.0.0.1:5000/ to interact with the model.
## Reference.
Kaggle dataset: https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification
