# Waste Classifier

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-purple.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Latest Release](https://img.shields.io/github/v/release/Stenberg-N/waste-classification?label=Latest)](https://github.com/Stenberg-N/waste-classification/releases)

A waste classifier app using DenseNet201 to make predictions of users' uploaded images. The model was trained with the TrashNet dataset by [garythung](https://github.com/garythung) and [Mindy Yang](https://github.com/yangmindy4). This was built to gain understanding of CNN (Convolutional Neural Network) machine learning models and also gain experience with them. The trained weights of this app's DenseNet201 model got an accuracy of **97.27**% from model tests.

## Screenshots
### Dark theme
<img width="939" height="1126" alt="darktheme" src="https://github.com/user-attachments/assets/abe095f1-9044-4d0c-84c2-6e1aecd5483f" />

### Light theme
<img width="937" height="1127" alt="lighttheme" src="https://github.com/user-attachments/assets/76396e6e-3dc1-41d0-8c83-0d58cf130178" />

### Confusion matrix with heatmap from test run
<img width="1390" height="1142" alt="heatmap" src="https://github.com/user-attachments/assets/f0f4652e-cf2c-498e-8814-bb67ffc0b5f8" />
Test accuracy: 97.27% | Brief explanation how this works: when the X and Y axis match on the same label, the model got the prediction right.

### Example predictions of unseen data
#### Cardboard
<img width="716" height="1045" alt="cardboardexample" src="https://github.com/user-attachments/assets/d331b71d-1e69-4ed7-a6b2-735a35995f0e" />

#### Metal
<img width="715" height="1048" alt="metalexample" src="https://github.com/user-attachments/assets/9e051ff9-65f9-411a-a68c-37a8fcd81bc2" />

#### Glass
<img width="715" height="1039" alt="glassexample" src="https://github.com/user-attachments/assets/6a0405d3-8a41-4e00-9d0b-fe8b0db0669a" />

#### Plastic
<img width="716" height="1041" alt="plasticexample" src="https://github.com/user-attachments/assets/b9e2c030-25c8-4cd9-8305-beb8e398979c" />

#### Paper
<img width="715" height="1045" alt="paperexample" src="https://github.com/user-attachments/assets/b12fb7df-f1b9-4890-9e5d-8ecbbd8f5e87" />

## Installation
### Repository
1. Clone the repository:<br><br>
    ```text
    git clone https://github.com/Stenberg-N/waste-classification.git
    cd waste-classification
    ```
2. Set up a virtual environment:<br><br>
    ```text
    python -m venv venv
    venv/Scripts/activate
    ```
3. Install dependencies:<br><br>
    ```text
    pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu126
    pip install -r requirements.txt
    ```
4. Download the dataset:  
    Go to [garythung's trashnet repo on Hugging Face](https://huggingface.co/datasets/garythung/trashnet/blob/main/dataset-original.zip) and press download.
5. Set up the dataset:  
    Place the zip in the raw data folder inside the root directory at **waste-classification --> data --> raw** (i.e. waste-classification\data\raw) and extract it there.
6. Downloading a model or training it yourself:  
    If you want to tweak the parameters or add/remove parts and train the model yourself, you can train it by running this command from the root:<br><br>
    ```text
    python -m src.train
    ```
    If you want an already trained model, you can go [here](https://huggingface.co/Stenberg-N/waste-classification-model/tree/main) and download either of the models; stage2_best.pth is trained with MobileNetV4 Hybrid Medium (i.e. mobilenetv4_hybrid_medium.e500_r224_in1k). Note! The name of the model **needs** to be **stage2_best.pth**, so if you download the DenseNet201 model, remove the **DenseNet201_** from its name.  
7. Place the model in the models directory:  
    If you downloaded the trained model, you need to place it inside the models\ directory. Note! The name **must** be stage2_best.pth
8. Launch the app:<br><br>
    ```text
    python -m src.app
    ```

### Application
1. Go to [releases](https://github.com/Stenberg-N/waste-classification/releases) and download the WasteClassifier-vX.X.zip
2. Extract the zip
3. Launch the executable

## Extras
You can view logs of your model when training it with TensorBoard:
```text
tensorboard --logdir=logs/
```
There is also an Optuna dataloader and trainer if you desire to fine-tune your model or find the best parameters to use etc. You can run the Optuna trainer by running the command:
```text
python -m src.trainOptuna
```
To test the model, you can run the command:
```text
python -c "from src.evaluate import evaluate_test_set; evaluate_test_set(checkpoint_path='models/stage2_best.pth')"
```

## Technologies
- **GUI**: PyQt6
- **Machine Learning**: PyTorch, Scikit-learn, Optuna
- **Packaging**: PyInstaller

## Acknowledgments
Thanks to [garythung](https://github.com/garythung) and [Mindy Yang](https://github.com/yangmindy4) for the TrashNet dataset!

## License
This project is licensed under the GNU General Public License Version 3 (GNU GPLv3). See the LICENSE file for details.
