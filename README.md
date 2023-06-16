# Project Title - Venture Funding with Deep Learning


## Introduction
As a risk management associate at Alphabet Soup, a venture capital firm, the primary objective of our team is to create a model that predicts whether applicants will be successful if funded by Alphabet Soup.  


## Technologies
**Libraries**

`python`<br>
`pandas`<br>
`tensorflow`<br>
`keras`<br>
`sklearn`<br>


## Installation Guide
```
# Run Jupyter Notebook in Google Colab 
```
[GC_venture_funding_with_deep_learning.ipynb](https://colab.research.google.com/github/AlexanderValenzuela/Venture_Funding_with_Deep_Learning/blob/main/GC_venture_funding_with_deep_learning.ipynb)
```

# Import the required libraries and dependencies
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder

# Clone this repository on your local machine
git clone https://github.com/AlexanderValenzuela/Venture_Funding_with_Deep_Learning

# Activate conda environment
conda activate dev

# Install the required libraries
pip install -U scikit-learn
pip install --upgrade tensorflow

# Verify installation of TensorFlow.  The output of this command should show version 2.0.0 or higher. 
python -c "import tensorflow as tf;print(tf.__version__)"

# Verify installation of Keras.  Keras is now included with TensorFlow 2.0.  The output should show version 2.2.4-tf or later. 
python -c "import tensorflow as tf;print(tf.keras.__version__)"

# In the conda environment, run Jupyter Lab
jupyter lab 

# Browse to the directory, Venture_Funding_with_Deep_Learning, and launch `GC_venture_funding_with_deep_learning.ipynb`
```
---


## Usage

The steps for this project are broken out into the following three sections:<br>

#### 1. Prepare the Data for Use on a Neural Network Model
Using your knowledge of Pandas and scikit-learn’s `StandardScaler()`, preprocess the dataset so that you can use it to compile and evaluate the neural network model later. Complete the following data preparation steps below:<br>

- Read the `applicants_data.csv file` into a Pandas DataFrame. Review the DataFrame, looking for categorical variables that will need to be encoded, as well as columns that could eventually define your features and target variables.<br>
- Drop the “EIN” (Employer Identification Number) and “NAME” columns from the DataFrame, because they are not relevant to the binary classification model.<br>
- Encode the dataset’s categorical variables using `OneHotEncoder`, and then place the encoded variables into a new DataFrame.<br>
- Add the original DataFrame’s numerical variables to the DataFrame containing the encoded variables.<br>

To complete this step, you will employ the Pandas `concat()` function.<br>

- Using the preprocessed data, create the features (`X`) and target (`y`) datasets. The target dataset should be defined by the preprocessed DataFrame column “IS_SUCCESSFUL”. The remaining columns should define the features dataset.
- Split the features and target sets into training and testing datasets.
- Use scikit-learn's `StandardScaler` to scale the features data.

#### 2. Compile and Evaluate a Binary Classification Model Using a Neural Network
- Create a deep neural network by assigning the number of input features, the number of layers, and the number of neurons on each layer using Tensorflow’s Keras.<br>
- Compile and fit the model using the `binary_crossentropy` loss function, the `adam` optimizer, and the `accuracy` evaluation metric.<br>
- Evaluate the model using the test data to determine the model’s loss and accuracy.<br>
- Save and export the model to an HDF5 file, and name the file `AlphabetSoup.h5`.

#### 3. Optimize the Neural Network Model
- Define at least two new deep neural network models (resulting in the original model, plus two optimization attempts). With each, try to improve on the first model’s predictive accuracy.<br>
- After finishing the models, display the accuracy scores achieved by each model, and compare the results.<br>
- Save each of the models as an HDF5 file.


## Contributors
- Firas Obeid, UC Berkeley FinTech Instructor
[GitHub](<https://github.com/firobeid>)
- Zubair Shaikh, UC Berkeley FinTech Tutoring Team
- Alexander Valenzuela<br>
[LinkedIn Profile](<https://www.linkedin.com/in/alex-valenzuela-97826842/>)


## License
Licensed under the MIT License

