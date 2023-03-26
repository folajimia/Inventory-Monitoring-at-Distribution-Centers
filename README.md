# Inventory Monitoring at Distribution Centers - Counting Objects in Amazon Bin Image Dataset using AWS SageMaker

As part of Udacity's Nano-Degree Program | Capstone Project

This project aims to demonstrate how to use Amazon SageMaker to develop and deploy a machine learning model for counting the number of objects in an image of a bin from the Amazon Bin Image Dataset. The project is focused on how to use Amazon SageMaker to perform various tasks such as dataset preparation, hyperparameter tuning, model training, model evaluation, and model deployment.

**Note:** _Note that the goal of this project is to demonstrate the use of Amazon SageMaker, and not to create an accurate machine learning model_.

### Dependencies

```
Python 3.7
PyTorch >=3.6
boto3 1.26.96
IPython 7.34.0
jupyter_client 7.4.8
jupyter_core 4.12.0
jupyterlab 3.5.2
notebook 6.5.2
sagemaker 2.140.1
sklearn 0.22.1
tqdm 4.42.1
```

### Installation

For this project, it is highly recommended to use Sagemaker Studio from the course provided AWS workspace. This will simplify much of the installation needed to get started.

For local development, you will need to setup a jupyter lab instance.

- Follow the [jupyter install](https://jupyter.org/install.html) link for best practices to install and start a jupyter lab instance.
- If you have a python virtual environment already installed you can just `pip` install it.

```
pip install jupyterlab
```

- There are also docker containers containing jupyter lab from [Jupyter Docker Stacks](https://jupyter-docker-stacks.readthedocs.io/en/latest/index.html).

## File Structure

- `sagemaker.ipynb`: Jupyter notebook that demonstrates how to use Amazon SageMaker to preprocess the dataset, train the model, and deploy the endpoint.

- `hpo.py`: Python script that is used as the entry point for an Amazon SageMaker Hyperparameter Tuning Job. This script defines the hyperparameter search job configuration.

- `train.py`: Python script that defines the model architecture, training loop, and evaluation metrics. This script is used to train the ResNet50 model using PyTorch and Amazon SageMaker PyTorch training.

- `inference.py`: Python script that defines the endpoint prediction function. This script is used to generate predictions from the deployed endpoint.

- `ProfilerReport.zip`: Zip file that contains the profiler report generated by Amazon SageMaker Profiler.

## Project Set Up and Installation

This Project was carried out using Amazon SageMaker Studio. To set up this project, you need to have access to Amazon SageMaker. You can create a SageMaker notebook instance or use the SageMaker Studio. The project can also be run locally. To run it locally, you will need to install the AWS CLI and configure it to use SageMaker. Here are the steps to do that:

1. Install the AWS CLI by following the instructions in the AWS CLI User Guide.

2. Once the CLI is installed, open a terminal or command prompt and run aws configure to set up your credentials. You will need to provide your Access Key ID, Secret Access Key, default region name, and default output format. You can find your Access Key ID and Secret Access Key in the IAM console.

3. Next, you can clone this repository and navigate to the project directory:

   ```
   git clone https://github.com/your-username/your-repository.git
   cd your-repository
   ```

4. Create a virtual environment and activate it:

   ```
   python3 -m venv env
   source env/bin/activate
   ```

5. Install the required Dependencies:

   see Dependencies section above

6. To run the Jupyter notebook locally, run the following command:

   `jupyter notebook`

This will open a new browser window with the Jupyter notebook interface. You can then navigate to the `sagemaker.ipynb` notebook and run it.

Note that if you plan to use Amazon SageMaker for this project, you don't need to install the AWS CLI or configure it. Instead, you can follow the instructions in the notebook to create a SageMaker notebook instance and run the code directly in the notebook.

## Dataset

### Overview

The Amazon Bin Image Dataset contains images of bins containing various objects. The dataset is available for public use and can be accessed from the Amazon S3 bucket [here](https://registry.opendata.aws/amazon-bin-imagery/).

### Access

The dataset can be accessed using the boto3 python library to interact with the S3 bucket. The **SageMaker Processing Job** was used in this project to perform ETL on the dataset.

### Subset

The model in this project is using 10% of the original dataset. This is because the Amazon Bin Image Dataset is quite large, hence the use of a subset of the data to conserve AWS costs.

## Model Training

This project applied transfer learning to improve the accuracy of the model. The ResNet50 model was used as the base model and fine-tuned on the Amazon Bin Image Dataset.

## License

[License](LICENSE.txt)
