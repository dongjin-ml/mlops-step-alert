# FROM 683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.7-1

# RUN pip install --no-cache-dir --upgrade pip \
#  && pip install --no-cache-dir -U sagemaker boto3
 
# WORKDIR /


FROM 366743142698.dkr.ecr.ap-northeast-2.amazonaws.com/sagemaker-xgboost:1.5-1
RUN pip install shap
RUN pip install sagemaker-inference
RUN pip install sagemaker
RUN pip install seaborn==0.11.2
ENV PATH="/opt/ml/code:${PATH}"
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

# /opt/ml and all subdirectories are utilized by SageMaker, 
# use the /code subdirectory to store your user code.
#COPY train.py /opt/ml/code/train.py

# Defines train.py as script entrypoint
#ENV SAGEMAKER_PROGRAM train.py
ENV PYTHONUNBUFFERED=TRUE