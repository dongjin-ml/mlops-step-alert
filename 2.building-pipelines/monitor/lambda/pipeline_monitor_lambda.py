
import os
import boto3
import json
from pprint import pprint

TOPIC_ARN = os.environ['TOPIC_ARN']

def lambda_handler(event, context):
    
    # TODO implement
    
    pprint (event)
    print ("==")
    
    strPipelineArn = event["detail"]["pipelineArn"]
    strStepName = event["detail"]["stepName"]
    strCurrentStepStatus = event["detail"]["currentStepStatus"]
    strFailReasion = event["detail"]["failureReason"]
    strEndTime = event["detail"]["stepEndTime"]
    strMetaData = str(event["detail"]["metadata"])
    
    
    print (f'strPipelineArn: {strPipelineArn}')
    print (f'strStepName: {strStepName}')
    print (f'strMetaData: {strMetaData}')
    print (f'strCurrentStepStatus: {strCurrentStepStatus}')
    print (f'strFailReasion: {strFailReasion}')
    print (f'strEndTime: {strEndTime}')
    
    print ("TOPIC_ARN", TOPIC_ARN)
    
        # Send message to SNS
    MY_SNS_TOPIC_ARN = TOPIC_ARN #'<Topic ARN, SNS - Topics에서 확인, arn:aws:sns:us-east-1:계정:TestTopic >'
    msg = "\n".join(
        [
            f'Pipeline ARN: {strPipelineArn}',
            f'Step Name: {strStepName}',
            f'Job Name: {strMetaData}',
            f'Status: {strCurrentStepStatus}',
            f'Fail Reason: {strFailReasion}',
            f'End time: {strEndTime}'
        ]
    )
    
    sns_client = boto3.client('sns')
    sns_client.publish(
        TopicArn=MY_SNS_TOPIC_ARN,
        Subject='[AWS Notification] Monitor for SageMaker pipleine',
        Message=msg
    )

    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }
