"""
Lambda Function: data-trigger
Triggers daily pipeline on EC2 via AWS Systems Manager (SSM)
Triggered by: EventBridge (daily 2 AM SGT)
"""

import boto3
import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

ssm_client = boto3.client('ssm')
sns_client = boto3.client('sns')

# Configuration
EC2_INSTANCE_ID = 'i-xxxxxxxxxxxxx'  
SNS_TOPIC_ARN = 'arn:aws:sns:ap-southeast-1:ACCOUNT_ID:hdb-pipeline-alerts'


def lambda_handler(event, context):
    try:
        logger.info("Triggering daily HDB prediction pipeline")
        
        response = ssm_client.send_command(
            InstanceIds=[EC2_INSTANCE_ID],
            DocumentName='AWS-RunShellScript',
            Parameters={
                'commands': [
                    'bash /home/ubuntu/pipeline/daily_pipeline.sh'
                ]
            },
            TimeoutSeconds=1800,
            Comment='Daily HDB prediction pipeline'
        )
        
        command_id = response['Command']['CommandId']
        
        logger.info(f"Command sent successfully. Command ID: {command_id}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Pipeline triggered successfully',
                'command_id': command_id
            })
        }
        
    except Exception as e:
        logger.error(f"Error triggering pipeline: {e}", exc_info=True)
        
        # Send SNS alert
        try:
            sns_client.publish(
                TopicArn=SNS_TOPIC_ARN,
                Subject='HDB Pipeline Trigger Failed',
                Message=f'Failed to trigger daily pipeline: {str(e)}'
            )
        except:
            pass
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'message': 'Pipeline trigger failed',
                'error': str(e)
            })
        }