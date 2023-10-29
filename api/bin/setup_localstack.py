import boto3

# Initialize clients for S3, SQS, and DynamoDB connecting to LocalStack
s3 = boto3.client('s3', endpoint_url='http://localhost:4566')
sqs = boto3.client('sqs', endpoint_url='http://localhost:4566')
dynamodb = boto3.client('dynamodb', endpoint_url='http://localhost:4566')

s3.create_bucket(Bucket='lla-reports')

sqs.create_queue(QueueName='process-telegrams')

dynamodb.create_table(
    TableName='FraudPrevention',
    KeySchema=[
        {'AttributeName': 'id', 'KeyType': 'HASH'}
    ],
    AttributeDefinitions=[
        {'AttributeName': 'id', 'AttributeType': 'S'}
    ],
    ProvisionedThroughput={
        'ReadCapacityUnits': 5,
        'WriteCapacityUnits': 5
    }
)

dynamodb.create_table(
    TableName='users',
    KeySchema=[
        {'AttributeName': 'id', 'KeyType': 'HASH'}
    ],
    AttributeDefinitions=[
        {'AttributeName': 'id', 'AttributeType': 'S'},
        {'AttributeName': 'dni', 'AttributeType': 'S'} 
    ],
    ProvisionedThroughput={
        'ReadCapacityUnits': 5,
        'WriteCapacityUnits': 5
    },
    GlobalSecondaryIndexes=[
        {
            'IndexName': 'DniIndex',
            'KeySchema': [
                {'AttributeName': 'dni', 'KeyType': 'HASH'}
            ],
            'Projection': {
                'ProjectionType': 'ALL'
            },
            'ProvisionedThroughput': {
                'ReadCapacityUnits': 5,
                'WriteCapacityUnits': 5
            }
        }
    ]
)

print("Infrastructure set up successfully! 🎉")
