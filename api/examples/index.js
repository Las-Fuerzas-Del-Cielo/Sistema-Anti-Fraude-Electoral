require('dotenv').config()
const fs = require('fs-extra')
const path = require('path')
const { DynamoDBClient, GetItemCommand, PutItemCommand, ListTablesCommand } = require('@aws-sdk/client-dynamodb')
const { S3Client, PutObjectCommand } = require('@aws-sdk/client-s3')
const { SQSClient, SendMessageCommand } = require('@aws-sdk/client-sqs')

const awsEndpoint = process.env.AWS_ENDPOINT
const dynamodbClient = new DynamoDBClient({ endpoint: awsEndpoint })
const s3Client = new S3Client({ endpoint: awsEndpoint })
const sqsClient = new SQSClient({ endpoint: awsEndpoint })

async function insertDBItem(id, description) {
  const params = {
    TableName: process.env.DYNAMODB_TABLE_NAME,
    Item: {
      'id': { S: id },
      'description': { S: description }
    }
  }
  const command = new PutItemCommand(params)
  const result = await dynamodbClient.send(command)
  console.log('DynamoDB insertItem success')
}

async function fetchDBItemById(id) {
  const params = {
    TableName: process.env.DYNAMODB_TABLE_NAME,
    Key: {
      'id': { S: id }
    }
  }
  const command = new GetItemCommand(params)
  const result = await dynamodbClient.send(command)
  console.log(`DynamoDB fetchItemById success`)
}

async function uploadToS3(Key, filePath) {
  const file = await fs.readFile(path.resolve(__dirname, filePath))
  const command = new PutObjectCommand({
    Key,
    Body: file,
    Bucket: process.env.S3_REPORTS_BUCKET,
    ContentType: 'text/plain'
  })
  const res = await s3Client.send(command)
  console.log(`uploadToS3 success`)
}

async function sendMessageToSQS(MessageBody) {
  const command = new SendMessageCommand({ QueueUrl: process.env.QUEUE_URL, MessageBody })
  const res = await sqsClient.send(command)
  console.log(`sendMessageToSQS message sent`)
}

// test AWS services
insertDBItem('12345', 'This is a test item').catch(console.log)
fetchDBItemById('1234').catch(console.log)
uploadToS3('assets', 'README.md').catch(console.log)
sendMessageToSQS(JSON.stringify({ message: 'test message' })).catch(console.log)
