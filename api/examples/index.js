require('dotenv').config()
const fs = require('fs-extra')
const { DynamoDBClient, GetItemCommand, PutItemCommand, ListTablesCommand } = require('@aws-sdk/client-dynamodb')
const { S3Client, PutObjectCommand } = require('@aws-sdk/client-s3')
const { SQSClient, SendMessageCommand } = require('@aws-sdk/client-sqs')
const endpoint = process.env.AWS_ENDPOINT
const dynamodbClient = new DynamoDBClient({ endpoint })
const s3Client = new S3Client({ endpoint })
const sqsClient = new SQSClient({ endpoint })


async function insertItem(id, description) {
  const params = {
    TableName: process.env.DYNAMODB_TABLE_NAME,
    Item: {
      'id': { S: id },
      'description': { S: description }
    }
  }
  const command = new PutItemCommand(params)
  const result = await dynamodbClient.send(command)
  console.log(`Item inserted:`, result)
}

async function fetchItemById(id) {
  const params = {
    TableName: process.env.DYNAMODB_TABLE_NAME,
    Key: {
      'id': { S: id }
    }
  }
  const command = new GetItemCommand(params)
  const result = await dynamodbClient.send(command)
  console.log(`fetchItemById  result:`, result)
}

async function uploadToS3(Key, filePath) {
  const fileBuffer = await fs.readFile(filePath, 'base64')
 
  const command = new PutObjectCommand({ Key, Body: fileBuffer, Bucket: 'lla-reports' })
  const res = await s3Client.send(command)
  console.log(`uploadToS3  res:`, res)
}

async function sendMessageToSQS(MessageBody) {
  const command = new SendMessageCommand({ QueueUrl: process.env.QUEUE_URL, MessageBody })
  const res = await sqsClient.send(command)
  console.log(`sendMessageToSQS  res:`, res)
}

insertItem('12345', 'This is a test item').then(console.log).catch(console.log)
fetchItemById('1234').then(console.log).catch(console.log)
uploadToS3('assets', 'index.js').then(console.log).catch(console.log)
sendMessageToSQS(JSON.stringify({ message: 'test message' }))
  .then(console.log)
  .catch(console.log)
