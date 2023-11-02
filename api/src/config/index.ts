// Global configs

const config = {
  port: process.env.PORT || 3000,
  dynamodbEndpoint: process.env.DYNAMODB_ENDPOINT,
  awsRegion: process.env.AWS_REGION
};

export default config;
