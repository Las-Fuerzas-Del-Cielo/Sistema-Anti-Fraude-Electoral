import {Audit} from '../types/models';
import {generateUniqueId} from '../utils/generateUniqueId';
import {DynamoDBClient, PutItemCommand} from '@aws-sdk/client-dynamodb';
import config from "../config";
import {PutItemCommandInput} from "@aws-sdk/client-dynamodb/dist-types/commands/PutItemCommand";

const dynamodbClient = new DynamoDBClient({
    endpoint: config.dynamodbEndpoint,
    region: config.awsRegion
})

export async function insertAudit(audit: Audit): Promise<boolean> {
    const id = generateUniqueId()
    const params = {
        TableName: process.env.DYNAMODB_TABLE_NAME,
        Item: {
            id: { S: id },
            imagen: { S: audit.imagen },
            validado: { BOOL: audit.validado ? 1 : 0 },
            errores: { BOOL: audit.errores ? 1 : 0 },
            observaciones: { S: audit.observaciones },
        }
    }
    const command = new PutItemCommand(params as PutItemCommandInput)
    try {
        await dynamodbClient.send(command)
        console.debug('Inserted audit successfully.')
        return true
    } catch (err) {
        console.error('Error while trying to insert Audit.', err)
    }
    return false
}