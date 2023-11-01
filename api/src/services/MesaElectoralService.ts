import {generateUniqueId} from "../utils/generateUniqueId";
import {DynamoDBClient, GetItemCommand, PutItemCommand} from "@aws-sdk/client-dynamodb";
import {Mesa} from "../types/models";

const dynamodbClient = new DynamoDBClient({
    endpoint: process.env.DYNAMODB_ENDPOINT,
    region: process.env.AWS_REGION
})

class MesaElectoralService {

    async insertMesa(mesa: Mesa): Promise<boolean> {
        const id = generateUniqueId()
        const params = {
            TableName: process.env.DYNAMODB_TABLE_NAME,
            Item: {
                id: {S: id},
                numero: {S: mesa.numero.toString()},
                escuelaId: {S: mesa.escuelaId},
            }
        }
        const command = new PutItemCommand(params)
        try {
            await dynamodbClient.send(command);
            console.debug("Inserted audit successfully.")
            return true
        } catch (err) {
            console.error("Error while trying to insert Audit.", err)
        }
        return false
    }

    async findMesaByEscuela(escuelaId: string ) {
        const params = {
            TableName: process.env.DYNAMODB_TABLE_NAME,
            Key: {
                escuelaId: {S: escuelaId.toString()},
            }
        }
        const command = new GetItemCommand(params)
        return await dynamodbClient
            .send(command)
            .then( (data)=>{
                //Escuela
                //Mesa
                if (data.Item){
                    const dynamoDBItem = data.Item;
                    const resultado : Mesa[] = [];
                    for (const key in dynamoDBItem) {
                        if (Object.prototype.hasOwnProperty.call(dynamoDBItem, key)) {
                            resultado.push({
                                id: key,
                                escuelaId: dynamoDBItem[key].S!,
                                numero: Number(dynamoDBItem[key].N),
                            });
                        }
                    }
                    return resultado;
                }else{
                    console.log("error!");
                    const resultado : Mesa[] = [];
                    return resultado;
                }
            });
    }

    async findMesaByNumero(numeroDeMesa: number) {

        const params = {
            TableName: process.env.DYNAMODB_TABLE_NAME,
            Key: {
                numero:  {S: numeroDeMesa.toString()}
            }
        }
        const command = new GetItemCommand(params)
        return await dynamodbClient
            .send(command)
            .then( (data)=>{
                //Escuela
                //Mesa
                if (data.Item){
                    const dynamoDBItem = data.Item;
                    const resultado : Mesa[] = [];

                    for (const key in dynamoDBItem) {
                        if (Object.prototype.hasOwnProperty.call(dynamoDBItem, key)) {
                            resultado.push({
                                id: key,
                                escuelaId: dynamoDBItem[key].S!,
                                numero: Number(dynamoDBItem[key].N),
                            });
                        }
                    }
                    return resultado[1];
                }else{
                    console.log("error!");
                }
            });
    }
}
export default new MesaElectoralService();