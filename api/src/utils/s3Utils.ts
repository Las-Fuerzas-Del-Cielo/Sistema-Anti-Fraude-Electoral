import { PutObjectCommand, PutObjectCommandOutput, S3Client } from '@aws-sdk/client-s3';
import { ReportFaltaFiscal } from '../types/models';
import { ResultadoRegistroS3 } from '../types/models';
import { ERROR_CODES } from './errorConstants';

type S3Config = {
  region: string;
}

type S3UploadParams = {
  Bucket: string;
  Key: string;
  Body: string;
}

// Inicializa el cliente de S3 con las configuraciones necesarias
const s3Config: S3Config = { region: 'tu-region' };
const s3Client: S3Client = new S3Client(s3Config);

/**
 * Sube un reporte al bucket S3 especificado.
 * 
 * @param {ReportFaltaFiscal} reporte - El objeto reporte que quieres subir.
 * @return {Promise<ResultadoRegistroS3>} - El resultado de la operación de subida.
 */
export async function registrarReporteEnS3(reporte: ReportFaltaFiscal): Promise<ResultadoRegistroS3> {
    const bucketName: string = 'nombre-de-tu-bucket';
    const objectKey: string = `reportes/${reporte.fiscalId}-${new Date().getTime()}.json`;

    const uploadParams: S3UploadParams = {
      Bucket: bucketName,
      Key: objectKey,
      Body: JSON.stringify(reporte),
    };

    const comando: PutObjectCommand = new PutObjectCommand(uploadParams);

    try {
        const resultado: PutObjectCommandOutput = await s3Client.send(comando);
        // Devuelve la información relevante sobre la operación de subida
        return { 
          key: objectKey, 
          bucket: bucketName, 
          status: 'Subido con éxito',
          resultadoMetadata: resultado.$metadata
        };
    } catch (error) {
        console.error('Error al subir a S3:', error);
        // Devuelve un objeto con detalles del error para un mejor manejo
        return {
          error: true,
          mensaje:  ERROR_CODES.S3_UPLOAD_ERROR.message,
          codigoError: ERROR_CODES.S3_UPLOAD_ERROR.status,
          detalles: error.message || 'Error no especificado'
        };
    }
}
