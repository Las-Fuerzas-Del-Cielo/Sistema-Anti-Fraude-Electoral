import { PutObjectCommand, S3Client } from '@aws-sdk/client-s3';
import { ReporteFaltaFiscal } from 'src/types/models';
import { ResultadoRegistroS3 } from 'src/types/models';
import { ERROR_CODES } from './errorConstants';

// Inicializa el cliente de S3 con las configuraciones necesarias
const s3Client = new S3Client({ region: 'tu-region' });

/**
 * Sube un reporte al bucket S3 especificado.
 * 
 * @param {ReporteFaltaFiscal} reporte - El objeto reporte que quieres subir.
 * @return {Promise<ResultadoRegistroS3>} - El resultado de la operación de subida.
 */
export async function registrarReporteEnS3(reporte: ReporteFaltaFiscal): Promise<ResultadoRegistroS3> {
    const bucketName = 'nombre-de-tu-bucket';
    const objectKey = `reportes/${reporte.fiscalId}-${new Date().getTime()}.json`;

    const comando = new PutObjectCommand({
        Bucket: bucketName,
        Key: objectKey,
        Body: JSON.stringify(reporte),
    });

    try {
        const resultado = await s3Client.send(comando);
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
