import { Router } from 'express';
import fileUpload from 'express-fileupload';
import { uploadFile } from '../controllers/upload';

const router = Router();

// Middleware para manejar archivos subidos
router.use(fileUpload());

// Ruta para subir archivos
router.post('', uploadFile);

export default router;