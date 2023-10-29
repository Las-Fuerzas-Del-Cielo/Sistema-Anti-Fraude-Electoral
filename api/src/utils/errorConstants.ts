export const ERROR_CODES = {
    INCOMPLETE_DATA: {
      status: 400,
      message: 'Datos incompletos. Por favor, complete todos los campos requeridos.'
    },
    UNAUTHORIZED_GENERAL: {
      status: 403,
      message: 'Acceso denegado. Se requiere ser fiscal general para realizar esta acción.'
    },
    RESOURCE_NOT_FOUND: {
      status: 404,
      message: 'Recurso no encontrado. Verifique los datos proporcionados.'
    },
    INSTITUTION_NOT_FOUND: {
        status: 404,
        message: "Institución del fiscal no encontrada."
    },
    INVALID_MESA_OR_ESCUELA: {
      status: 404,
      message: 'Mesa o escuela no válida. Verifique que la mesa pertenezca a la escuela indicada.'
    },
    INTERNAL_SERVER_ERROR: {
      status: 500,
      message: 'Error interno del servidor. Inténtelo de nuevo más tarde.'
    },
    S3_UPLOAD_ERROR: {
        status: 500,
        message: "Error al registrar el reporte en S3."
    },
    // Agregar más errores según sea necesario
  };
  