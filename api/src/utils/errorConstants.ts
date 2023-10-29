export const ERROR_CODES = {
    INCOMPLETE_DATA: {
      status: 400,
      message: 'Incomplete data. Please fill in all required fields.'
    },
    UNAUTHORIZED_GENERAL: {
      status: 403,
      message: 'Access denied. General fiscal role required for this action.'
    },
    RESOURCE_NOT_FOUND: {
      status: 404,
      message: 'Resource not found. Please verify the provided data.'
    },
    INSTITUTION_NOT_FOUND: {
        status: 404,
        message: "Fiscal's institution not found."
    },
    INVALID_MESA_OR_ESCUELA: {
      status: 404,
      message: 'Invalid mesa or school. Verify that the mesa belongs to the indicated school.'
    },
    INTERNAL_SERVER_ERROR: {
      status: 500,
      message: 'Internal server error. Please try again later.'
    },
    S3_UPLOAD_ERROR: {
        status: 500,
        message: "Error registering the report in S3."
    },
    // Add more errors as needed
};
