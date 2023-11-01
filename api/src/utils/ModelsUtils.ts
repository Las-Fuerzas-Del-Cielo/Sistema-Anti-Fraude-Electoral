import {Fiscalizar} from "../types/models";

export const validateFiscalizar: (obj: any) => obj is Fiscalizar = (obj: any): obj is Fiscalizar => {
    return "imagen" in obj &&
        "validado" in obj &&
        "errores" in obj &&
        "observaciones" in obj &&
        typeof obj.imagen === "string" &&
        typeof obj.validado === "boolean" &&
        typeof obj.errores === "boolean" &&
        typeof obj.observaciones === "string"
}