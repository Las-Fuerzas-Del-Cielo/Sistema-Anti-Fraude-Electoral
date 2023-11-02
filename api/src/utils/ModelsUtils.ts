import {Audit} from "../types/models";

export const validateAudit: (obj: any) => obj is Audit = (obj: any): obj is Audit => {
    return "imagen" in obj &&
        "validado" in obj &&
        "errores" in obj &&
        "observaciones" in obj &&
        typeof obj.imagen === "string" &&
        typeof obj.validado === "boolean" &&
        typeof obj.errores === "boolean" &&
        typeof obj.observaciones === "string"
}