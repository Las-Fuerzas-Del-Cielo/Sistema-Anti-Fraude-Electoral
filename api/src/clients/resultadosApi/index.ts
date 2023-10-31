import axios, { AxiosResponse } from "axios";

import {RESULTADOS_MININTERIOR_GOB_AR_API_RESULTADOS_URL} from "../../constants";
import {GetResultadosParamsRequest, GetResultadosResponse} from "./types";
import {EleccionTipo, MesaTipo, PadronTipo, RecuentoTipo, VotosTipo, Provincia} from './enums';

export interface IResultadosApi {
    getResultados(params: GetResultadosParamsRequest): Promise<GetResultadosResponse>;
}

class ResultadosApi implements IResultadosApi {
    private readonly baseUrl: string = RESULTADOS_MININTERIOR_GOB_AR_API_RESULTADOS_URL;

    async getResultados(params: GetResultadosParamsRequest): Promise<GetResultadosResponse> {
        const response: AxiosResponse<GetResultadosResponse> =
            await axios.get(this.baseUrl+'/getResultados', {params});

        return response.data;
    }
}

export {
    ResultadosApi,
    GetResultadosParamsRequest,
    GetResultadosResponse,
    EleccionTipo,
    RecuentoTipo,
    PadronTipo,
    MesaTipo,
    VotosTipo,
    Provincia,
}