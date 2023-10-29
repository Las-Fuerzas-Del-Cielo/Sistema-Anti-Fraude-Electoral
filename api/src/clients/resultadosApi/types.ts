import {EleccionTipo, RecuentoTipo} from "./enums";

interface GetResultadosParamsRequest {
    anioEleccion: string
    tipoRecuento: RecuentoTipo
    tipoEleccion: EleccionTipo
    categoriaId: string
    distritoId?: string
    seccionProvincialId?: string
    seccionId?: string
    circuitoId?: string
    mesaId?: string
}

interface EstadoRecuento {
    mesasEsperadas: number
    mesasTotalizadas: number
    mesasTotalizadasPorcentaje: number
    cantidadElectores: number
    cantidadVotantes: number
    participacionPorcentaje: any
}

interface ValoresTotalizadosOtros {
    votosNulos: number
    votosNulosPorcentaje: any
    votosEnBlanco: number
    votosEnBlancoPorcentaje: any
    votosRecurridosComandoImpugnados: number
    votosRecurridosComandoImpugnadosPorcentaje: any
}

interface GetResultadosResponse {
    fechaTotalizacion: string
    estadoRecuento: EstadoRecuento
    valoresTotalizadosPositivos: any[]
    valoresTotalizadosOtros: ValoresTotalizadosOtros
}

export {
    GetResultadosParamsRequest,
    GetResultadosResponse,
}