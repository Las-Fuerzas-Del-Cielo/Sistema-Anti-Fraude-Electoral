enum EleccionTipo {
    PASO = '1',
    GENERAL = '2',
    SEGUNDA_VUELTA = '3'
}

enum RecuentoTipo {
    PROVISORIO = '1',
    DEFINITIVO = '2',
}

enum PadronTipo {
    NORMAL = '1',
    COMANDO = '2',
    PRIVADOS_DE_LA_LIBERTAD = '3',
    RESIDENTES_EN_EL_EXTERIOR = '4',
}

enum MesaTipo {
    NATIVOS = '1',
    EXTRANJEROS = '2',
}

enum VotosTipo {
    POSITIVO = '1',
    EN_BLANCO = '2',
    IMPUGNADO = '3',
    RECURRIDO = '4',
    NULO = '5',
    COMANDO = '6',
}

enum Provincia {
    CABA = '1',
    BUENOS_AIRES = '2',
    CATAMARCA = '3',
    CORDOBA = '4',
    CORRIENTES = '5',
    CHACO = '6',
    CHUBUT = '7',
    ENTRE_RIOS = '8',
    FORMOSA = '9',
    JUJUY = '10',
    LA_PAMPA = '11',
    LA_RIOJA = '12',
    MENDOZA = '13',
    MISIONES = '14',
    NEUQUEN = '15',
    RIO_NEGRO = '16',
    SALTA = '17',
    SAN_JUAN = '18',
    SAN_LUIS = '19',
    SANTA_CRUZ = '20',
    SANTA_FE = '21',
    SANTIAGO_DEL_ESTERO = '22',
    TUCUMAN = '23',
    TIERRA_DEL_FUEGO = '24'
}

export {
    EleccionTipo,
    RecuentoTipo,
    PadronTipo,
    MesaTipo,
    VotosTipo,
    Provincia,
}