![GitHub Repo Size](https://img.shields.io/github/repo-size/Las-Fuerzas-Del-Cielo/Sistema-Anti-Fraude-Electoral)

# :lion: Sistema-Anti-Fraude-Electoral :lion:

Este es un proyecto **Open Source** (codigo libre) cuyo objetivo es minimizar y detectar la posibilidad de fraude electoral en las próximas elecciones presidenciales de Argentina, donde se define finalmente quien será presidente.

La intención de crear este sistema es mantener y preservar la democracia y la transparencia al pueblo argentino, mediante soluciones tecnológicas que impacten positivamente en nuestro país.

## Indice
- [:lion: Sistema-Anti-Fraude-Electoral :lion:](#lion-sistema-anti-fraude-electoral-lion)
  - [Indice](#indice)
  - [Objetivos](#objetivos)
  - [Componentes](#componentes)
  - [Repositorios y organización](#repositorios-y-organización)
  - [Tipos de Fraudes](#tipos-de-fraudes)
    - [Sumarización Fraudulenta](#sumarizacion-fraudulenta)
    - [Mesas Inexistentes](#mesas-inexistentes)
    - [Datos de Mesa Correctos, luego Incorrectos](#datos-de-mesa-correctos-luego-incorrectos)
    - [Fiscales Falsos](#fiscales-falsos)
    - [Fiscales Judas](#fiscales-judas)
    - [Ficales Incompetentes](#ficales-incompetentes)
    - [Conteo de Voto Corrupto por falta del Fiscal de Mesa](#conteo-de-voto-corrupto-por-falta-del-fiscal-de-mesa)
  - [Usuarios](#usuarios)
  - [Funcionalidad](#funcionalidad)
  - [Arquitectura del Sistema](#arquitectura-del-sistema)
    - [Componentes Principales](#componentes-principales)
      - [Base de Datos](#base-de-datos)
      - [Servicios de Backend](#servicios-de-backend)
      - [Frontend](#frontend)
      - [Procesos Batch](#procesos-batch)
  - [Responsables](#responsables)
  - [Links de interés](#links-de-interes)
- [Autores](#autores)
- [Contributors](#contributors)

## Objetivos

Los objetivos de este sistema son:

1. Identificar potenciales fraudes electorales.
2. Minimizar la ocurrencia e impacto de los fraudes electorales.
3. Acelerar la detección de los fraudes electorales para evitar la falsa declaración de un ganador con altos niveles de casos sospechosos.

## Componentes
- Frontend fiscales (carga de datos)
- Frontend público (para toda persona que quiera acceder a los datos)
- Backend (API)

## Repositorios y organización
Este repositorio tiene la finalidad de:
- Explicar el proyecto en general
- Alojar el [código para el frontend para fiscales](https://github.com/Las-Fuerzas-Del-Cielo/Sistema-Anti-Fraude-Electoral/tree/main/Frontend/fiscales-app-ts).

Estos repositorios son las partes complementarias del proyecto:
- Frontend público (_TO DO: Incluir link cuando sea creado_)
- [Backend API](https://github.com/Las-Fuerzas-Del-Cielo/api)

## Tipos de Fraudes

¿Qué tipos de fraude se esperan detectar? En una elección como esta, existen muchas formas de hacer fraude si alguien controla el sistema de computos oficial.

Esta es la lista de Tipos de Fraudes que se quieren atacar. Esa es la columna vertebral de este proyecto, de ahó deriva toda la funcionalidad que se va a construir. Si querés, podés codear algo de lo que se necesita para alguno de los tipos de fraudes identificados. Si se te ocurre algun otro tipo de fraude y como minimizarlo, lo podés agregar a la lista y especificar qué habría que hacer para eso. Y si tenés tiempo, energía y amor por la libertad, puedes arrancar programando lo que haga falta en lo que llegan otros y te ayudan. La ayuda nunca faltará.

### Sumarización Fraudulenta

En este caso, el Sistema Oficial de computos tiene cargado correctamente todos los datos, pero aún así el agrupamiento por Provicia/Nación es incorrecto.

### Mesas Inexistentes

En este caso, el Sistema Oficial de computos tiene cargado correctamente todos los datos, pero, adicional a las mesas reales, hay mesas que solo existen en el sistema y que se usan para cambiar los resultados globales.

### Datos de Mesa Correctos luego Incorrectos

El tercer tipo de fraude sería que las mesas cargadas en el Sistema Oficial tienen datos diferentes a las actas de los fiscales. Esto actualmente se puede detectar a mano si el fiscal revisa el Sistema Oficial. El problema es que el sistema oficial pudiera mostrarle los datos correctos por un tiempo y luego cambiarlos mas adelante cuando el fiscal ya los consultó y los vio correctos en su momento. Seguramente un fiscal solo verificaria una vez los datos y si están bien, daría por hecho de que eso luego no cambia más.

### Fiscales Falsos

Estos serían personas que se inscribieron como fiscales solo para quedarse con una mesa o grupo de mesas, y luego no ir o abandonar a último momento cuando es muy difícil conseguir reemplazo y reasignar sus mesas.

El sistema debe tener pre-cargado los datos de los Fiscales de Mesa y su trabajo en las elecciones PASO y GENERALES para que si en ellas se ausentaron sin causa, o de manera sospechosa, no se los vuelva a convocar para el BALLOTAGE.

### Fiscales Judas

Este tipo de fraude se basa en reclutar e inscribir Fiscales de Mesa que en vez de fiscalizar y defender los votos de LLA, en realidad hacen lo opuesto. Las mesas con Fiscales Judas pudieran permitir que se carguen en el Sistema Oficial datos falsos porque la contabilización ya fue corrupta.

El sistema debe permitir a los usuarios del tipo **Auditores Externos** ver los datos de los Fiscales de Mesa para poder investigarlos y así poder inferir si son del tipo Judas o no. Los Fiscales de Mesa serán invitados a proveer información de manera opcional de sus identidades en redes sociales, y a los que lo hagan, sumarán puntos en su reputacion como Fiscal. Del otro lado, un ejército de auditores externos puede investigar con su DNI y esas identidades sociales cuál es el background real de esas personas y, con base a los resultados de estas investigaciones, asignarle a cada fiscal un score que puede usarse en un caso extermo para descalificarlo o al menos para observarlo con precaución.

### Ficales Incompetentes

El sistema debe cubrir el caso de fiscales que simplemente son incompetentes y por ignoracia cargan mal los datos en el sistema. Esto significa que deberán existir mecanismos para excluir datos de este tipo de fiscales o que algún tipo de usuario los pueda sobreescribir basándose, por ejemplo, en las fotografías de las actas.

### Conteo de Voto Corrupto por falta del Fiscal de Mesa

El sistema debe ayudar a administrar los Fiscales de Mesa y los Fiscales Generales, para que por lo menos haya un Fiscal General en cada escuela como mínimo. Los casos donde no hay ni siquiera un Fiscal General por escuela son una perdida gigante de votos. Tambien debe gestionar la asignacion de Fiscales de Mesa, para que ante la escaces, estos sean enviados a los lugares donde más se necesitan. Por ejemplo, a las escuelas donde haya una mayor cantidad de electores combinado con una menor cantidad de Fiscales de Mesa.

## Usuarios

Esta APP tendría estos tipos de usuarios:

1. **Fiscal de Mesa:** El principal usuario de esta APP serían los Fiscales de Mesa de LLA. Serían quienes cargan los datos.

2. **Fiscal General:** Supervisa a los Fiscales de Mesa en una determinada escuela. El fiscal general DEBE tomar fotos de todas las actas de todas las mesas de la escuela al cuál fue asignado. En teoría, siempre está presente al menos un fiscal general, aunque pudiera no haber algún Fiscal de Mesa. Si lo hace y las sube al sistema, podemos tener atrás un ejercito de voluntarios por internet que transcriba los valores de las fotos en registros numéricos y así tener la primera versión de datos en el sistema bien temprano despues del cierre de la votación. Debiera ser una responsabilidad del Fiscal General tomar las fotos de todas las mesas de una escuela, porque de esa manera pudieramos evitar el tipo de fraude **Mesas Inexistentes** en la que en el sistema oficial aparece luego alguna mesa que en la realidad no existió. Si cada uno de nuestros fiscales toma foto de TODAS las mesas fisicamente presentes en cada escuela, pudieramos detectar esas mesas fantasmas.

3. **Delegado del Partido:** Son personas de confianza de LLA que durante la votación pueden ir de una escuela a otra para apoyar y auditar a los Fiscales Generales y a los Fiscales de Mesa.

4. **Auditor Interno:** Serían gente de LLA que analizarían los datos cargados por los Fiscales de Mesa y los compararían con los datos oficiales. La APP trataría de automatizar esa comparación con el objetivo de detectar diferencias y potenciales fraudes.

5. **Auditor Externo:** Sería cualquier persona que complete el proceso de sign up como auditor externo. Este perfil tendría acceso a la funcionalidad necesaria para consultar datos y documentos digitalizados y reportar anomalías que luego serían levantadas por los Auditores Internos. Es importante notar que el fraude se tiene que detectar lo antes posible, porque una vez que se declara a alguien ganador, difícilmente eso se vuelva atrás sin importar la cantidad de denuncias de fraude que haya después. Eso elimina la posibilidad de solo cargar una foto y luego que alguien en algún momento digite los datos de esa foto. En general, la detección de los distintos tipos de fraude debería ser instantánea y, es más, de ser posible, cualquier persona de la población en general debería poder ver el recuento de votos de acuerdo a los fiscales de LLA, incluso antes de que salgan los datos oficiales, cosa de que antes de que se nombre un ganador ya haya una reacción del público general ante un potencial fraude de gran escala.

6. **Público:** Cualquier persona que quiera ver los resultados en línea, de acuerdo a los datos cargados por los Fiscales de Mesa de LLA. También podrán navegar por toda la información disponible en el sistema.

## Funcionalidad

1. **Carga de Datos:**: La APP permitiría a los Fiscales de Mesa cargar los datos de las mesas que supervisan. El sistema acumularía todos los datos en una base de datos.

2. **Reportes para Auditores:**: El sistema generaría distintos tipos de reportes orientados a detectar fraude, basándose en los datos cargados por los Fiscales de Mesa y los datos Oficiales.

3. **Consultas para el Publico:** El sistema permitiría ejecutar diferentes tipos de consultas al público en general.

4. **Mapa / Reporte de Fiscales Trabajando:** El sistema debería permitir saber en línea dónde hay y dónde no hay fiscales, así, a través de las redes, se puede agitar para que la gente vaya a fiscalizar, y sobre todo en los casos más extremos donde, por ejemplo, no hay nadie. Un reporte en línea ordenado por gravedad de dónde hacen falta fiscales con urgencia sería óptimo. Sería más grave en las escuelas con mayor cantidad de electores, donde hay la menor cantidad de fiscales. De ahí, pudieran tomar los datos que estarían actualizados a toda hora durante la votación de dónde es más crítico llamar por las redes a que se refuercen esas escuelas.

5. **Mapa / Info de Bunkers-Sucursales:** El sistema debe de permitir visualizar un mapa o varios mapas, en los cuáles se debería poder identificar dónde ir a buscar boletas personales y así que la gente pueda ir a buscarlas.

## Arquitectura del Sistema

Hay muchas formas de encarar la arquitectura de un sistema como este. Primero, vamos a listar los criterios mas importantes que queremos seguir y, a partir de ellos, vamos a derivar la arquitectura que se pueda construir de estos.

1. El sistema completo se tiene que desarrollar y probar en tiempo récord. Ese es el principal contraint o requerimiento.

2. Necesitamos poner a trabajar mucha gente en paralelo, con la mínima fricción entre ellos. Para esto tenemos que partir el sistema en bloques de casos de uso que van a interactuar entre sí a través de interfaces bien definidas.

3. Tenemos que minimizar el TRUST en cada gente que participe, porque nadie se conoce y nadie sabe quién es quién, y quién pudiera tomar alguna responsabilidad con la intención explicita de no cumplir, entre otras cosas.

4. Tenemos que minimizar el riesgo de que algo falle el día de las elecciones, asi que debemos tener redundancia no solo a nivel de hardware, pero tambien de software.

5. Creemos en OPEN SOURCE, en sistemas PERMISSIONLESS y DECENTRALIZADOS (hasta donde se pueda y sea razonable para este caso). Queremos desarrollar un sistema que no solo cualquiera pueda auditar su código, por ser open source, sino que cualquiera pueda registrarse con cualquiera de sus roles / tipos de usuarios, para que de esta manera y por primera vez en la historia, cualquier persona del mundo, este donde este a traves del internet pueda ayudar a auditar la elección y evitar futuros fraudes.

### Componentes Principales

#### Base de Datos

**Base de Datos Principal**

La base de datos del sistema es en nuestro caso el SINGLE POINT OF FAILURE (salvo que esté replicada). Visualizamos tener al menos una base de datos para la información recogida por los Fiscales de Mesa y los Fiscales Generales, que sera de lectura/escritura y desde ellá se servirá la información para las funcionalidades de esos roles (Fiscales de Mesa, Fiscales Generales).

**Base de Datos Read-Only**

Para consultas del Público en General, o del ejército online de auditores, debido que es difícil estimar la cantidad de usuarios en esos roles en un sistema abierto y permissionless, es posible que tengamos una réplica de la base de datos anterior pero de solo lectura, o una versión in-memory o caché para servir todo tipo de requerimiento de consultas por parte de estos tipos de usuarios.

**Base de Datos de Usuarios**

Estaría separada del resto para que sea construida, mantenida y operada por gente especializada en Seguridad de Sistemas y que nadie ajeno a ese team pueda romper algún funcionamiento.

#### Servicios de Backend

**Backend Principal**

El backend principal será el que tenga la business logic de los casos de uso principales, que son los que corresponden a los Fiscales de Mesa, Fiscales Generales, Delagados del Partido.

**Backend Read Only"

Es posible que tengamos un backend para las operaciones read-only del público en general / auditores externos a LLA. Es posible que este backend trabaje con una replica off line de la Base de Datos Princial, actualizada cada tanto.

**Backend para Logins / Signups / Mantenimiento de Usuarios**

Normalmente esto sería parte del Backend Principal, pero como tenemos tan poco tiempo, pudieramos separar este grupo fucionalidades para que un equipo especializado desarrolle esto sin tocarse en nada con el resto del sistema.

#### Frontend

**UI Web / Mobile para Fiscales**

La UI para los Fiscales debe considerarse de misión crítica. Si ella no funcionara, no tendríamos nada, porque los fiscales son los que cargan los datos que son la base de todas las auditorías que el sistema va a permitir realizar. Basándonos en los criterios antes expuestos de minimizar el riesgo de que algún módulo no esté listo o que no funcione bien, la propuesta es abrir la cancha y que múltiples desarrolladores desarrollen múltiples UIs. Luego publicaríamos los links a las que pasen las pruebas que hagamos y el resto quedarían abandonadas. Cada quien es libre de elegir el framework y tecnologías a usar para su UI para Fiscales, porque cada quien invierte su propio tiempo construyéndola. Todas estas UI se conectarían al Backend principal vía una API predefinida y desde cualquiera de ellas se pudieran realizar los casos de uso definidos/a definir.

Como una extensión del criterio anterior, sería incluso óptimo si cada desarrollador hosteara él mismo en su propio servidor su UI, incluyendo su propio dominio si lo quisiera. Esto haría el sistema más resiliente si hubiera más de una opción. Esto aplica para la siguiente UI también.

Si todas las mesas tuvieran fiscales, estamos hablando de una cantidad de potenciales usuarios de entre 100K y 150K, porque hay más o menos esa cantidad de mesas de votación a nivel nacional.

**UI Web para el público en general/auditores externos**

La UI para el público en general/auditores externos y ideas de funcionalidades misión no-crítica, debería ser una web app. En este caso, la masa potencial de usuarios es tremendamente mayor a la anterior, en el orden de los 30 o 40 millones de personas potencialmente que pudieran querer consultar los resultados como los ve LLA y algún número menor que ese de gente que quiera jugar el rol de auditor externo y controlar lo que el sistema le permita controlar. Permitir que cualquier número de personas entre al sistema a auditar puede ser la clave para que, combinado al uso/denuncias a través de redes sociales de un gran número de personas, se puedan desaconsejar los posibles fraudes que la gente que controla el sistema oficial (que es un sistema cerrado y opaco) pudieran querer hacer.

En este caso, también permitiríamos que cualquier desarrollador pueda crear su propia versión de este sitio para el público en general y auditores externos, en la tecnología que quiera, y luego publicaríamos los links a los sitios que pasen correctamente las pruebas que hagamos. Al mismo tiempo, si hubiera varias versiones del sitio, disminuiríamos la carga individual en cada uno y bajaríamos el riesgo de no tener algo funcionando para el día de las elecciones.

**UI Login/SignUp/Mantenimiento de Usuarios**

Esta sería la UI específica para estos casos de usos, a cargo de gente especializada en Seguridad de Sistemas.

#### Procesos Batch

**Extracción de Datos del Sistema Oficial**

El sistema oficial provee aquí(https://resultados.mininterior.gob.ar/desarrollo) instrucciones de como acceder a ciertos datos del mismos a través de una API. Nosotros debieramos tener un proceso que extraiga dichos datos cada cierto tiempo (5 minutos?) y actualice nuestra base de datos.

Para poder abrir el juego y que varios developers puedan hacer diferentes procesos usando los datos del Sitio Oficial más los datos subidos por los Fiscales de Mesa, es mejor si hay un proceso que solo extraiga los datos del sitio oficial y los graba en nuestra base de datos. Luego de que corra ese proceso cada cierto tiempo, pueden haber (n) procesos, de (n) developers distintos, cada uno buscando detectar algún tipo de fraude diferente.

**Procesos de Detección de Fraudes**

Con los datos cargados por los Fiscales a traves de la mobile app mas los datos extraídos del sistema oficial, el sistema tiene la capicidad de correr múltiples procesos, cada uno especializado en detectar algún tipo de fraude.

Se debe analizar los procesos que se necesitan para detectar los tipos de fraude previamente especificados.

## Responsables

Al mejor estilo Open Source, el que quiera hacerse responsable de alguna parte del sistema, puede auto listarse aquí abajo, modificando este readme vía PR, con mucha responsabilidad.

- Analisis General [@Luis-Fernando-Molina](https://www.github.com/Luis-Fernando-Molina)
- UX/UI [@JoseOrtega02](https://github.com/JoseOrtega02), [quien quiera sumarse]
- FrontEnd [@JoseOrtega02](https://github.com/JoseOrtega02), [quien quiera sumarse]

## Links de interés
- Figma: https://www.figma.com/file/nyWx6CewFyvb3a7y3g1r7W/Libertarios-APP?type=design&node-id=0%3A1&mode=design&t=L4k93Fh2vw4b8yku-1
- trello: https://trello.com/invite/b/3sdCNjhp/ATTI0ee6e921ed507577043c8411266d7206D016745E/libertarios-app-ux-ui-fronted

# Autores

- [@Luis-Fernando-Molina](https://www.github.com/Luis-Fernando-Molina)
- [@switTV](https://www.github.com/switTV)

# Contributors
<a href="https://github.com/Las-Fuerzas-Del-Cielo/Sistema-Anti-Fraude-Electoral/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Las-Fuerzas-Del-Cielo/Sistema-Anti-Fraude-Electoral" height="50"/>
</a>

# Discord para Devs

https://discord.gg/BWDqcpXn

“Cuando la Patria está en peligro todo está permitido, excepto no defenderla”.
