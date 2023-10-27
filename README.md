# Sistema-Anti-Fraude-Electoral

Los objetivos de este sistema son: 

1. Identificar potenciales fraudes electorales.
2. Minimizar su ocurrencia e impacto.
3. Acelerar su deteccion para evitar la falsa declaracion de un ganador con altos niveles de fraude / casos sospechosos.

## Autores

- [@Luis-Fernando-Molina](https://www.github.com/Luis-Fernando-Molina)
- [@switTV](https://www.github.com/switTV)

## Intro

Este es un proyecto Open Source cuyo objetivo es minimizar la posibilidad de fraude electoral en las proximas elecciones presidenciales de Argentina donde se define finalmente quien sera presidente. Entendemos que tener un sistema anti-fraude seria algo muy deseable para una fuerza nueva como lo es LLA, que compite con un partido que actualmente esta en el gobierno y tiene una estructura formada durante muchos anios.

## Tipos de Fraudes 

Que tipos de fraude se esperan detectar? En una eleccion como esta hay muchas formas de hacer fraudes si alguien controla el sistema de computos oficial. 

### Sumarizacion Fraudulenta

En este caso el Sistema Oficial de computos tiene cargado correctamente todos los datos pero aun asi el agrupamiento por Provicia / Nacion es incorrecto.

### Mesas Inexistentes

En este caso el Sistema Oficial de computos tiene cargado correctamente todos los datos pero adicional a las mesas reales hay mesas que solo existen en el sistema y que se usan para cambiar los resultados globales.

### Datos de Mesa Correctos luego Incorrectos

El tercer tipo de fraude seria que las mesas cargadas en el Sistema Oficial tienen datos diferentes a las actas de los fiscales. Esto actualmente se puede detectar a mano si el fiscal revisa el Sistema Oficial. El problema es que el sistema oficial pudiera mostrarle los datos correctos por un tiempo y luego cambiarlos mas adelante cuando el fiscal ya los consulto y los vio correctos. Seguramente un fiscal solo verificaria una vez los datos y si estan bien daria por hecho de que eso luego no cambia mas.

### Fiscales Falsos

Estos serian gente que se inscribieron como fiscales solo para quedarse con una mesa o grupo de mesas y luego no ir o abandonar a ultimo momento cuando es muy dificil conseguir reemplazo y reasignar sus mesass.

### Fiscales Judas

Este tipo de fraude se basa en reclutar e inscribir Fiscales de Mesa que en vez de fiscalizar y defender los votos de LLA, en realidad hacen lo opuesto.  Las mesas con fiscales falsos pudieran permitir que se carguen en el Sistema Oficial datos falsos.

El sistema debe permitir a los usuarios del tipo **Auditores Externos** ver los datos de los Fiscales de Mesa para poder investigarlos y asi poder inferir si son judas o no. Los Fiscales de Mesa seran invitados a proveer informacion de manera opcional de sus identidades en redes sociales, y a los que lo hagan, sumaran puntos en su reputacion. Del otro lado, un ejercito de auditores externos puede investigar con su DNI y esas identidades sociales cual es el background real de esas personas y en base a los resultados de estas investigaciones, asignarle a cada fiscal un score que puede usarse en un caso extermo para descalificarlo o al menos para observarlo con precaucion.

### Ficales Incompetentes

El sistema debe cubrir el caso de fiscales que simplemente son incompetentes y por ignoracia cargan mal los datos en el sistema. Esto significa que deberian existir mecanismos para excluir datos de este tipo de fiscales o que algun tipo de usuario los puead sobreescribir basandose por ejemplo en las fotografias de las actas.

### Conteo de Voto Corrupto por falta del Fiscal de Mesa

El sistema debe ayudar a administrar los Fiscales de Mesa y los Fiscales Generales, para que por lo menos haya un Fiscal General en cada escuela como minimo. Los casos donde no hay ni siquiera un Fiscal General por escuela son una perdida gigante de votos. Tambien debe gestionar la asignacion de Fiscales de Mesa, para que ante la escaces, estos sean enviados a los lugares donde mas se necesitan. Por ejemplo, a las escuelas donde haya una mayor cantidad de electores combinado con una menor cantidad de Fiscales de Mesa.

## Usuarios

Esta APP tendria estos tipos de usuarios:

1. **Fiscal de Mesa:** El principal usuario de esta APP serian los Fiscales de Mesa de LLA. Serian quienes cargan los datos.

2. **Fiscal General:** Supervisa a los Fiscales de Mesa en una determinada escuela. El fiscal general DEBE tomar fotos de todas las actas de todas las mesas de la escuala al cual fue asignado. En teoria siempre hay aunque sea un fiscal general, aunque pudiera no haber ningun Fiscal de Mesa. Si lo hace y las sube al sistema, podemos tener atras un ejercito de voluntarios x internet que transcriba los valores de las fotos en registros numericos y asi tener la primera version de datos en el sistema bien temprano despues del cierre de la votacion.
   
Debiera ser una responsabilidad del Fiscal General tomar las fotos de todas las mesas de una escuela, porque de esa manera pudieramos evitar el tipo de fraude **Mesas Inexistentes** en la que en el sistema oficial aparece luego alguna mesa que en la realidad no existio. Si cada uno de nuestros fiscales toma foto de TODAS las mesas fisicamente presentes en cada escuela, pudieramos detectar esas mesas fantasmas.

3. **Delegado del Partido:** Son personas de confianza de LLA que durante la votacion puden ir de una escuela a otra para apoyar y auditar a los Fiscales Generales y a los Fiscales de Mesa.
   
4. **Auditor Interno:** Serian gente de LLA que analizarian los datos cargados por los Fiscales de Mesa y los compararian con los datos oficiales. La APP trataria de automatizar esa comparacion con el objetivo de detectar diferencias y potenciales fraudes.

5. **Auditor Externo:** Serian cualquier persona que complete el proceso de sign up como auditor externo. Este perfil tendria acceso a la funcionalidad necesaria para consultar datos y documentos digitalizados y reportar anomalias que luego serian levantadas por los Auditores Internos. Es importante notar que el fraude se tiene que detectar lo antes posible porque una vez que se declara a alguien ganador, dificilmente eso se vuelva para atras sin importar la cantidad de denuncias de fraude que haya despues. Eso elimina la posibilidad de solo cargar una foto y luego que alguien en algun momento digite los datos de esa foto. En general la deteccion de los distintos tipos de fraude deberia ser instantanea y es mas, si es posible cualquier persona de la poblacion en general deberia poder ver el recuento de votos de acuerdo a los fiscales de LLA incluso antes que salgan los datos oficiales, cosa de que antes de que se nombre un ganador ya haya una reaccion del publico general ante un potencial fraude de gran escala.

6. **Publico:** Cualquier persona que quiera ver los resultados en linea, de acuerdo a los datos cargados por los Fiscales de Mesa de LLA. Tambien podran navegar por toda la informacion disponible en el sistema.

## Funcionalidad

1. **Carga de Datos:**: La APP permitiria a los Fisacales de Mesa cargar los datos de las mesas que supervisan. El sistema acumularia todos los datos en una base de datos.

2. **Reportes para Auditores:**: El sistema generaria distintos tipo de reportes orientados a detectar fraude, basandose en los datos cargados por los Fiscales de Mesa y los datos Oficiales.

3. **Consultas para el Publico:** El sistema permitiria ejecutar diferentes tipos de consultas al publico en general.
  
4. **Mapa / Reporte de Fiscales Trabajando:** El sistema deberia permitir saber en linea donde hay y donde no hay fiscales asi a traves de las redes se puede agitar para que la gente vaya a fiscalizar ysobre todo en los casos mas extremos donde por ejemplo no hay nadie. Un reporte en linea ordenado por gravedad de donde hacen faltan fiscales con urgencia seria optimo. Seria mas grave en las escualas con mayor cantidad de electores donde hay la menor cantidad de fiscales. De ahi pudieran tomar los datos que estarian actualizados a toda hora durante la votacion de donde es mas critico llamar por las redes a que se refuercen esas escuelas.
   
5. **Mapa / Info de Bunkers-Sucursales:** El sistema debe de permitir visualizar un mapa o varios mapas en los cuales se deberia poder visualizar donde ir a buscar boletas personales y asi que la gente pueda ver donde ir a buscarlas.

## Arquitectura del Sistema

Hay muchas formas de encarar la arquitectura de un sistema como este. Vamos a listar antes que nada cuales son los criterios mas importantes que querriamos seguir y a partir de ellos vamos a derivar la arquitectura que emerja de ahi.

1. El sistema completo se tiene que desarrollar y probar en tiempo record. Ese es el principal contraint.

2. Necesitamos poner a trabajar mucha gente en paralelo, con la minima friccion entre ellos. Para esto tenemos que partir el sistema en bloques de casos de uso que van a interactuar entre si a traves de interfaces bien definidas.

3. Tenemos que minimizar el TRUST en cada gente que participe, porque nadie se conoce y nadie sabe quien es quien, y quien pudiera tomar alguna responsabilidad con la intencion explicita de no cumplir, entre otras cosas.

4. Tenemos que minimizar el riesgo de que algo falle el dia de las elecciones asi que deberiamos tener redundancia no solo a nivel de hardware, pero tambien de software.

5. Creemos en OPEN SOURCE, en sistemas PERMISSIONLESS y DECENTRALIZADOS (hasta donde se pueda y sea razonable para este caso). Queremos desarrollar un sistema que no solo cualquiera pueda auditar su codigo, por ser open source, sino que cualquiera pueda registrarse con cualquiera de sus roles / tipos de usuarios, para que de esta manera y por primera vez en la historia, cualquier persona del mundo, este donde este a traves del internet pueda ayudar a auditar la eleccion y evitar fraudes.

### Componentes Principales

#### Base de Datos

**Base de Datos Principal**

La base de datos del sistema es en nuestro caso el SINGLE POINT OF FAILURE (salvo que este replicada). Visualizamos tener al menos una base de datos para la informacion recogida por los Fiscales de Mesa y los Fiscales Generales, que sera de lectura / escritura y desde ella se servira la informacion para las funcionalidades de esos roles (Fiscales de Mesa, Fiscales Generales).

**Base de Datos Read-Only**

Para consultas del Publico en General, o del ejercito online de auditores, debido que es dificil estimar la cantidad de usuarios en esos roles en un sistema abierto y permissionless, es posible que tengamos una replica de la base de datos anterior pero de solo lectura, o una versio in-memory o cache para servir todo tipo de requerimiento de consultas por parte de estos tipos de usuarios.

**Base de Datos de Usuarios**

Estaria separada del resto para que sea construida, mantenida y operada por gente especializada en Seguridad de Sistemas y que nadie ageno a ese team pueda romper nada aqui.

#### Servicios de Backend

**Backend Principal**

El backend principal sera el que tenga la business logic de los casos de uso principales, que son los que corresponden a los Fiscales de Mesa, Fiscales Generales, Delagados del Partido.

**Backend Read Only"

Es posible que tengamos un backend para las operaciones read-only del publico en general / auditores externos a LLA. Es posible que este backend trabaje con una replica off line de la Base de Datos Princial, actualizada cada tanto.

**Backend para Logins / Signups / Mantenimiento de Usuarios**

Normalmente esto seria parte del Backend Principal, pero como tenemos tan poco tiempo, pudieramos separar este grupo fucionalidades para que un equipo especializado desarrolle esto sin tocarse en nada con el resto del sistema.  

#### Frontend 

**UI Web / Mobile para Fiscales**

La UI para los Fiscales debe considerarse de mision critica. Si ella no funcionara no tendriamos nada, porque los fiscales son los que cargan los datos que son la base de todas las auditorias que el sistema va a permitir realizar. Basandonos en los criterios antes expuestos de minimizar el riesgo de que algun modulo no este listo o que no funcione bien, la propuesta es abrir la cancha y que multiples desarrolladores desarrollen multiples UIs. Luego publicariamos los links a las que pasen las pruebas que hagamos y el resto quedarian abandonadas. Cada quien es libre de elegir el framework y tecnologias a usar para su UI para Fiscales, porque cada quien invierte su propio tiempo construyendola. Todas estas UI se conectarian al Backend principal via una API pre-definida y desde cualquiera de ellas se pudieran realizar los casos de uso definidos / a definir. 

Como una extension del criterio anterior, seria incluso optimo si cada desarrollador hosteara el mismo en su propio servidor su UI inlcuyendo su propio dominio si lo quisiera. Esto haria el sistema mas resiliente si hubiera mas de una opcion. Esto aplica para la siguiente UI tambien.

Si todas las mesas tuvieran fiscales estamos hablando de una cantidad de potenciales usuarios de entre 100K y 150K porque hay mas o menos esa cantidad de mesas de votacion a nivel nacional.

**UI Web para el publico en general / auditores externos**

La UI para el publico en general / auditores externos y ideas de funcionalidades mision no-critica, deberia ser una web app. En este caso la masa potencial de usuarios es tremendamente mayor a la anterior, en el orden de los 30 o 40 millones de personas potencialmente que pudieran querer consultar los resultados como los ve LLA y algun numero menor que ese de gente que quiera jugar el rol de auditor externo y controlar lo que el sistema le permita controlar. Permitir que cualquier numero de personas entre al sistema a auditar puede ser la clave para que combinado al usao / denuncias a traves de redes sociales de un gran numero de personas, se puedan desaconsejar los posibles fraudes que la gente que controla el sistema oficial (que es un sistema cerrado y opaco) pudieran querer hacer.

En este caso tambien permitiriamos que cualquier developer pueda crear su propia version de este sito para el publico en general y auditores externos, en la tecnologia que quiera, y luego publicariamos los links a los sitios que pasen correctamente las pruebas que hagamos. Al mismo tiempo, si hubiera varias versiones del sitio, disminuiriamos la carga individual en cada uno, y bajariamos el riesgo de no tener algo funcionando para el dia de las elecciones. 

**UI Login / Signup / Mantenimiento de Usuarios**

Esta seria la UI especifica para estos casos de usos, a cargo de gente especializada en Seguridad de Sistemas.

#### Procesos Batch

** Extraccion de Datos del Sistema Oficial**

El sistema oficial provee aqui(https://resultados.mininterior.gob.ar/desarrollo) instrucciones de como acceder a ciertos datos del mismos a traves de una API. Nostros debieramos tener un proceso que extraiga dichos datos cada cierto tiempo (5 minutos?) y actualice nuestra base de datos. 

Para poder abrir el juego y que varios developers puedan hacer diferentes procesos usando los datos del Sitio Oficial mas los datos subidos por los Fiscales de Mesa, es mejor si hay un proceso que solo extraiga los datos del sitio oficial y los graba en nuestra base de datos. Luego de que corra ese proceso cada cierto tiempo, pueden haber n procesos, de n developers distintos cada uno buscando detectar algun tipo de fraude diferente. 

** Procesoss de Deteccion de Fraudes**

Con los datos cargados por los Fiscales a traves de la mobile app mas los datos extraidos del sistema oficial, el sistema tiene la capicidad de correr multiples procesos cada uno especializado en detectar algun tipo de fraude.

Se debe analizar los procesos que se necesitan para detectar los tipos de fraude previamente especificados.

## Responsables

Al mejor estilo Open Source, el que quiera hacerse responsable de alguna parte del sistema, puede auto listarse aqui abajo, modificando este readme vi PR

- Analisis General [@Luis-Fernando-Molina](https://www.github.com/Luis-Fernando-Molina)
- UX/UI [@JoseOrtega02](https://github.com/JoseOrtega02), [quien quiera sumarse]
- FrontEnd [@JoseOrtega02](https://github.com/JoseOrtega02), [quien quiera sumarse]

  ## Links de interes
- Figma: https://www.figma.com/file/nyWx6CewFyvb3a7y3g1r7W/Libertarios-APP?type=design&node-id=0%3A1&mode=design&t=L4k93Fh2vw4b8yku-1
- trello: https://trello.com/invite/b/3sdCNjhp/ATTI0ee6e921ed507577043c8411266d7206D016745E/libertarios-app-ux-ui-fronted

