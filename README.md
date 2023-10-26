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

## Usuarios

Esta APP tendria 4 tipos de usuarios:

1. **Fiscal de Mesa:** El principal usuario de esta APP serian los Fiscales de Mesa de LLA. Serian quienes cargan los datos.

2. **Fiscal General:** Supervisa a los Fiscales de Mesa en una determinada escuela.

4. **Delegado del Partido:** Son personas de confianza de LLA que durante la votacion puden ir de una escuela a otra para apoyar y auditar a los Fiscales Generales y a los Fiscales de Mesa.
   
5. **Auditores:** Serian gente de LLA que analizarian los datos cargados por los Fiscales de Mesa y los compararian con los datos oficiales. La APP trataria de automatizar esa comparacion con el objetivo de detectar diferencias y potenciales fraudes.

6. **Publico:** Cualquier persona que quiera ver los resultados en linea, de acuerdo a los datos cargados por los Fiscales de Mesa de LLA. Es importante notar que el fraude se tiene que detectar lo antes posible porque una vez que se declara a alguien ganador, dificilmente eso se vuelva para atras sin importar la cantidad de denuncias de fraude que haya despues. Eso elimina la posibilidad de solo cargar una foto y luego que alguien en algun momento digite los datos de esa foto. En general la deteccion de los distintos tipos de fraude deberia ser instantanea y es mas, si es posible cualquier persona de la poblacion en general deberia poder ver el recuento de votos de acuerdo a los fiscales de LLA incluso antes que salgan los datos oficiales, cosa de que antes de que se nombre un ganador ya haya una reaccion del publico general ante un potencial fraude de gran escala.

## Funcionalidad

1. **Carga de Datos:**: La APP permitiria a los Fisacales de Mesa cargar los datos de las mesas que supervisan. El sistema acumularia todos los datos en una base de datos.

2. **Reportes para Auditores:**: El sistema generaria distintos tipo de reportes orientados a detectar fraude, basandose en los datos cargados por los Fiscales de Mesa y los datos Oficiales.

3. **Consultas para el Publico:** El sistema permitiria ejecutar diferentes tipos de consultas al publico en general. NOTA: solo si da el tiempo para desarrolar esto.
  
4. **Mapa / Reporte de Fiscales Trabajando:** El sistema deberia permitir saber en linea donde hay y donde no hay fiscales asi a traves de las redes se puede agitar para que la gente vaya a fiscalizar ysobre todo en los casos mas extremos donde por ejemplo no hay nadie. Un reporte en linea ordenado por gravedad de donde hacen faltan fiscales con urgencia seria optimo. Seria mas grave en las escualas con mayor cantidad de electores donde hay la menor cantidad de fiscales. De ahi pudieran tomar los datos que estarian actualizados a toda hora durante la votacion de donde es mas critico llamar por las redes a que se refuercen esas escuelas.

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

### Ficales Incompetentes

El sistema debe cubrir el caso de fiscales que simplemente son incompetentes y por ignoracia cargan mal los datos en el sistema. Esto significa que deberian existir mecanismos para excluir datos de este tipo de fiscales o que algun tipo de usuario los puead sobreescribir basandose por ejemplo en las fotografias de las actas.
