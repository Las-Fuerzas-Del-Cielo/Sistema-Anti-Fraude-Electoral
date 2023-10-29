SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Base de datos: `fiscales_db`
--
-- -----------------------
-- Usuario
-- -----------------------
CREATE TABLE `gen_user_status` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `description` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE `gen_user_level` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `description` varchar(45) NOT NULL,
  `enabled` tinyint(4) NOT NULL DEFAULT '1',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE `gen_user` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `user_name` varchar(100) NOT NULL,
  `password` varchar(100) NOT NULL,
  `firebase_uid` varchar(255) DEFAULT NULL,
  `first_name` varchar(100) NOT NULL,
  `last_name` varchar(100) NOT NULL,
  `national_id` varchar(45) DEFAULT NULL,
  `level_id` int(11) DEFAULT NULL,
  `phone` varchar(45) DEFAULT NULL,
  `contact_email` varchar(100) DEFAULT NULL,
  `avatar` blob,
  `enabled` tinyint(4) NOT NULL DEFAULT '1',
  `create_date` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `create_user_id` int(11) NOT NULL,
  `update_date` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `update_user_id` int(11) NOT NULL,
  `status_id` int(11) NOT NULL DEFAULT '1',
  PRIMARY KEY (`id`),
  UNIQUE KEY `email_UNIQUE` (`user_name`),
  KEY `user_level_idx` (`level_id`),
  KEY `user_sttatus_idx` (`status_id`),
  CONSTRAINT `user_level` FOREIGN KEY (`level_id`) REFERENCES `gen_user_level` (`id`) ON DELETE NO ACTION ON UPDATE NO ACTION,
  CONSTRAINT `user_status` FOREIGN KEY (`status_id`) REFERENCES `gen_user_status` (`id`) ON DELETE NO ACTION ON UPDATE NO ACTION
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- -----------------------
-- Seguridad
-- -----------------------

CREATE TABLE `sec_privilege` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `description` varchar(45) NOT NULL,
  `create_date` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `create_user_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `privilege_user_idx` (`create_user_id`),
  CONSTRAINT `privilege_user` FOREIGN KEY (`create_user_id`) REFERENCES `gen_user` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE `sec_rol` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `description` varchar(45) NOT NULL,
  `create_date` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `create_user_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `rol_user_idx` (`create_user_id`),
  CONSTRAINT `rol_user` FOREIGN KEY (`create_user_id`) REFERENCES `gen_user` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE `sec_rol_privilege` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `rol_id` int(11) NOT NULL,
  `privilege_id` int(11) NOT NULL,
  `create_date` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `create_user_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `Unique` (`rol_id`,`privilege_id`),
  KEY `rol_privilege_privilege_idx` (`privilege_id`),
  KEY `rol_privilege_create_user_idx` (`create_user_id`),
  CONSTRAINT `rol_privilege_create_user` FOREIGN KEY (`create_user_id`) REFERENCES `gen_user` (`id`),
  CONSTRAINT `rol_privilege_privilege` FOREIGN KEY (`privilege_id`) REFERENCES `sec_privilege` (`id`),
  CONSTRAINT `rol_privilege_rol` FOREIGN KEY (`rol_id`) REFERENCES `sec_rol` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;



CREATE TABLE `gen_user_rol` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `user_id` int(11) NOT NULL,
  `rol_id` int(11) NOT NULL,
  `create_date` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `create_user_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `user_rol_user_idx` (`user_id`),
  KEY `user_rol_rol_idx` (`rol_id`),
  KEY `user_rol_create_user_idx` (`create_user_id`),
  CONSTRAINT `user_rol_create_user` FOREIGN KEY (`create_user_id`) REFERENCES `gen_user` (`id`),
  CONSTRAINT `user_rol_rol` FOREIGN KEY (`rol_id`) REFERENCES `sec_rol` (`id`),
  CONSTRAINT `user_rol_user` FOREIGN KEY (`user_id`) REFERENCES `gen_user` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- -----------------------
-- Estructura territorial
-- -----------------------

CREATE TABLE `gen_distrito` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `descripcion` varchar(100) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

	-- Existen distintos modelos entre provincias / BsAs,Mza,Tucuman / CABA en la seccion, pero unificamos todo en seccion para mantener la estructura estandar
	-- -----------------------
CREATE TABLE `gen_seccion` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `distrito_id` int(11) NOT NULL,
  `descripcion` varchar(100) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `seccion_distrito_idx` (`distrito_id`),
  CONSTRAINT `seccion_distrito` FOREIGN KEY (`distrito_id`) REFERENCES `gen_distrito` (`id`) ON DELETE NO ACTION ON UPDATE NO ACTION
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE `gen_circuito` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `seccion_id` int(11) NOT NULL,
  `descripcion` varchar(100) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `circuito_seccion_idx` (`seccion_id`),
  CONSTRAINT `circuito_seccion` FOREIGN KEY (`seccion_id`) REFERENCES `gen_seccion` (`id`) ON DELETE NO ACTION ON UPDATE NO ACTION
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE `gen_local_comicio` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `circuito_id` int(11) NOT NULL,
  `descripcion` varchar(100) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `localcomicio_circuito_idx` (`circuito_id`),
  CONSTRAINT `localcomicio_circuito` FOREIGN KEY (`circuito_id`) REFERENCES `gen_circuito` (`id`) ON DELETE NO ACTION ON UPDATE NO ACTION
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE `gen_mesa` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `local_comicio_id` int(11) NOT NULL,
  `descripcion` varchar(100) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `mesa_localcomicio_idx` (`local_comicio_id`),
  CONSTRAINT `mesa_localcomicio` FOREIGN KEY (`local_comicio_id`) REFERENCES `gen_local_comicio` (`id`) ON DELETE NO ACTION ON UPDATE NO ACTION
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- -----------------------
-- fiscalizcion
-- -----------------------

CREATE TABLE `fis_fiscalizacion_estado` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `descripcion` varchar(100) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE `fis_fiscalizacion` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `estado_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `fiscalizacion_estado_idx` (`estado_id`),
  CONSTRAINT `fiscalizacion_estado` FOREIGN KEY (`estado_id`) REFERENCES `fis_fiscalizacion_estado` (`id`) ON DELETE NO ACTION ON UPDATE NO ACTION
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE `fis_fiscalizacion_mesa_estado` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `descripcion` varchar(100) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE `fis_fiscalizacion_mesa` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `fiscalizacion_id` int(11) NOT NULL,
  `mesa_id` int(11) NOT NULL,
  `total_votantes` int(11) DEFAULT NULL,
  `votos_massarasa` int(11) DEFAULT NULL,
  `votos_milei` int(11) DEFAULT NULL,
  `votos_blanco` int(11) DEFAULT NULL,
  `votos_nulos` int(11) DEFAULT NULL,
  `votos_impugnados_massarasa` int(11) DEFAULT NULL,
  `votos_impugnados_milei` int(11) DEFAULT NULL,
  `estado_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `fiscalizacionmesa_estado_idx` (`estado_id`),
  KEY `fiscalizacionmesa_fiscalizacion_idx` (`fiscalizacion_id`),
  KEY `fiscalizacionmesa_mesa_idx` (`mesa_id`),
  CONSTRAINT `fiscalizacionmesa_estado` FOREIGN KEY (`estado_id`) REFERENCES `fis_fiscalizacion_mesa_estado` (`id`) ON DELETE NO ACTION ON UPDATE NO ACTION,
  CONSTRAINT `fiscalizacionmesa_fiscalizacion` FOREIGN KEY (`fiscalizacion_id`) REFERENCES `fis_fiscalizacion` (`id`) ON DELETE NO ACTION ON UPDATE NO ACTION,
  CONSTRAINT `fiscalizacionmesa_mesa` FOREIGN KEY (`mesa_id`) REFERENCES `gen_mesa` (`id`) ON DELETE NO ACTION ON UPDATE NO ACTION
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


CREATE TABLE `fis_fiscal_mesa` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `user_id` int(11) NOT NULL,
  `fiscalizacion_mesa_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `fiscalmesa_user_idx` (`user_id`),
  KEY `fiscalmesa_fiscalizacionmesa_idx` (`fiscalizacion_mesa_id`),
  CONSTRAINT `fiscalmesa_fiscalizacionmesa` FOREIGN KEY (`fiscalizacion_mesa_id`) REFERENCES `fis_fiscalizacion_mesa` (`id`) ON DELETE NO ACTION ON UPDATE NO ACTION,
  CONSTRAINT `fiscalmesa_user` FOREIGN KEY (`user_id`) REFERENCES `gen_user` (`id`) ON DELETE NO ACTION ON UPDATE NO ACTION
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE `fis_fiscal_general` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `user_id` int(11) NOT NULL,
  `gen_local_comicio` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `fiscalgeneral_user_idx` (`user_id`),
  KEY `fiscalgeneral_localcomicio_idx` (`gen_local_comicio`),
  CONSTRAINT `fiscalgeneral_localcomicio` FOREIGN KEY (`gen_local_comicio`) REFERENCES `gen_local_comicio` (`id`) ON DELETE NO ACTION ON UPDATE NO ACTION,
  CONSTRAINT `fiscalgeneral_user` FOREIGN KEY (`user_id`) REFERENCES `gen_user` (`id`) ON DELETE NO ACTION ON UPDATE NO ACTION
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE `fis_fiscal_coordinador` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `user_id` int(11) NOT NULL,
  `circuito_id` int(11) DEFAULT NULL,
  `seccion_id` int(11) DEFAULT NULL,
  `distrito_id` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `coordinador_idx` (`circuito_id`),
  KEY `coordinador_seccion_idx` (`seccion_id`),
  KEY `coordinador_distrito_idx` (`distrito_id`),
  KEY `coordinador_user_idx` (`user_id`),
  CONSTRAINT `coordinador_circuito` FOREIGN KEY (`circuito_id`) REFERENCES `gen_circuito` (`id`) ON DELETE NO ACTION ON UPDATE NO ACTION,
  CONSTRAINT `coordinador_distrito` FOREIGN KEY (`distrito_id`) REFERENCES `gen_distrito` (`id`) ON DELETE NO ACTION ON UPDATE NO ACTION,
  CONSTRAINT `coordinador_seccion` FOREIGN KEY (`seccion_id`) REFERENCES `gen_seccion` (`id`) ON DELETE NO ACTION ON UPDATE NO ACTION,
  CONSTRAINT `coordinador_user` FOREIGN KEY (`user_id`) REFERENCES `gen_user` (`id`) ON DELETE NO ACTION ON UPDATE NO ACTION
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- -----------------------
-- fiscalizcion - centro computo
-- -----------------------
CREATE TABLE `fis_centro_computo` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `descripcion` varchar(100) NOT NULL,
  `reponsable_user_id` int(11) NOT NULL,
  `circuito_id` int(11) DEFAULT NULL,
  `seccion_id` int(11) DEFAULT NULL,
  `distrito_id` int(11) DEFAULT NULL,
  `telefono_recepcion_imagen` varchar(45) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `centrocomputo_responsable_idx` (`reponsable_user_id`),
  KEY `centrocomputo_circuito_idx` (`circuito_id`),
  KEY `centrocomputo_seccion_idx` (`seccion_id`),
  KEY `centrocomputo_distrito_idx` (`distrito_id`),
  CONSTRAINT `centrocomputo_circuito` FOREIGN KEY (`circuito_id`) REFERENCES `gen_circuito` (`id`) ON DELETE NO ACTION ON UPDATE NO ACTION,
  CONSTRAINT `centrocomputo_distrito` FOREIGN KEY (`distrito_id`) REFERENCES `gen_distrito` (`id`) ON DELETE NO ACTION ON UPDATE NO ACTION,
  CONSTRAINT `centrocomputo_responsable` FOREIGN KEY (`reponsable_user_id`) REFERENCES `gen_user` (`id`) ON DELETE NO ACTION ON UPDATE NO ACTION,
  CONSTRAINT `centrocomputo_seccion` FOREIGN KEY (`seccion_id`) REFERENCES `gen_seccion` (`id`) ON DELETE NO ACTION ON UPDATE NO ACTION
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE `fis_centro_computo_auditor` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `centro_computo_id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `centrocomputoauditor_centrocomputo_idx` (`centro_computo_id`),
  KEY `centrocomputoauditor_user_idx` (`user_id`),
  CONSTRAINT `centrocomputoauditor_centrocomputo` FOREIGN KEY (`centro_computo_id`) REFERENCES `fis_centro_computo` (`id`) ON DELETE NO ACTION ON UPDATE NO ACTION,
  CONSTRAINT `centrocomputoauditor_user` FOREIGN KEY (`user_id`) REFERENCES `gen_user` (`id`) ON DELETE NO ACTION ON UPDATE NO ACTION
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE `fis_centro_computo_auditoria_acta_fiscal_estado` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `descripcion` varchar(100) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE `fis_centro_computo_auditoria_acta_correo_estado` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `descripcion` varchar(100) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE `fis_centro_computo_auditoria` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `createDate` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `auditor_id` int(11) NOT NULL,
  `mesa_id` int(11) NOT NULL,
  `acta_fiscal_estado_id` int(11) DEFAULT NULL,
  `acta_correo_estado_id` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `centrocomputoauditoria_auditor_idx` (`auditor_id`),
  KEY `centro_computo_auditoria_mesaid_idx` (`mesa_id`),
  KEY `centrocomputoauditoria_estadoacta_idx` (`acta_fiscal_estado_id`),
  KEY `centrocomputoauditoria_estadocorreoacta_idx` (`acta_correo_estado_id`),
  CONSTRAINT `centrocomputoauditoria_auditor` FOREIGN KEY (`auditor_id`) REFERENCES `fis_centro_computo_auditor` (`id`) ON DELETE NO ACTION ON UPDATE NO ACTION,
  CONSTRAINT `centrocomputoauditoria_estadoacta` FOREIGN KEY (`acta_fiscal_estado_id`) REFERENCES `fis_centro_computo_auditoria_acta_fiscal_estado` (`id`) ON DELETE NO ACTION ON UPDATE NO ACTION,
  CONSTRAINT `centrocomputoauditoria_estadocorreoacta` FOREIGN KEY (`acta_correo_estado_id`) REFERENCES `fis_centro_computo_auditoria_acta_correo_estado` (`id`) ON DELETE NO ACTION ON UPDATE NO ACTION,
  CONSTRAINT `centrocomputoauditoria_mesaid` FOREIGN KEY (`mesa_id`) REFERENCES `gen_mesa` (`id`) ON DELETE NO ACTION ON UPDATE NO ACTION
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;



/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
