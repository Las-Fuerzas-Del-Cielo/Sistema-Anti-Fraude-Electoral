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

-- --------------------------------------------------------

-- -------
-- Estado de los usuarios 
-- -------
CREATE TABLE `gen_user_status` (
  `id` int(11) NOT NULL,
  `description` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- -------
-- Niveles de usuarios para seguridad (alternativa a Rol)
-- -------
CREATE TABLE `gen_user_level` (
  `id` int(11) NOT NULL,
  `description` varchar(45) NOT NULL,
  `enabled` tinyint(4) NOT NULL DEFAULT '1',
  PRIMARY KEY (`id`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- -------
-- Usuarios, con firebaseId
-- -------
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
) ENGINE=MyISAM AUTO_INCREMENT=34 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- -------
-- Rol
-- -------
CREATE TABLE `sec_rol` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `description` varchar(45) NOT NULL,
  `create_date` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `create_user_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `rol_user_idx` (`create_user_id`),
  CONSTRAINT `rol_user` FOREIGN KEY (`create_user_id`) REFERENCES `gen_user` (`id`)
) ENGINE=MyISAM AUTO_INCREMENT=11 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- -------
-- Privilegio
-- -------
CREATE TABLE `sec_privilege` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `description` varchar(45) NOT NULL,
  `create_date` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `create_user_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `privilege_user_idx` (`create_user_id`),
  CONSTRAINT `privilege_user` FOREIGN KEY (`create_user_id`) REFERENCES `gen_user` (`id`)
) ENGINE=MyISAM AUTO_INCREMENT=3 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- -------
-- Rol Privilegio
-- -------
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
) ENGINE=MyISAM AUTO_INCREMENT=6 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;


-- -------
-- Relacion usuario Rol
-- -------
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
) ENGINE=MyISAM AUTO_INCREMENT=28 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;



/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
