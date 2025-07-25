
MySQL表單建立語法

CREATE TABLE `qa_table` (
  `id` int NOT NULL AUTO_INCREMENT,
  `session_id` varchar(80) DEFAULT NULL,
  `lib` varchar(45) DEFAULT NULL,
  `question` varchar(200) DEFAULT NULL,
  `answer` varchar(500) DEFAULT NULL,
  `thumbs` varchar(10) DEFAULT NULL,
  `day_time` varchar(20) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `id_UNIQUE` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=156 DEFAULT CHARSET=utf8mb3;
