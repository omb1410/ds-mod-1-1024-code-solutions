/* sql-create-insert-delete exercise */

USE dsstudent;

SHOW TABLES;

CREATE TABLE person_Toan
			(person_id INT,
			first_name VARCHAR(20),
			last_name VARCHAR(25), 
			city VARCHAR(20),
			CONSTRAINT pk_person_Toan PRIMARY KEY (person_id));
			

SELECT * FROM person_Toan;

DELETE FROM person_Toan
WHERE person_id = 1;

INSERT INTO person_Toan 
			(person_id, first_name, last_name, city)
VALUES 
	(1, "Toan", "Tran", "Fountain Valley");

INSERT INTO person_Toan 
			(person_id, first_name, last_name, city)
VALUES 
	(2, "Vicky", "Bui", "Santa Ana"),
	(3, "Vekn", "Abdelmasih", "Westminster");

ALTER TABLE person_Toan
ADD gender VARCHAR(1);

UPDATE person_Toan 
SET gender = "M";

UPDATE person_Toan 
SET gender = "F"
WHERE person_id = 2;

ALTER TABLE person_Toan
DROP COLUMN gender;

DELETE FROM person_Toan 
where person_id = 2;

DROP TABLE person_Toan;


