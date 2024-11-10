use dsstudent;

#1
CREATE TABLE customer_toan
				(customer_id smallINT,
				name VARCHAR(20),
				location VARCHAR(20),
				total_expenditure VARCHAR(20)
				);
			
DESCRIBE customer_toan;

#2
INSERT INTO customer_toan 
			(customer_id, name, location, total_expenditure)
VALUES
(1701, "John", "Newport Beach, CA", 2000 ),
(1707, "Tracy", "Irvine, CA", 1500),
(1711, "Daniel", "Newport Beach, CA", 2500),
(1703, "Ella", "Santa Ana, CA", 1800), 
(1708, "Mel", "Orange, CA", 1700),
(1716, "Steve", "Irvine, CA", 18000);

SELECT *
FROM customer_toan ct;

#3
UPDATE customer_toan 
SET total_expenditure = 1800
WHERE customer_id = 1716;

#4
ALTER TABLE customer_toan 
ADD gender VARCHAR(20);

#5
UPDATE customer_toan 
SET gender = "M"
WHERE customer_id IN (1701, 1711, 1716);

UPDATE customer_toan 
SET gender = "F"
WHERE customer_id IN (1707, 1703, 1708);

#6
DELETE FROM customer_toan 
where customer_id = 1716;

#7
ALTER TABLE customer_toan 
ADD store VARCHAR(20);	

#8
ALTER TABLE customer_toan 
DROP column store;

#9
SELECT *
FROM customer_toan ct;

#10
SELECT name, total_expenditure
FROM customer_toan ct;

#11
SELECT name AS n, total_expenditure AS total_exp
FROM customer_toan ct;

#12
ALTER TABLE customer_toan 
MODIFY COLUMN total_expenditure smallint;

#13 
SELECT total_expenditure 
FROM customer_toan ct 
ORDER BY total_expenditure DESC;

#14
SELECT name, total_expenditure 
FROM customer_toan ct 
ORDER BY total_expenditure DESC
LIMIT 3;

#15
SELECT COUNT(DISTINCT(location)) AS nuniques
FROM customer_toan ct;

#16 
SELECT DISTINCT(location) as unique_cities
FROM customer_toan;

#17 
SELECT *
FROM customer_toan
WHERE gender = "M";

#18
SELECT *
FROM customer_toan
WHERE gender = "F";

#19 
SELECT *
FROM customer_toan
WHERE location = "Irvine, CA";

#20
SELECT name, location
FROM customer_toan ct
WHERE total_expenditure < 2000
ORDER BY name ASC;

#21 
DROP TABLE customer_toan;




