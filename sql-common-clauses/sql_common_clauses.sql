USE telecom;

SELECT id, log_feature AS log, volume AS vol
FROM log_feature lf; 

SELECT id, resource_type
FROM resource_type rt 
ORDER BY id ASC
LIMIT 5;

SELECT id, resource_type
FROM resource_type rt 
ORDER BY id DESC
LIMIT 5;

SELECT id, resource_type
FROM resource_type rt 
ORDER BY id ASC, resource_type DESC
LIMIT 5;

SELECT COUNT(*) AS numbers_row, 
COUNT(DISTINCT(id)) AS id_unique, 
COUNT(DISTINCT(severity_type)) AS severity_type_unique
FROM severity_type st;

SELECT id, log_feature, volume
from log_feature lf 
WHERE log_feature = "feature 201"
AND volume BETWEEN 100 AND 300
ORDER BY volume ASC;
