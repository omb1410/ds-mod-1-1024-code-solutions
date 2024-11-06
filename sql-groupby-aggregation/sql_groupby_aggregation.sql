USE telecom;

CREATE TEMPORARY TABLE dsstudent.joint
SELECT t.id, location, fault_severity, event_type, severity_type, resource_type, log_feature, volume
FROM train t
	LEFT OUTER JOIN log_feature lf
	ON t.id = lf.id
	LEFT OUTER JOIN event_type et 
	ON t.id = et.id
	LEFT OUTER JOIN resource_type rt 
	ON t.id = rt.id 
	LEFT OUTER JOIN severity_type st 
	on t.id = st.id;

SELECT location, COUNT(DISTINCT(event_type)) num_unique_event_type
FROM dsstudent.joint
GROUP BY location
ORDER BY location ASC;

SELECT location, SUM(volume) total_volume
FROM dsstudent.joint
GROUP BY location
ORDER BY total_volume DESC
LIMIT 3;

SELECT fault_severity, COUNT(DISTINCT(location)) num_of_unique_location
FROM dsstudent.joint
GROUP BY fault_severity;

SELECT fault_severity, COUNT(DISTINCT(location)) num_of_unique_location
FROM dsstudent.joint
GROUP BY fault_severity
HAVING fault_severity > 1;

USE hr;

SELECT Attrition, MIN(Age) min_age, MAX(Age) max_age, AVG(Age) avg_age
FROM employee e
GROUP BY Attrition;

SELECT Attrition, Department, COUNT(*) num_quantity
FROM employee e
GROUP BY Attrition, Department
HAVING Attrition IS NOT NULL
ORDER BY Attrition, Department ASC;

SELECT Attrition, Department, COUNT(*) num_quantity
FROM employee e
GROUP BY Attrition, Department
HAVING num_quantity > 100 AND Attrition IS NOT NULL
ORDER BY Attrition, Department ASC;
