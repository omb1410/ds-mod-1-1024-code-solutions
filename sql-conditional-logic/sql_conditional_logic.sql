USE telecom;


SELECT id, log_feature, volume
from log_feature lf; 

CREATE TABLE dsstudent.volume_1
SELECT id, log_feature, volume,
		CASE 
			when volume < 100 then "low"
			when volume > 500 then "large"
			else "medium"
		END volume_1
from log_feature lf;

SELECT volume_1, COUNT(*) value_counts
FROM dsstudent.volume_1
GROUP BY volume_1;

use hr;
SELECT EmployeeNumber, HourlyRate,
		CASE 
			when HourlyRate >= 80 then "high hourly rate"
			when HourlyRate < 40 then "low hourly rate"
			else "medium hourly rate"	
		END HourlyRate_1
from employee;


SELECT Gender,
		CASE 
			when Gender = "Female" then "0"
			when Gender = "Male" then "1"	
		END Gender_1
from employee;		


