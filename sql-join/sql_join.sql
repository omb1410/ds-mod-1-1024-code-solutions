use telecom;

SELECT 
	TABLE_NAME,
	COLUMN_NAME,
	CONSTRAINT_NAME,
	REFERENCED_TABLE_NAME, 
	REFERENCED_COLUMN_NAME
FROM
	INFORMATION_SCHEMA.KEY_COLUMN_USAGE 
WHERE 
	REFERENCED_TABLE_SCHEMA = 'telecom'; 

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

	
	
	