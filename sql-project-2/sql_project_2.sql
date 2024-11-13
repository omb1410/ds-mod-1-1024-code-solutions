#1
CREATE TEMPORARY TABLE dsstudent.toandb
		(table_name VARCHAR(25),
		row_quantity INT);
	
INSERT INTO dsstudent.toandb
	(table_name, row_quantity)
SELECT
	TABLE_NAME,
	TABLE_ROWS AS row_quantity
FROM 
	INFORMATION_SCHEMA.TABLES
WHERE 
	TABLE_SCHEMA = 'loandb';

SELECT *
FROM dsstudent.toandb;

DROP TABLE IF EXISTS dsstudent.toandb;

SELECT *
FROM train;

#2
SELECT AMT_INCOME_TOTAL annual_income, (AMT_INCOME_TOTAL/12) monthly_income
FROM train;

#3
SELECT ROUND(DAYS_BIRTH/-365) age
FROM train;

#4
SELECT OCCUPATION_TYPE, COUNT(*) quantity 
FROM train 
WHERE OCCUPATION_TYPE IS NOT NULL
GROUP BY OCCUPATION_TYPE 
ORDER BY quantity DESC;

#5
SELECT DAYS_EMPLOYED,
			CASE
				WHEN DAYS_EMPLOYED > 1 THEN "bad data"
				ELSE "normal data"
			END flag_for_bad_data
FROM train;

#6
SELECT 
	TABLE_NAME,
	COLUMN_NAME,
	CONSTRAINT_NAME,
	REFERENCED_TABLE_NAME, 
	REFERENCED_COLUMN_NAME
FROM
	INFORMATION_SCHEMA.KEY_COLUMN_USAGE 
WHERE 
	REFERENCED_TABLE_SCHEMA = 'loandb';

DESC train;

SELECT t.TARGET, MIN(DAYS_INSTALMENT) min_day_installment, MAX(DAYS_INSTALMENT) max_day_installment, MIN(DAYS_ENTRY_PAYMENT) min_days_entry_payment, MAX(DAYS_ENTRY_PAYMENT) max_days_entry_payment
FROM installments_payments ip
	INNER JOIN credit_card_balance ccb
	ON ip.SK_ID_PREV = ccb.SK_ID_PREV
	INNER JOIN previous_application pa
	ON ip.SK_ID_PREV = pa.SK_ID_PREV
	INNER JOIN train t
	ON ip.SK_ID_CURR = t.SK_ID_CURR
GROUP BY t.Target;
	
	
			
		