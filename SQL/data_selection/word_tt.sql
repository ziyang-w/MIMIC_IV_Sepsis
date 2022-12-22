	
DROP TABLE IF EXISTS mimic_my.sepsis_fianl;


CREATE TABLE mimic_my.sepsis_fianl AS 
WITH t2 AS ( 
	SELECT first_day_urine_output.stay_id, MAX ( first_day_urine_output.urineoutput ) AS urine_max 
	FROM mimic_derived.first_day_urine_output 
	GROUP BY stay_id 
),


T3 AS ( 
	SELECT stay_id,
		spo2_max, spo2_min, spo2_mean,
		sbp_max, sbp_min, sbp_mean,
		dbp_max, dbp_min, dbp_mean,
		heart_rate_max,
		resp_rate_max,
		temperature_max,temperature_min
		 
	FROM mimic_derived.first_day_vitalsign T3
),
T4 AS (
	 SELECT MAX(charlson_comorbidity_index) as charlson_score_max,
	 MAX ( charlson.metastatic_solid_tumor ) AS metastatic_solid_tumor_max, 
	 charlson.subject_id 
	 FROM mimic_derived.charlson 
	 GROUP BY subject_id 
 ),
T5 AS ( 
	SELECT MAX(sapsii) AS sapsii,MAX(sapsii_prob) AS sapsii_prob_max, sapsii.stay_id 
	FROM mimic_derived.sapsii 
	GROUP BY stay_id 
),
vent as (
	SELECT stay_id,
	"max"(
		case 
			when vent.ventilation_status = 'Trach' then 5
			when vent.ventilation_status = 'InvasiveVent' then 4
			when vent.ventilation_status = 'NonInvasiveVent' then 3
			when vent.ventilation_status = 'HighFlow' then 2
			when vent.ventilation_status = 'Oxygen' then 1
			else 0
		end
	) as vent_status
	from mimic_derived.ventilation vent
	GROUP BY stay_id
),
infection as (
	select stay_id,
	"count"(antibiotic) as antibiotic_num,
	"count"(specimen) as specimen_count,
	"max"(positive_culture ) as positive_culture
	from mimic_derived.suspicion_of_infection inflection
	group by stay_id
),
inflammation as(
	select max(crp) as crp_max,subject_id
	from mimic_derived.inflammation inflammation
	group by subject_id
)

SELECT
	mimic_my.sepsis_refine.subject_id,
	mimic_my.Sepsis_refine.stay_id,
	mimic_my.Sepsis_refine.age_mean,
	mimic_my.sepsis_refine.height_mean,
	mimic_my.sepsis_refine.los_icu,
	mimic_my.sepsis_refine.sofa_score,
	mimic_my.sepsis_refine.gender,
	mimic_my.sepsis_refine.los_hospital,
	mimic_my.Sepsis_refine.aniongap_max,
	mimic_my.Sepsis_refine.aniongap_min,
	mimic_my.sepsis_refine.bicarbonate_max,
	mimic_my.sepsis_refine.bicarbonate_min,
	mimic_my.Sepsis_refine.inr_max,	
	mimic_my.Sepsis_refine.sodium_max,
	mimic_my.Sepsis_refine.sodium_min,

	mimic_my.Sepsis_refine.chloride_max,
	mimic_my.Sepsis_refine.chloride_min,
	mimic_my.Sepsis_refine.bun_max,
	mimic_my.Sepsis_refine.bun_min,
	mimic_my.Sepsis_refine.wbc_max,
	mimic_my.Sepsis_refine.wbc_min,
	mimic_my.sepsis_refine.hematocrit_max,
	mimic_my.sepsis_refine.hematocrit_min,
	mimic_my.sepsis_refine.hemoglobin_max,
	mimic_my.sepsis_refine.hemoglobin_min,
	mimic_my.sepsis_refine.creatinine_max,
	mimic_my.sepsis_refine.creatinine_min,

	--T3
	T3.spo2_max, T3.spo2_min, T3.spo2_mean,
	T3.sbp_max, T3.sbp_min, T3.sbp_mean,
	T3.dbp_max, T3.dbp_min, T3.dbp_mean,
	T3.heart_rate_max,
	T3.resp_rate_max,
	T3.temperature_max,

	T2.urine_max,
	t4.metastatic_solid_tumor_max,
	t4.charlson_score_max,
	t5.sapsii,
	t5.sapsii_prob_max,
	
	vent.vent_status,
	
	sps.sus_anti_period,
	sps.cul_anti_period,
	
	infection.antibiotic_num,
	infection.specimen_count,
	infection.positive_culture,
	
	inflammation.crp_max,
	sps.death_within_30_days

FROM
	mimic_my.sepsis_refine
	left join sepsis_last_icu sps on mimic_my.sepsis_refine.stay_id = sps.stay_id
	LEFT JOIN T2 ON mimic_my.sepsis_refine.stay_id = T2.stay_id
	LEFT JOIN T3 ON mimic_my.sepsis_refine.stay_id = T3.stay_id
	LEFT JOIN T4 ON mimic_my.sepsis_refine.subject_id = T4.subject_id
	LEFT JOIN T5 ON mimic_my.sepsis_refine.stay_id = T5.stay_id
	left join vent on mimic_my.sepsis_refine.stay_id = vent.stay_id
	left join infection on mimic_my.sepsis_refine.stay_id = infection.stay_id
	left join inflammation on mimic_my.sepsis_refine.subject_id = inflammation.subject_id
;
SELECT
	"count" ( subject_id ) 
FROM
	mimic_my.tt2;