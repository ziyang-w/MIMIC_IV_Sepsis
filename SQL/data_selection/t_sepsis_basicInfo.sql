-- it contains the 35010 sepsis patients and their basic info:
-- 1. Age, weight, gender, height
-- 2. charlson (CCI)
-- 3. suspicion_of_infection, 首次怀疑感染的时间,
-- 4. icu_los and other features

DROP TABLE IF EXISTS mimic_my.sepsis_basicInfo;
CREATE TABLE mimic_my.sepsis_basicInfo AS 

--age
with tb1 AS (
	SELECT 
		tb1.subject_id,	avg(tb1.age) as age_mean
	FROM mimic_derived.age tb1
	group by subject_id
),
--height
tb2 as(
	select stay_id,avg(height) as height_mean
	from mimic_derived.height tb2
	group by stay_id
),
--weight
tb3 as(
	SELECT stay_id, "avg"(weight) as weight_mean
	FROM mimic_derived.weight_durations
	GROUP BY stay_id
),
--CCI
-- find stay_id to hadm_id pairs
tb4 as(
	SELECT icu.subject_id, icu.stay_id, icu.hadm_id, charlson_comorbidity_index,
		icu.first_careunit,icu.last_careunit
	FROM mimic_derived.charlson ch
	INNER JOIN mimic_icu.icustays icu
		ON icu.hadm_id = ch.hadm_id
),
--找到再ICU期间的培养物时间
micro as(
select 
	sb.subject_id, sb.hadm_id, microevent_id, org_itemid, spec_itemid, storedate, storetime,
	ab_name,ab_itemid,interpretation,charttime,
	case  
		when org_itemid is null then 0
		else 1		
	end as is_culture

from mimic_derived.icustay_detail sb
left join mimic_hosp.microbiologyevents mb
on sb.subject_id = mb.subject_id
and sb.hadm_id = mb.hadm_id
-- where mb.charttime >= datetime_sub(sb.icu_intime,INTERVAL '3' DAY)
-- and mb.charttime <= datetime_add(sb.icu_outtime,INTERVAL '1' DAY)
),
microAgg as (
select 
-- 	sb.subject_id, sb.hadm_id, 
	sb.stay_id,
	count(distinct org_itemid) as org_itemid_num,
	count(distinct spec_itemid) as sepc_itemid_num,
	count(distinct ab_name) as ab_name_num,
	max(storedate) as storedate_max,
	min(storedate) as storedate_min,
	max(is_culture) as is_culture,
	max(storetime) as storetime_max,
	max(storetime) as storetime_min --后面是关于生物的药敏时间,ab_name,ab_itemid,interpretation
from mimic_derived.icustay_detail sb
left join micro mb
on sb.subject_id = mb.subject_id
and sb.hadm_id = mb.hadm_id
-- where org_itemid is not null
group by stay_id
)

SELECT
	sps.subject_id,
	sps.stay_id,
	icu.hadm_id,
	gender, 
-- 	dod, --date of death 
	ethnicity, 
	--tb1.age_mean, 改用icudetail.admissions_age
	admission_age,
	tb2.height_mean,
	tb3.weight_mean,
	sps.sofa_score,
	tb4.charlson_comorbidity_index as charlson,
	-- hospital level factors
	first_hosp_stay,
	first_icu_stay,
	
	tb4.first_careunit,
	tb4.last_careunit,
	--suspicion_of_infection time
	antibiotic_time,
	sps.culture_time,
	sps.suspected_infection_time,
	datetime_diff(sps.antibiotic_time,sps.suspected_infection_time,'DAY') as sus_anti_period,
	datetime_diff(sps.antibiotic_time,sps.culture_time,'DAY') as cul_anti_period,
	icu.icu_intime, icu.icu_outtime,
	
	-- 药敏
	org_itemid_num,
	sepc_itemid_num,
	ab_name_num,
	storedate_max,
	storedate_min,
	storetime_max,
	storetime_min,
	is_culture,
	
	-- targetY
	los_icu,
	los_hospital, 
	hospstay_seq,
	hospital_expire_flag,
	

	case 
		when hospital_expire_flag=1 THEN
			case
				when icu.dod < icu.icu_outtime and icu.dod >icu.icu_intime then 1 --death in icu
				else 2 --death in hosp 
			end
	else 0
	END as death_in_icu,
	
	CASE
    WHEN  los_icu<=30 and hospital_expire_flag = 1 THEN 1
    ELSE 0 END AS death_within_30_days
from mimic_derived.sepsis3 sps
LEFT JOIN mimic_derived.icustay_detail icu
	ON sps.subject_id = icu.subject_id
	AND sps.stay_id = icu.stay_id
LEFT JOIN tb1
	ON tb1.subject_id = sps.subject_id
LEFT JOIN tb2
	ON tb2.stay_id = sps.stay_id
LEFT JOIN tb3
	ON tb3.stay_id = sps.stay_id
LEFT JOIN tb4
	ON tb4.stay_id = sps.stay_id
left join microAgg
	on microAgg.stay_id = sps.stay_id;
-- 	on microAgg.subject_id = icu.subject_id
-- 	and microAgg.hadm_id = icu.hadm_id
	
	
-- select subject_id,hadm_id from mimic_my.sepsis_basicinfo group by subject_id,hadm_id;
-- 
-- select icu_intime, datetime_add(icu_intime,INTERVAL '1' DAY) as ps from mimic_my.sepsis_basicinfo_spec where subject_id = 14294957 and hadm_id = 23884704
	
