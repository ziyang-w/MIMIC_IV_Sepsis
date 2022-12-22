--target:
-- create table last_in_icu
-- create table icudetail 
--result: 53152 records

--选择最后一次进icu时患者的stay_id
drop table if EXISTS mimic_my.last_in_icu;

create table last_in_icu as 
-- 选择每一位患者的患者的最大入icu时间，并以此作为
with pa as(
	select pa.subject_id,"max"(icu_intime) as last_icu_intime
	from mimic_derived.icustay_detail pa
	group by pa.subject_id
)
	select st.subject_id,st.stay_id,st.icu_intime
	from mimic_derived.icustay_detail st
	where st.icu_intime in (select pa.last_icu_intime from pa)
;

--根据last_in_icu查找出来的stay_id编号对应最后一次的具体信息。
drop table if exists mimic_my.icudetail;
create table mimic_my.icudetail AS 
	SELECT icudetail.*
	from mimic_derived.icustay_detail icudetail
	where icudetail.stay_id in (select last_in_icu.stay_id from mimic_my.last_in_icu);

-- sepsis_last_icu
drop table if exists mimic_my.sepsis_last_icu;
create table mimic_my.sepsis_last_icu as 
select 
	sps.*,
	datetime_diff(sps.antibiotic_time,sps.suspected_infection_time,'DAY') as sus_anti_period,
	datetime_diff(sps.antibiotic_time,sps.culture_time,'DAY') as cul_anti_period,
	gender, dod,ethnicity, 
	-- hospital level factors
	los_hospital, 
	hospital_expire_flag,
	first_hosp_stay,
	los_icu,
	first_icu_stay,
	CASE
    WHEN  los_icu<=30 and hospital_expire_flag = 1 THEN 1
    ELSE 0 END AS death_within_30_days
from mimic_my.icudetail
inner join mimic_derived.sepsis3 sps
on sps.subject_id = icudetail.subject_id
and sps.stay_id = icudetail.stay_id
;

select count(subject_id) from mimic_my.icudetail;	--53152
select count(subject_id) from mimic_my.sepsis_last_icu; --23656

-- sepsis_refine
drop table if exists mimic_my.sepsis_refine;
create table mimic_my.sepsis_refine as 
select * from mimic_my.sepsis_features sps
where sps.age_mean>18 and sps.age_mean<89
and sps.los_icu>1
;