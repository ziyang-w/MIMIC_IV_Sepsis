drop table if exists mimic_my.sepsis_refine;
create table mimic_my.sepsis_refine as 

select sps.* ,icu.death_within_30_days,icu.sus_anti_period,
icu.cul_anti_period

from mimic_my.sepsis_features sps
left join mimic_my.sepsis_last_icu icu on icu.stay_id = sps.stay_id


where sps.age_mean>18 and sps.age_mean<89
and sps.los_icu>1