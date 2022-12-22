-- select spesis patients features:
-- 1. Age, weight, gender, height
-- 2. 

DROP TABLE IF EXISTS mimic_my.sepsis_features;
CREATE TABLE mimic_my.sepsis_features AS 

--all sepsis patients with gender,age,heitht,weight 

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

-- blood lab
-- 根据last_in_icu.stay_id查找对应的blood lab
flb as(
	select flb.*
	from mimic_derived.first_day_lab flb
	inner join mimic_my.last_in_icu 
	on flb.stay_id = mimic_my.last_in_icu.stay_id
-- 	WHERE  in (select last_in_icu.stay_id from mimic_my.last_in_icu)
)	
	

SELECT
	sps.*,
	
	tb1.age_mean,
	tb2.height_mean,
	
	--flb
		-- complete blood count
		 flb.hematocrit_min, flb.hematocrit_max
		, flb.hemoglobin_min, flb.hemoglobin_max
		, flb.platelets_min, flb.platelets_max
		, flb.wbc_min, flb.wbc_max
		-- chemistry
		, flb.albumin_min, flb.albumin_max
		, flb.globulin_min, flb.globulin_max
		, flb.total_protein_min, flb.total_protein_max
		, flb.aniongap_min, flb.aniongap_max
		, flb.bicarbonate_min, flb.bicarbonate_max
		, flb.bun_min, flb.bun_max
		, flb.calcium_min, flb.calcium_max
		, flb.chloride_min, flb.chloride_max
		, flb.creatinine_min, flb.creatinine_max
		, flb.glucose_min, flb.glucose_max
		, flb.sodium_min, flb.sodium_max
		, flb.potassium_min, flb.potassium_max
		-- blood differential
		, flb.abs_basophils_min, flb.abs_basophils_max
		, flb.abs_eosinophils_min, flb.abs_eosinophils_max
		, flb.abs_lymphocytes_min, flb.abs_lymphocytes_max
		, flb.abs_monocytes_min, flb.abs_monocytes_max
		, flb.abs_neutrophils_min, flb.abs_neutrophils_max
		, flb.atyps_min, flb.atyps_max
		, flb.bands_min, flb.bands_max
		, flb.imm_granulocytes_min, flb.imm_granulocytes_max
		, flb.metas_min, flb.metas_max
		, flb.nrbc_min, flb.nrbc_max
		-- coagulation
		, flb.d_dimer_min, flb.d_dimer_max
		, flb.fibrinogen_min, flb.fibrinogen_max
		, flb.thrombin_min, flb.thrombin_max
		, flb.inr_min, flb.inr_max
		, flb.pt_min, flb.pt_max
		, flb.ptt_min, flb.ptt_max
		-- enzymes and bilirubin
		, flb.alt_min, flb.alt_max
		, flb.alp_min, flb.alp_max
		, flb.ast_min, flb.ast_max
		, flb.amylase_min, flb.amylase_max
		, flb.bilirubin_total_min, flb.bilirubin_total_max
		, flb.bilirubin_direct_min, flb.bilirubin_direct_max
		, flb.bilirubin_indirect_min, flb.bilirubin_indirect_max
		, flb.ck_cpk_min, flb.ck_cpk_max
		, flb.ck_mb_min, flb.ck_mb_max
		, flb.ggt_min, ggt_max
		, flb.ld_ldh_min, flb.ld_ldh_max
	
FROM mimic_my.sepsis_last_icu sps
LEFT JOIN tb1 ON tb1.subject_id = sps.subject_id
LEFT JOIN tb2 on tb2.stay_id = sps.stay_id
LEFT join mimic_derived.first_day_lab flb on flb.stay_id = sps.stay_id;


--返回新建表中的记录数量		
select count(subject_id) from mimic_my.sepsis_features;  --23656