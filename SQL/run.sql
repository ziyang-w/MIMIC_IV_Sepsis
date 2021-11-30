-- NOTE: many scripts *require* you to use mimic_derived as the schema for outputting concepts
-- change the search path at your peril!
set search_path to mimic_derived, mimic_core, mimic_hosp, mimic_icu, mimic_ed;
\i postgres-functions.sql 
\i postgres-make-concepts.sql
