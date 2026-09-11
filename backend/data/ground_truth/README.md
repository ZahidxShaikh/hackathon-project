# GSI manganese exploration evidence

Source: Geological Survey of India (GSI), NGDR/Bhu-Chayan report
`20260724162936.935_CR_Final_Report_49367_2024_25_JBP.pdf`, NUID 49367,
*Reconnaissance survey for manganese in Sausar Group of rocks in Jogitola,
Tikari and Dongargaon areas, Balaghat district, Madhya Pradesh* (G4 stage,
field season 2024-25).

Coordinates are extracted exactly from Annexure I-IV report tables and written
as EPSG:4326 decimal-degree Points. BRS means Bed Rock Samples; PTS means
pitting/trenching samples; PCS and PS are the report's petrochemical and
petrological sample categories, respectively. MnO is included only when it
could be parsed from the report's Annexure VI-VIII analytical tables; `null`
means no reliable parsed value, not zero.

Validation uses the report-declared block boundary (21°39'10"–21°45'00" N,
79°37'45"–79°44'45" E). Points outside it are retained and flagged
`out_of_study_area`; suspicious values are never repaired. Repeated coordinate
pairs are retained because the report assigns distinct sample IDs, commonly
representing separate samples or intervals at the same location.

Important: the report block is around 79.63–79.75°E, while the current
satellite AOI is 80.10–80.30°E. Therefore this evidence does not intersect the
current raster AOI and must not be used for pixel-level training there without
changing/reprocessing the imagery AOI. It is field/exploration evidence, **not
proof of economically mineable reserves**.

