form directory
	real time_step 0
	real pitch_bottom 75
	real pitch_up 600
	real max_interval 0.02
	sentence source_path C:\praatoutput\test\wav
	sentence save_path C:\praatoutput\test\csv
endform
Create Strings as file list... files 'source_path$'/*.wav
head_words = selected("Strings")
file_count = Get number of strings

for file from 1 to file_count
	select Strings files
	filename$ = Get string... file
	Read from file... 'source_path$'/'filename$'
	name$ = selected$("Sound")
	To Pitch: time_step, pitch_bottom, pitch_up
	To PointProcess
	To PitchTier: max_interval
	Save as PitchTier spreadsheet file: "'save_path$'/'name$'.csv"
	
	select Sound 'name$'
	plus Pitch 'name$'
	plus PointProcess 'name$'
	plus PitchTier 'name$'
	Remove
endfor
pause Done!
