form directory
	sentence folder: C:\Users\Дмитрий\Desktop\acoustic model 2nd year project\mozilla\mozilla
	sentence File_name_or_initial_substring
	sentence File_extension .wav
endform

Create Strings as directory list: "directoryList", "C:\Users\Дмитрий\Desktop\acoustic model 2nd year project\mozilla\mozilla"
head_words = selected("Strings")
nof = Get number of strings

for cd from 1 to nof
	#select Strings list
	dirname$ = Get string... cd
	Create Strings as file list... list 'folder$'/'dirname$'/'file_name_or_initial_substring$'*'file_extension$'
	file_words = selected("Strings")
	file_count = Get number of strings
	
	for current_file from 1 to file_count
   		select Strings list
   		filename$ = Get string... current_file
   		Read from file... 'folder$'/'dirname$'/'filename$'
	endfor
		
	#select file_st
	#Remove
	appendInfoLine: dirname$
	
endfor

select 'head_words'
Remove
