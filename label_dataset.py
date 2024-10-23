'''
@brief a quick script for automating the generation of a labelled csv file indicating the maliciousness of a file

@author Sean Kannanaikal (stk5106)
'''

#necessary imports
import os
import pandas as pd
import csv

#set of file locations that can be manipulated for each unique running environment
GLOBAL_LABELS_FILE = "/home/stk5106/raw_byte_classifier/dataset/labels.csv" #csv file which will be populated
GLOBAL_BENIGN_EXECUTABLES_LOCATION = "/home/stk5106/raw_byte_classifier/dataset/benign/" #folder containing benign PEs
GLOBAL_MALICIOUS_EXECUTABLES_LOCATION = "/home/stk5106/raw_byte_classifier/dataset/malware" #folder containing malicious PEs


'''
@brief this is the main loop which will go in and fill out the csv file
'''
def main():
	#referencing global variables for important file locations
	global GLOBAL_LABELS_FILE
	global GLOBAL_BENIGN_EXECUTABLES_LOCATION
	global GLOBAL_MALICIOUS_EXECUTABLES_LOCATION

	#obtaining directory listings of all the malicous and beningn files for labelling
	benignExecutableNames = os.listdir(GLOBAL_BENIGN_EXECUTABLES_LOCATION)
	maliciousExecutableNames = os.listdir(GLOBAL_MALICIOUS_EXECUTABLES_LOCATION)

	print('[+] Identified Listing of all Benign and Malicous Executables')

	with open(GLOBAL_LABELS_FILE, "w", newline='') as datasetLabels:
		#making csv writer
		writer = csv.writer(datasetLabels)

		#creating header
		#headers = ['name', 'score']
		#writer.writerow(headers)
		
		#labelling and entering all entries for benign executables
		for executable in benignExecutableNames:
			writer.writerow([f'{executable}', 0])

		print('[+] Added all Benign Executables into CSV')

		#labelling and entering all entries for malicious executables
		for executable in maliciousExecutableNames:
			writer.writerow([f'{executable}', 1])

		print('[+] Added all Malicious Executables into CSV')

	print('[+] Completed Creating CSV File')

if __name__ == "__main__":
	main()
