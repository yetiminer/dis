import os
import pandas as pd

class importData(object):



	def get_folders(home_dir):
	#get the names of the folders of the data

		folders=[]
		for dirg in os.listdir(home_dir):
			if os.path.isdir(os.path.join(home_dir,dirg)) and dirg[0]=='2' :

					folders.append(dirg)
		return folders
		
	def mem_usage(pandas_obj):
		if isinstance(pandas_obj,pd.DataFrame):
			usage_b = pandas_obj.memory_usage(deep=True).sum()
		else: # we assume if not a df it's a series
			usage_b = pandas_obj.memory_usage(deep=True)
		usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
		return "{:03.2f} MB".format(usage_mb)

		
	def categorizeDF(columnlist,df):
		#converts a list of columns into category type for pandas df
		for col in columnlist:
			try:
				df[col]=df[col].astype('category')
			except NotImplementedError:
				print(col, ' only one datum ')
              
		return df
		

	def importSEC(home_dir,report, filters,import_cols,categorical_cols,folders):
		#imports and amalgamtes data from separate files in separate folders
		k=1
		report=report+'.txt'
		for dirname in folders:
		
			target_dir=os.path.join(home_dir,dirname,report)
			
			try:
				z=pd.read_table(target_dir,encoding = "ISO-8859-1",usecols=import_cols, low_memory=False)


				for fil in filters:

					z=z[z[fil].isin(filters[fil])]


				z=categorizeDF(categorical_cols,z)

				if k==1:
					allz=z
					k=k+1
				else:
					k=k+1
					allz=allz.append(z)
					#allz=categorizeDF(categorical_cols,allz)
					print(z.shape,dirname,round((k-1)/len(folders),2))
			except ValueError:
				print(dirname, 'error')
		
		print(mem_usage(allz))        
		allz=categorizeDF(categorical_cols,allz)
		print(mem_usage(allz))
		print(allz.shape)
		return allz
		
