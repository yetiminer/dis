import pandas as pd
import os
import argparse
import yaml

class importDataTools(object):

	def __init__(self, name):
		self.folders=[]
		self.name=name
		self.import_cols=[]
		self.cat_cols={}
		self.df=pd.DataFrame()
		self.report=[]
		self.filters={}
		self.home_dir=[]
		self.memusage=[]
		self.mem_disp=False


	def add_folders(self):
	#get the names of the folders of the data

		folders=[]
		for dirg in os.listdir(self.home_dir):
			if os.path.isdir(os.path.join(self.home_dir,dirg)) and dirg[0]=='2' :

					folders.append(dirg)
		self.folders=folders
			
	def add_report(self,report):
		self.report=report
		
	def add_cat_cols(self,cat_cols):
		self.cat_cols=cat_cols
             
		
	def add_filters(self,filters):
		assert isinstance(filters,dict)
		for fil in filters:
			assert isinstance(filters[fil], (list, tuple,object)) 
			#assert not isinstance(fil, basestring)
		self.filters=filters
	
	def mem_usage(self):
		if isinstance(self.df,pd.DataFrame):
			usage_b =self.df.memory_usage(deep=True).sum()
		else: # we assume if not a df it's a series
			usage_b = self.df.memory_usage(deep=True)
		usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
		memusage="{:03.2f} MB".format(usage_mb)
		print(memusage)
		self.memusage=memusage


	def categorizeDF(self):
		#converts a list of columns into category type for pandas df
		for col in self.cat_cols:
			try:
				self.df[col]=self.df[col].astype(self.cat_cols[col])
			except NotImplementedError:
				print(col, ' only one datum ')


	def filterdf(self,z):
		for fil in self.filters:
			z=z[z[fil].isin(self.filters[fil])]
		return z

	def importSEC(self):
		#imports and amalgamtes data from separate files in separate folders
		k=1
		report=self.report+'.txt'
		for dirname in self.folders:
            
			target_dir=os.path.join(self.home_dir,dirname,report)
			
			try:
				z=pd.read_table(target_dir,encoding = "ISO-8859-1",usecols=self.import_cols, low_memory=False)
			except ValueError:
				print(dirname, 'error')

			z=self.filterdf(z)

			#z=categorizeDF()

			if k==1:
				self.df=z
				k=k+1
			else:
				k=k+1
				self.df=self.df.append(z)
				#allz=categorizeDF(categorical_cols,allz)
				print(z.shape,dirname,round((k-1)/len(self.folders),2))

		
		if self.mem_disp: self.mem_usage()       
		self.categorizeDF()
		if self.mem_disp: self.mem_usage()
		print(self.df.shape)
		
def SecLoader(cfg,name,filters=None,mem_disp=False):
	allsubz=importDataTools(name)
	allsubz.home_dir=cfg['home_dir']
	allsubz.import_cols=(cfg['import_cols'])
	allsubz.add_cat_cols(cfg['cat_cols'])
	allsubz.add_folders()
	allsubz.report=cfg['report']
	allsubz.mem_disp=mem_disp
	if filters==None:
		try:
			allsubz.add_filters(cfg['filters'])
		except KeyError:
				print("no filters on import")
	else:
		allsubz.add_filters(filters)
	allsubz.importSEC()

	return allsubz
	
def yamlLoad(path):
	
	with open(path, 'r') as stream:
		try:
			cfg=yaml.load(stream)
		except yaml.YAMLError as exc:
			print(exc)
	return cfg
	
# def get_parser():
    # import argparse

    # parser = argparse.ArgumentParser(description="Tree calculation and plotting")
    # # parser.add_argument("-l","--left",
                         # # default=-0.2, type=float,
                        # # help="How much left branch rotates at fork (negative) default -0.2")
    # # parser.add_argument("-r","--right",
                         # # default=0.2, type=float,
                        # # help="How much right branch rotates at fork default 0.2")
    # # parser.add_argument("-bs","--shrink",
                         # # default=0.6, type=float,
                        # # help="How much branches shrink at each layer default 0.6")
    # # parser.add_argument("-s","--splits",
                         # # default=5, type=int,
                        # # help="How many layers of tree")
    # # parser.add_argument("-p","--draw",
                         # # default=True, type=str2bool,nargs='?', const=True,
                        # # help="Whether to plot picture or not default True")
    # # parser.add_argument("-c","--config",
                         # # default="TreeConfig.yaml", type=str,
                        # # help="Config file containing root coordinates of tree, first split and initial branch  length")
    # # parser.add_argument("--check",
                         # # default=False, type=str2bool,nargs='?', const=True,
                        # # help="Enable the inbuilt output checker")

    # return parser

# if __name__=='__SecLoader__'
