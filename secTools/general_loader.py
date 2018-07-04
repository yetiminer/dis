from secTools import yamlLoad, load_from_db, streamline, load_from_pickle
from load_data1 import SecTable, SECdataset
from sqlalchemy import create_engine, MetaData

def ds_from_db(**kwargs):

	cfg=kwargs['cfg']
	threshold=kwargs['tag_min_count_threshold']
	normalise_cols=kwargs['normalise_cols']
	
	if 'pickle_file' in kwargs:
		pickle_file=kwargs['pickle_file']
		ds=load_from_pickle(pickle_file)
		
	elif 'reimport_db' in kwargs:
		reimport_db=kwargs['reimport_db']
		db_location=kwargs['db_location']
		
		engine=create_engine('sqlite:///'+db_location, echo=True)
		data=load_from_db(engine)
		data=streamline(data,cfg)
		ds=SECdataset('ds',cfg,**data)
	
		ds.tag_count_plot(inplace=True)
		ds.feature_prune(thresh=threshold,inplace=True)

		ds.tag=ds.tag_prune(ds.tag)
		ds.pre=ds.tag_prune(ds.pre)
		ds.num=ds.tag_prune(ds.num)
	
		ds.create_feature_table(inplace=True)
		ds.normalise(inplace=True,cols=normalise_cols)
	
	return ds
	
	
